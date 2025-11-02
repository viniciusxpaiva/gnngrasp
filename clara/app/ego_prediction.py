import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import matthews_corrcoef

from app.nn.gnn.ego_protein_gnn import EgoProteinGNN
from data.blast.blast import BLAST
from app.utils.plotting import plot_training_loss_curve


class EgoPrediction:
    """
    Class to handle template selection, model initialization, training, and prediction.

    It loads node/edge features and global embeddings, builds and trains the GNN
    (GAT or basic), and generates binding site predictions using parameters from a
    centralized configuration dictionary.
    """

    def __init__(self, dirs, prediction_params):
        """
        Initialize the Prediction object with configured hyperparameters.

        Args:
            dirs (dict): Dictionary with predefined folder structure and file paths.
            prediction_params (dict): Dictionary containing model and training parameters,
                                      including architecture configuration and prediction threshold.
        """
        self.dirs = dirs
        self.prediction_params = prediction_params

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = "cpu"

        # Attributes to be initialized during dataset loading and model setup
        self.model = None
        self.loader = None
        self.train_data_class_weight = None
        self.node_embd_dim = None
        self.edge_prop_dim = None

    def prepare_ego_graph_dataset(self, input_protein):
        """
        For each selected template, build ego-graphs (one per residue) and prepare the full dataset.
        Each ego-graph will be a small subgraph centered on a residue with its neighbors.
        """
        print("[!] Preparing ego-graph dataset...")

        # Run BLAST
        blast = BLAST(self.dirs["data"]["blast"])
        selected_templates = blast.run(input_protein, 50)

        # --- DEBUG --- Print which templates will be processed
        print(f"[!] Selected {len(selected_templates)} templates")

        ego_graphs = []
        for tpt_id in selected_templates:
            # print(f"[DEBUG] Building ego-graphs for template: {tpt_id}")
            local_graphs = self._build_ego_graphs_from_template(tpt_id)
            # print(
            #    f"[DEBUG] {len(local_graphs)} ego-graphs created for template {tpt_id}"
            # )
            ego_graphs.extend(local_graphs)

        if not ego_graphs:
            raise ValueError("[ERROR] No ego-graphs found in selected templates.")

        self.node_embd_dim = ego_graphs[0].x.shape[1]
        self.edge_prop_dim = (
            ego_graphs[0].edge_attr.shape[1]
            if ego_graphs[0].edge_attr is not None
            else 0
        )

        self.loader = DataLoader(ego_graphs, batch_size=256, shuffle=True)
        print(
            f"[+] Loaded {len(ego_graphs)} ego-graphs from {len(selected_templates)} templates."
        )
        return self.loader

    def initialize_model_ego(self):
        """
        Initialize a single GNN model for training using the provided training data.
        This method creates the model architecture according to the specified configuration.
        """
        if not hasattr(self, "loader") or self.loader == None:
            raise ValueError(
                "[ERROR] Data loader not prepared. Please run prepare_dataset() first."
            )

        print(
            f"[!] Initializing GNN model: {self.prediction_params['layer_type']} "
            f"({self.prediction_params['num_gnn_layers']} layers)"
        )

        self.model = EgoProteinGNN(
            node_emb_in_channels=self.node_embd_dim,
            edge_in_channels=self.edge_prop_dim,
            hidden_channels=self.prediction_params["gnn_params"]["hidden_dim"],
            num_gnn_layers=self.prediction_params["num_gnn_layers"],
            layer_type=self.prediction_params["layer_type"],
            dropout=self.prediction_params["gnn_params"]["dropout"],
        ).to(self._device)

    def train_model_ego(self, epochs=100):
        """
        Train a GNN model for graph-level classification using ego-graphs.
        Uses MCC (Matthews Correlation Coefficient) to track the best-performing model.

        Args:
            epochs (int): Number of training epochs.
        """

        # --- Step 1: Validation checks ---
        if not hasattr(self, "model"):
            raise ValueError(
                "[ERROR] GNN model not initialized. Please call initialize_model() first."
            )
        if not hasattr(self, "loader") or self.loader is None:
            raise ValueError(
                "[ERROR] Dataset not loaded. Please run prepare_dataset() first."
            )

        print(f"[!] Starting training the GNN model | Device: {self._device}")
        loader = self.loader

        # --- Step 2: Compute class imbalance (binding vs non-binding ego-graphs) ---
        all_labels = torch.cat([data.y for data in loader.dataset])
        num_positive = (all_labels == 1).sum().item()
        num_negative = (all_labels == 0).sum().item()
        total = num_positive + num_negative

        print(
            f"[!] Training set: {num_positive} positive | "
            f"{num_negative} negative | {total} total ego-graphs."
        )

        class_weights = self._compute_weight_class(
            num_positive, num_negative, return_tensor=True
        ).to(self._device)

        print(
            f"[!] Class weights | 0 (non-binding): {class_weights[0]:.4f} | "
            f"1 (binding): {class_weights[1]:.4f}"
        )

        # --- Step 3: Setup optimizer and loss function ---
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.prediction_params["gnn_params"]["lr"],
            weight_decay=self.prediction_params["gnn_params"]["weight_decay"],
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # --- Step 4: Initialize training ---
        self.model.train()
        loss_history = []
        best_mcc = -1.0
        best_model_state = None

        # --- Step 5: Training loop over epochs ---
        for epoch in range(epochs):
            total_loss = 0
            true_labels = []
            pred_labels = []

            for batch in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                # --- Step 5a: Forward pass ---
                out = self.model(
                    batch.x,  # Node features
                    batch.edge_index,  # Edge indices
                    batch.edge_attr,  # Edge features (or None)
                    batch.batch,  # Mapping from node → graph
                )

                # --- Step 5b: Backward pass and optimization ---
                loss = criterion(out, batch.y)  # Labels are per-graph
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # --- Step 5c: Collect predictions for MCC evaluation ---
                preds = torch.argmax(out, dim=1).detach().cpu().numpy()
                labels = batch.y.detach().cpu().numpy()
                pred_labels.extend(preds)
                true_labels.extend(labels)

            # --- Step 6: Epoch summary ---
            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)

            mcc = matthews_corrcoef(true_labels, pred_labels)

            # --- Step 7: Update best model if MCC improves ---
            if mcc > best_mcc:
                best_mcc = mcc
                best_model_state = self.model.state_dict()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"[Epoch {epoch+1:>3}] Loss: {avg_loss:.4f} | MCC: {best_mcc:.3f}"
                )

        # --- Step 8: Load best model back into self.model ---
        self.model.load_state_dict(best_model_state)
        print(
            f"[✓] Best model (MCC={best_mcc:.4f}) loaded into memory for evaluation/prediction."
        )

        # --- Step 9: Save training loss curve ---
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"], "ego_training_loss_dashboard.html"
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def predict_binding_sites_ego(self, input_ego_graphs):
        """
        Predict binding sites for an input protein using ego-graphs and a trained GNN model.

        Args:
            input_ego_graphs (List[Data]): List of ego-graph Data objects (one per residue).

        Returns:
            pd.DataFrame: A DataFrame with residue_id, predicted_label, and binding_probability.
        """

        # --- Step 1: Check if model is trained ---
        if not hasattr(self, "model"):
            raise ValueError(
                "[ERROR] No trained model found. Please train the model first."
            )

        if not input_ego_graphs:
            raise ValueError("[ERROR] No ego-graphs provided for prediction.")

        # --- Step 2: Create DataLoader ---
        loader = DataLoader(input_ego_graphs, batch_size=256, shuffle=False)
        self.model.eval()

        threshold = self.prediction_params["prediction_threshold"]
        print(f"[!] Running predictions with threshold = {threshold:.2f}")

        all_preds = []
        all_probs = []
        all_residues = []

        # --- Step 3: Run predictions batch-wise ---
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)

                if "edge_attr" in batch:
                    edge_attr = batch.edge_attr
                else:
                    edge_attr = None
                    print(batch)
                    print("falhou")
                    exit()

                logits = self.model(batch.x, batch.edge_index, edge_attr, batch.batch)

                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
                preds = (probs >= threshold).long()

                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_residues.extend(batch.ego_center)

        # --- Step 4: Return predictions as DataFrame ---
        result_df = pd.DataFrame(
            {
                "residue_id": all_residues,
                "predicted_label": all_preds,
                "binding_probability": all_probs,
            }
        )

        # Round probability for readability
        result_df["binding_probability"] = result_df["binding_probability"].map(
            lambda x: round(x, 2)
        )

        return result_df

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _build_ego_graphs_from_template(
        self, template_id, num_layers=2, verbose=False, use_edge_features=True
    ):
        """
        For a given template, build ego-graphs (one per residue) with neighbors up to 'num_layers' hops.
        The edges in each ego-graph will only be those that actually exist between nodes in the global graph,
        as defined by the neighbors CSV file. Optionally includes edge features.

        Args:
            template_id (str): Template identifier, e.g., '1abc_A'.
            num_layers (int): Number of hops (layers) to include in the ego-graph.
            verbose (bool): Whether to print debug information about the graphs.
            use_edge_features (bool): Whether to attach edge features from the edge_properties file.

        Returns:
            list: List of PyG Data objects (ego-graphs for each residue).
        """

        if verbose:
            print(f"Building ego-graphs for {template_id}")

        # --- Step 1: Load files ---
        node_embd_path = os.path.join(
            self.dirs["data"]["embd_templates"]["node_embeddings"],
            f"{template_id}_node_embeddings.csv.zip",
        )

        edge_prop_path = os.path.join(
            self.dirs["data"]["prop_templates"]["edge_properties"],
            f"{template_id}_edge_properties.csv.zip",
        )

        neighbors_path = os.path.join(
            self.dirs["data"]["ego_templates"],
            f"{template_id}_neighbors.csv.zip",
        )

        if not (
            os.path.exists(node_embd_path)
            and os.path.exists(neighbors_path)
            and (not use_edge_features or os.path.exists(edge_prop_path))
        ):
            print(f"[WARNING] Missing files for {template_id}")
            return []

        node_df = pd.read_csv(node_embd_path, compression="zip")
        neighbors_df = pd.read_csv(neighbors_path, compression="zip")

        # Set of valid residues based on node_df
        valid_residues = set(node_df["residue_id"])

        # Filter out edges where either source or target is not in node_df
        if use_edge_features:
            edge_df = pd.read_csv(edge_prop_path, compression="zip")
            edge_df = edge_df[
                edge_df["source"].isin(valid_residues)
                & edge_df["target"].isin(valid_residues)
            ]

            edge_features_cols = [
                col for col in edge_df.columns if col not in ["source", "target"]
            ]
            edge_lookup = {
                (row["source"], row["target"]): row[edge_features_cols]
                .astype(float)
                .values
                for _, row in edge_df.iterrows()
            }

        # --- Step 2: Create residue and neighbor maps ---

        # Filter neighbors_df to only keep residue_ids present in node_df
        neighbors_df = neighbors_df[
            neighbors_df["residue_id"].isin(valid_residues)
        ].copy()

        # Also filter neighbors inside the "neighbors" column
        def filter_neighbors(neighbor_str):
            if pd.isna(neighbor_str):
                return ""
            neighbors = neighbor_str.split(",")
            filtered = [n for n in neighbors if n in valid_residues]
            return ",".join(filtered)

        neighbors_df["neighbors"] = neighbors_df["neighbors"].apply(filter_neighbors)
        neighbors_dict = {
            rid: (nbrs.split(",") if pd.notnull(nbrs) else [])
            for rid, nbrs in zip(neighbors_df["residue_id"], neighbors_df["neighbors"])
        }

        residue_map = {
            row.residue_id: {
                "embedding": row.drop(["residue_id", "label"]).astype(float).values,
                "label": row.label,
            }
            for _, row in node_df.iterrows()
        }

        ego_graphs = []

        # --- Step 3: Build ego-graphs ---
        for _, row in neighbors_df.iterrows():
            center_res = row["residue_id"]

            # --- BFS N-layer expansion ---
            visited = set([center_res])
            frontier = set([center_res])
            for _ in range(num_layers):
                next_frontier = set()
                for node in frontier:
                    for nbr in neighbors_dict.get(node, []):
                        if nbr in residue_map and nbr not in visited:
                            next_frontier.add(nbr)
                visited.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break

            ego_nodes = [center_res] + sorted(n for n in visited if n != center_res)
            ego_node_set = set(ego_nodes)
            node_to_idx = {n: i for i, n in enumerate(ego_nodes)}

            edge_index = []
            edge_attr = []

            for src in ego_nodes:
                for tgt in neighbors_dict.get(src, []):
                    if src in ego_node_set and tgt in ego_node_set:
                        i, j = node_to_idx[src], node_to_idx[tgt]

                        # Avoid duplicated undirected edges
                        if i <= j:
                            edge_index.append([i, j])
                            edge_index.append([j, i])
                            if use_edge_features:
                                # Add both directions if available
                                edge_attr.append(
                                    edge_lookup.get(
                                        (src, tgt), np.zeros(len(edge_features_cols))
                                    )
                                )
                                edge_attr.append(
                                    edge_lookup.get(
                                        (tgt, src), np.zeros(len(edge_features_cols))
                                    )
                                )

            edge_index = (
                torch.tensor(edge_index, dtype=torch.long).t()
                if edge_index
                else torch.empty((2, 0), dtype=torch.long)
            )
            edge_attr = (
                torch.tensor(np.array(edge_attr), dtype=torch.float)
                if use_edge_features and edge_attr
                else None
            )

            x = torch.tensor(
                np.array([residue_map[n]["embedding"] for n in ego_nodes]),
                dtype=torch.float,
            )
            y = torch.tensor([residue_map[center_res]["label"]], dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                ego_center=center_res,
                template_id=template_id,
                ego_nodes=ego_nodes,
            )
            ego_graphs.append(data)

        if verbose:
            self._print_ego_graph_debug_info(ego_graphs)
            print(
                f"[DEBUG] Finished building {len(ego_graphs)} {num_layers}-layer ego-graphs for template {template_id}"
            )

        return ego_graphs

    def _compute_weight_class(self, num_positive, num_negative, return_tensor=False):
        """
        Compute and normalize class weights based on inverse frequency.

        Args:
            num_positive (int): Number of class 1 (binding site) residues.
            num_negative (int): Number of class 0 (non-binding) residues.
            return_tensor (bool): If True, returns the weights instead of updating the attribute.

        Returns:
            Tensor (optional): Normalized weights tensor if return_tensor=True.
        """
        # Compute inverse frequency weights
        total = num_positive + num_negative
        w0 = total / (2 * num_negative)
        w1 = total / (2 * num_positive)

        # Normalize weights: sum to 1
        w_sum = w0 + w1
        w0_normalized = w0 / w_sum
        w1_normalized = w1 / w_sum

        weights = torch.tensor([w0_normalized, w1_normalized], dtype=torch.float)
        # weights = torch.tensor([0.35, 0.65], dtype=torch.float)

        if return_tensor:
            return weights
        else:
            self.train_data_class_weight = weights.to(self._device)

    def _get_input_dimension(self, tensor):
        """
        Safely get the number of feature dimensions from a tensor.

        Args:
            tensor (Tensor): Input tensor (could be empty).

        Returns:
            int or None: Number of feature dimensions if available, otherwise None.
        """
        if tensor is not None and tensor.numel() > 0:
            return tensor.shape[1]
        else:
            return None

    def _print_ego_graph_debug_info(self, ego_graphs, indices=(0, 5)):
        """
        Print debug information for a slice of ego-graphs.

        Args:
            ego_graphs (list): List of PyG Data objects.
            indices (tuple): Range of ego-graphs to inspect (start, end).
        """
        start, end = indices
        for i, g in enumerate(ego_graphs[start:end]):
            print(
                f"[DEBUG] Ego-graph {i+start}: center={g.ego_center}, "
                f"x.shape={g.x.shape}, edge_index.shape={g.edge_index.shape}, label={g.y.item()}"
            )
            print(f"         nodes: {g.ego_nodes}")
            print("         edges:")
            edge_list = g.edge_index.t().tolist()

            for edge_idx, (src_idx, tgt_idx) in enumerate(edge_list):
                src_name = g.ego_nodes[src_idx]
                tgt_name = g.ego_nodes[tgt_idx]
                if hasattr(g, "edge_attr") and g.edge_attr is not None:
                    edge_feat = g.edge_attr[edge_idx].tolist()
                    print(
                        f"           {src_name} <--> {tgt_name} | features: {edge_feat}"
                    )
                else:
                    print(f"           {src_name} <--> {tgt_name}")

            print(f"         x:\n{g.x}\n")
