import copy
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from app.nn.deep_grasp_model import DeepGraspModel
from app.nn.gnn.protein_gnn import ProteinGNN
from app.utils.prediction_utils import (
    prepare_node_tensor,
    prepare_edge_data,
    cleanup_cuda,
)
from data.blast.blast import BLAST
from app.utils.plotting import plot_training_loss_curve
import scipy.sparse as sp
import subprocess

from smote.data_load import load_data_protein, load_input_protein
from nodeimport.main_nodeimport import run_nodeimport, get_default_args


class NodeImportPrediction:
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
        self.use_node_properties = prediction_params["use_embd_projection"]
        self.top_n_templates = prediction_params["top_n_templates"]

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = "cpu"

        # Attributes to be initialized during dataset loading and model setup
        self.model = None
        self.loader = None
        self.train_data_class_weight = None
        self.embeddings_dim = 1280
        self.property_dim = None
        self.edge_dim = None

    def _run_nodeimport_training(self, graph):
        # Garante compatibilidade com NodeImport
        # Argumentos padrão do NodeImport
        parser = get_default_args()
        args = parser.parse_args(args=[])
        args.dataset = graph.template_id

        # Roda o NodeImport com o grafo atual
        results = run_nodeimport(graph, args)

        # graph = results["graph"]

        return results

    def prepare_dataset_nodeimport(
        self, input_protein, output_dir="nodeimport_graphs", verbose=False
    ):
        """
        Prepara grafos em formato PyG (Data) para o método NodeImport.
        Cada grafo recebe x, edge_index, y, e as máscaras train/val/test.

        Args:
            input_protein (str): Caminho para o PDB de entrada.
            output_dir (str): Pasta para salvar os grafos.
            verbose (bool): Exibir logs detalhados.

        Returns:
            list: Lista de objetos `Data` prontos para o NodeImport.
        """
        print("[!] Preparing dataset for NodeImport...")

        # Etapa 1: BLAST + seleção de templates
        blast = BLAST(self.dirs["data"]["blast"])
        selected_templates = blast.run(input_protein, self.top_n_templates)

        # DEBUG opcional para testar apenas um template
        # selected_templates = {"1a8t_A", "2313", "dasdasd"}

        # selected_templates = list(selected_templates)[:20]

        print(
            f"[!] Selected {len(selected_templates)} templates | Generating PyG graphs..."
        )

        os.makedirs(output_dir, exist_ok=True)
        data_list = []

        for tpt_id in selected_templates:
            graph = self._build_protein_graph(tpt_id)
            if graph is None:
                continue

            num_nodes = graph.num_nodes
            # print(f"Num of positive class: {graph.y.sum()} in {num_nodes} total nodes")

            # Criação das máscaras
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            perm = torch.randperm(num_nodes)
            train_mask[perm[: int(0.6 * num_nodes)]] = True
            val_mask[perm[int(0.6 * num_nodes) : int(0.8 * num_nodes)]] = True
            test_mask[perm[int(0.8 * num_nodes) :]] = True

            graph.train_mask = train_mask
            graph.val_mask = val_mask
            graph.test_mask = test_mask
            graph.template_id = tpt_id

            data_list.append(graph)

        # Etapa 3: Treinar modelo NodeImport com todos os grafos
        print(f"[!] Training NodeImport on {len(data_list)} graphs...")
        graph_copies = [copy.deepcopy(g) for g in data_list]
        batched_graph = Batch.from_data_list(graph_copies)
        results = self._run_nodeimport_training(batched_graph)
        model = results["model"]

        # Etapa 4: Aplicar o modelo treinado em cada grafo individual
        results = []
        for graph in data_list:
            graph = graph.to("cuda" if torch.cuda.is_available() else "cpu")
            if graph.x is None or graph.edge_index is None:
                print(f"[!] Atributos ausentes no grafo {graph.template_id}")
                continue
            model.eval()
            with torch.no_grad():
                logits = model(graph.x, graph.edge_index)
                probs = torch.softmax(logits, dim=1)
                graph.y_prob = probs[:, 1]  # Probabilidade da classe positiva
                graph.y_pred = (graph.y_prob > 0.7).long()  # Threshold conservador
            results.append(graph)

            if verbose:
                print(f"\n[+] Template: {graph.template_id}")
                print(f"Original: {graph.y.sum()} positivos")
                print(f"NodeImport y_pred: {graph.y_pred.sum()}")

        cleanup_cuda(model)
        print(
            f"[✓] Finalizado: {len(results)} grafos com labels gerados via NodeImport."
        )
        self.loader = DataLoader(results, batch_size=2, shuffle=True)

    def initialize_gnn(self):
        """
        Initialize a single GNN model for training using the prepared training data.
        This method builds the model architecture based on the specified configuration
        and the input feature dimensions extracted during dataset preparation.
        """
        if not hasattr(self, "loader") or self.loader is None:
            raise ValueError(
                "[ERROR] Data loader not prepared. Please run prepare_dataset() first."
            )

        print(
            f"[!] Initializing GNN model: {self.prediction_params['layer_type']} "
            f"({self.prediction_params['num_gnn_layers']} layers)"
        )

        self.model = ProteinGNN(
            embeddings_dim=self.embeddings_dim,
            property_dim=self.property_dim,
            projected_embd_dim=self.prediction_params["gnn_params"][
                "projected_embd_dim"
            ],
            edge_dim=self.edge_dim,
            hidden_channels=self.prediction_params["gnn_params"]["hidden_dim"],
            num_gnn_layers=self.prediction_params["num_gnn_layers"],
            layer_type=self.prediction_params["layer_type"],
            dropout=self.prediction_params["gnn_params"]["dropout"],
            use_edge_attr=self.prediction_params["use_edge_attr"],
            num_heads=self.prediction_params["gnn_params"]["num_heads"],
            use_embd_projection=self.prediction_params["use_embd_projection"],
        ).to(self._device)

    def train_gnn(self):
        """
        Train a single GNN model using full protein graphs with message passing.
        Supports models with or without edge features and handles class imbalance
        using weighted cross-entropy loss.
        """
        # --- Step 1: Validate model and data ---
        if not hasattr(self, "model"):
            raise ValueError(
                "GNN model not initialized. Please call initialize_model() first."
            )

        if not hasattr(self, "loader") or self.loader is None:
            raise ValueError("Expected a single DataLoader in self.loader.")

        print(f"[!] Starting training the GNN model | Device: {self._device}")
        loader = self.loader

        # --- Step 2: Compute class distribution for weighted loss ---
        all_labels = torch.cat([batch.y for batch in loader.dataset])
        num_positive = (all_labels == 1).sum().item()
        num_negative = (all_labels == 0).sum().item()
        total = num_positive + num_negative

        print(
            f"[!] Training set: {num_positive} positive | {num_negative} negative | {total} total residues."
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

        self.model.train()
        loss_history = []
        best_f1 = -1.0
        best_model_state = None

        # --- Step 4: Training loop ---
        for epoch in range(self.prediction_params["epochs"]):
            total_loss = 0.0
            true_labels = []
            pred_labels = []

            for batch in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                output = self.model(batch.x, batch.edge_index)

                # --- Step 4b: Compute loss and backpropagate ---
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # --- Step 4c: Collect predictions for MCC calculation ---
                preds = torch.argmax(output, dim=1).detach().cpu().numpy()
                labels = batch.y.detach().cpu().numpy()
                pred_labels.extend(preds)
                true_labels.extend(labels)

            # --- Step 5: Compute metrics for epoch ---
            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)
            mcc = f1_score(true_labels, pred_labels)

            # --- Step 6: Track best model by MCC ---
            if mcc > best_f1:
                best_f1 = mcc
                best_model_state = self.model.state_dict()

            if epoch % 10 == 0 or epoch == self.prediction_params["epochs"] - 1:
                print(f"[Epoch {epoch+1:>3}] Loss: {avg_loss:.4f} | MCC: {best_f1:.3f}")

        # --- Step 7: Load best model into memory ---
        self.model.load_state_dict(best_model_state)
        print(
            f"[✓] Best model (MCC={best_f1:.4f}) loaded into memory for evaluation/prediction."
        )

        # --- Step 8: Plot training loss curve ---
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"], "training_loss_dashboard.html"
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def predict_gnn(self, input_protein, verbose: bool = False):
        """
        Predict binding sites for an input protein using a trained GNN model.

        Args:
            input_protein (EmbedProtein or FeatProtein): The input protein object containing node and edge data.

        Returns:
            DataFrame: A DataFrame with residue IDs and predicted labels (0 = non-binding, 1 = binding).
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train the model first.")

        # --- Step 1: Prepare tensors for model input ---
        input_graph = self._prepare_inputs(input_protein)

        # --- Step 2: Run prediction ---
        print("[!] Running prediction with trained GNN model...")

        with torch.no_grad():
            probs = self._predict_single_model(
                self.model,
                input_graph.x,
                input_graph.edge_index,
            )

        # --- Step 3: Apply classification threshold ---
        threshold = self.prediction_params["prediction_threshold"]
        print(f"[!] Thresholding predictions at {threshold:.2f}...")
        predicted_labels = (probs >= threshold).long()

        cleanup_cuda(self.model)

        # --- Step 4: Return DataFrame with results ---
        return pd.DataFrame(
            {
                "residue_id": input_graph.residues_id,
                "predicted_label": predicted_labels.cpu().numpy(),
            }
        )

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _prepare_inputs(self, input_protein):
        """
        Prepare tensors required to run a GNN model, including node embeddings,
        optional node properties, edge connectivity, edge features, and batch information.

        Returns:
            Tuple:
                x_embeddings (Tensor): Node embeddings.
                x_properties (Tensor): Handcrafted node features (or empty if unused).
                edge_index_input (Tensor): Edge indices.
                edge_features_input (Tensor): Edge feature matrix.
                batch (Tensor): Batch vector.
                node_list (Series): Series with residue IDs.
        """
        embeddings_df = input_protein.node_embeddings
        edge_df = input_protein.edge_properties

        # --- Required ---
        x_embeddings = prepare_node_tensor(embeddings_df).to(self._device)
        edge_index_input, _ = prepare_edge_data(edge_df, embeddings_df)
        edge_index_input = edge_index_input.to(self._device)
        residues_id = embeddings_df["residue_id"]

        # --- Batch vector ---
        # batch = torch.zeros(x_embeddings.size(0), dtype=torch.long, device=self._device)

        return Data(
            x=x_embeddings,
            edge_index=edge_index_input,
            residues_id=residues_id,
            # batch,
            # embeddings_df["residue_id"],
        )

    def _build_protein_graph(
        self, tpt_pdb_id: str, use_edge_attr: bool = True, verbose: bool = False
    ):
        """
        Build a PyG Data graph from a template protein, including reduced node embeddings
        concatenated with physicochemical properties, and optional edge features.

        Args:
            tpt_pdb_id (str): Template identifier (e.g., "1abc_A").
            use_edge_attr (bool): Whether to include edge features in the graph.
            verbose (bool): Whether to print graph debug information.

        Returns:
            Data or None: PyG Data object containing graph information, or None if required files are missing.
        """
        # --- Step 1: Define paths to embeddings, node properties, and edge features ---
        if self.prediction_params["embd_type"] == "ESM":
            embd_node_path = os.path.join(
                self.dirs["data"]["esm_templates"]["node_embeddings"],
                f"{tpt_pdb_id}_node_embeddings.csv.zip",
            )
        elif self.prediction_params["embd_type"] == "PT5":
            embd_node_path = os.path.join(
                self.dirs["data"]["pt5_templates"]["node_embeddings"],
                f"{tpt_pdb_id}_node_embeddings.csv.zip",
            )
        else:
            embd_node_path = os.path.join(
                self.dirs["data"]["pbert_templates"]["node_embeddings"],
                f"{tpt_pdb_id}_node_embeddings.csv.zip",
            )

        prop_node_path = os.path.join(
            self.dirs["data"]["prop_templates"]["node_properties"],
            f"{tpt_pdb_id}_node_properties.csv.zip",
        )

        prop_edge_path = os.path.join(
            self.dirs["data"]["prop_templates"]["edge_properties"],
            f"{tpt_pdb_id}_edge_properties.csv.zip",
        )

        # --- Step 2: Check for missing files ---
        if not os.path.exists(embd_node_path):
            print(f"[WARNING] Missing node embeddings: {embd_node_path}")
            return None
        if not os.path.exists(prop_node_path):
            print(f"[WARNING] Missing node properties: {prop_node_path}")
            return None
        if not os.path.exists(prop_edge_path):
            print(f"[WARNING] Missing edge properties: {prop_edge_path}")
            return None

        # --- Step 3: Load embeddings and properties ---
        embd_df = pd.read_csv(embd_node_path, compression="zip")
        prop_df = pd.read_csv(prop_node_path, compression="zip")

        # Normalize properties BEFORE aligning or converting to tensors
        prop_df = self._normalize_node_properties(prop_df)

        # Align them safely (handles duplicates, order, missing residue_ids)
        embd_df, prop_df = self._align_node_dfs(embd_df, prop_df, tpt_pdb_id)

        residue_ids = embd_df["residue_id"].values
        labels = embd_df["label"].values
        y = torch.tensor(labels, dtype=torch.long)

        # Extract raw embedding and property tensors
        x_embeddings = torch.tensor(
            embd_df.drop(columns=["residue_id", "label"]).values, dtype=torch.float
        )
        x_properties = torch.tensor(
            prop_df.drop(columns=["residue_id", "label"]).values, dtype=torch.float
        )

        # --- Step 4: Load and process edge features ---
        edge_df = pd.read_csv(prop_edge_path, compression="zip")
        residue_to_index = {rid: idx for idx, rid in enumerate(residue_ids)}
        edge_indices = []
        edge_features = []

        edge_feature_cols = edge_df.columns.difference(["source", "target"])

        for idx, (src, tgt) in edge_df[["source", "target"]].iterrows():
            if src in residue_to_index and tgt in residue_to_index:
                edge_indices.append([residue_to_index[src], residue_to_index[tgt]])

                if use_edge_attr:
                    values = edge_df.loc[idx, edge_feature_cols].values.astype(float)
                    if np.isnan(values).any():
                        continue
                    edge_features.append(values)

        if not edge_indices:
            print(f"[WARNING] No valid edges for {tpt_pdb_id}")
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()

        # --- Step 5: Create Data object ---
        data = Data(
            edge_index=edge_index,
            y=y,
            # batch=torch.zeros(y.size(0), dtype=torch.long),  # optional, for batching
        )
        # Attach custom attributes
        data.x = x_embeddings
        # data.x_properties = x_properties
        data.template_id = tpt_pdb_id
        data.num_nodes = x_embeddings.size(0)

        # --- Step 6: Debug visualization ---
        if verbose and data is not None:
            self._print_graph_debug_info(
                data=data,
                residue_ids=residue_ids,
                max_nodes=5,
            )
        return data

    def _predict_single_model(
        self,
        model,
        x,
        # x_properties,
        edge_index,
        # edge_features_input=None,
    ):
        """
        Run a forward pass through a trained GNN model and return probabilities for each node.

        Args:
            model (ProteinGNN): The GNN model instance.
            x (Tensor): Node embeddings.
            x_properties (Tensor): Node physicochemical properties.
            edge_index (Tensor): Graph edge indices.
            edge_features_input (Tensor, optional): Edge feature matrix.

        Returns:
            Tensor: Predicted probability (class 1) for each node.
        """
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]  # Probability for class 1
        return probs.cpu()

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

    def _print_graph_debug_info(self, data, residue_ids, max_nodes=3):
        """
        Print debug information about the graph-level PyG Data object.

        Args:
            data (Data): PyG Data object with attributes x, y, edge_index, edge_attr.
            residue_ids (List[str]): List of residue names in the same order as x.
            max_nodes (int): Number of residues to display edge info for.
        """
        print(f"[DEBUG] Graph from template: {getattr(data, 'template_id', 'unknown')}")
        print(
            f"         x_embeddings.shape={data.x_embeddings.shape}, x_properties.shape={data.x_properties.shape}, edge_index.shape={data.edge_index.shape}, y.shape={data.y.shape}"
        )

        # Print last 5 residues to verify alignment
        print("         [DEBUG] Last 5 residues and their features:")
        for i in range(-5, 0):
            idx = i + len(residue_ids)  # convert to positive index
            res_id = residue_ids[idx]
            emb = data.x_embeddings[idx].tolist()
            props = data.x_properties[idx].tolist()
            print(f"           {res_id} | emb[0:3]={emb[:3]} | props={props}")
        print(
            f"         x_embeddings.shape={data.x_embeddings.shape}, x_properties.shape={data.x_properties.shape}, edge_index.shape={data.edge_index.shape}, y.shape={data.y.shape}"
        )

        edge_list = data.edge_index.t().tolist()
        edge_counts = {i: 0 for i in range(len(residue_ids))}
        edge_map = {i: [] for i in range(len(residue_ids))}

        for idx, (src, tgt) in enumerate(edge_list):
            edge_counts[src] += 1
            edge_counts[tgt] += 1
            edge_map[src].append((src, tgt, idx))  # Save edge index too
            edge_map[tgt].append((tgt, src, idx))  # Bidirectional view

        # Get top max_nodes residues with most edges (or just first N)
        top_nodes = sorted(edge_counts.keys(), key=lambda i: -edge_counts[i])[
            :max_nodes
        ]

        for node_idx in top_nodes:
            node_name = residue_ids[node_idx]
            print(
                f"         Residue: {node_name} (index {node_idx}) | Degree: {edge_counts[node_idx]}"
            )
            for src_idx, tgt_idx, edge_i in edge_map[node_idx]:
                src_name = residue_ids[src_idx]
                tgt_name = residue_ids[tgt_idx]
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    edge_feat = data.edge_attr[edge_i].tolist()
                    print(
                        f"           {src_name} --> {tgt_name} | features: {edge_feat}"
                    )
                else:
                    print(f"           {src_name} --> {tgt_name}")
            print()

    def _align_node_dfs(
        self, embd_df: pd.DataFrame, prop_df: pd.DataFrame, tpt_pdb_id: str = ""
    ) -> pd.DataFrame:
        """
        Aligns the node properties DataFrame (prop_df) to match the order and content
        of the embeddings DataFrame (embd_df), based on 'residue_id'.

        Handles:
        - Duplicate residue_id rows in prop_df (keeps first)
        - Mismatched order of residue_id
        - Missing residue_ids in prop_df (fills with NaN)

        Args:
            embd_df (pd.DataFrame): DataFrame with node embeddings, must contain 'residue_id'.
            prop_df (pd.DataFrame): DataFrame with node properties, must contain 'residue_id'.
            tpt_pdb_id (str): Template identifier, used for debug messages.

        Returns:
            pd.DataFrame: Aligned prop_df, same order and residue_ids as embd_df.
        """
        if "residue_id" not in embd_df.columns or "residue_id" not in prop_df.columns:
            raise ValueError("Both DataFrames must contain 'residue_id' column.")

        # Step 1: Drop duplicates in prop_df and embd_df
        prop_df = prop_df.drop_duplicates(subset="residue_id", keep="first").copy()
        embd_df = embd_df.drop_duplicates(subset="residue_id", keep="first").copy()

        # Step 2: Set residue_id as index for reordering
        prop_df = prop_df.set_index("residue_id")
        embd_df = embd_df.set_index("residue_id")

        # Step 3: Reindex prop_df to match embd_df (fill missing rows with NaN)
        aligned_props = prop_df.reindex(embd_df.index)

        # Step 4: Report mismatches if any
        num_missing = aligned_props.isnull().all(axis=1).sum()
        if num_missing > 0:
            print(
                f"[WARNING] {num_missing} residues missing in prop_df for {tpt_pdb_id}"
            )

        # Step 5: Fill NaN with 0 (optional)
        aligned_props = aligned_props.fillna(0)

        # Step 6: Reset index to restore residue_id as column
        aligned_props = aligned_props.reset_index()
        embd_df = embd_df.reset_index()

        return embd_df, aligned_props

    def _normalize_node_properties(self, prop_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the physicochemical properties to the range [-1, 1].

        Args:
            prop_df (pd.DataFrame): DataFrame with residue_id and numeric properties.

        Returns:
            pd.DataFrame: Normalized DataFrame with same shape and columns.
        """
        df = prop_df.copy()
        exclude_cols = (
            ["residue_id", "label"] if "label" in df.columns else ["residue_id"]
        )

        for col in df.columns:
            if col in exclude_cols:
                continue
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df[col] = 2 * (df[col] - min_val) / (max_val - min_val) - 1
            else:
                df[col] = 0  # ou 1, se preferir definir um valor constante

        return df

    def _export_graph_to_npz(self, graph, prot_name):
        save_dir = self.dirs["smote"]["templates"]
        os.makedirs(save_dir, exist_ok=True)
        name = prot_name + "_training_data"
        save_path = os.path.join(save_dir, name)
        if os.path.exists(save_path + ".npz"):
            return save_path + ".npz"

        X = graph.x_embeddings
        Y = graph.y
        edge_index = graph.edge_index

        num_nodes = X.shape[0]
        rows, cols = edge_index[0].numpy(), edge_index[1].numpy()
        adj = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes)
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        idx = np.where(Y.numpy() >= 0)[0]
        np.random.shuffle(idx)
        n = len(idx)
        train_idx = idx[: int(0.6 * n)]
        val_idx = idx[int(0.6 * n) : int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]

        np.savez(
            save_path,
            features=X.numpy().astype(np.float32),
            labels=Y.numpy().astype(np.int64),
            adj=adj,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        return save_path + ".npz"

    def _export_input_to_npz(self, graph):
        prot_name = "input"
        save_dir = self.dirs["output"]["prot_out_dir"]
        os.makedirs(save_dir, exist_ok=True)
        name = prot_name + "_training_data"
        save_path = os.path.join(save_dir, name)

        X = graph.x_embeddings.cpu()
        edge_index = graph.edge_index.cpu()

        num_nodes = X.shape[0]
        rows, cols = edge_index[0].numpy(), edge_index[1].numpy()
        adj = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes)
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        np.savez(
            save_path,
            features=X.numpy().astype(np.float32),
            adj=adj,
        )

    def _run_graphsmote_training(self, npz_path):
        protein_id = os.path.basename(npz_path).replace("_training_data.npz", "")
        checklist_dir = f"{self.dirs["smote"]["checkpoint"]}/{protein_id}"
        training_path = checklist_dir + f"/recon_2000_False_0.5.pth"
        model_path = checklist_dir + f"/newG_cls_2000_False_0.5.pth"

        if not os.path.exists(training_path):
            print(f"[*] Running GraphSMOTE training for {protein_id}...")
            os.makedirs(checklist_dir, exist_ok=True)
            cmd_recon = f"python {self.dirs["smote"]["base"]}/main.py --imbalance --no-cuda --dataset={protein_id} --setting=recon"
            subprocess.run(cmd_recon, shell=True, check=True)

        if not os.path.exists(model_path):
            print(f"[*] Running GraphSMOTE fine-tuning for {protein_id}...")
            cmd_finetune = f"python {self.dirs["smote"]["base"]}/main.py --imbalance --no-cuda --dataset={protein_id} --setting=newG_cls --load=recon_2000_False_0.5"
            subprocess.run(cmd_finetune, shell=True, check=True)

        return model_path

    def _load_graphsmote_balanced_graph(self, npz_path, model_path):
        from smote.models import Sage_En, Decoder
        from smote import utils  # certifique-se que o utils.py está acessível

        # Step 1: Load original graph data
        adj, features, labels = load_data_protein(npz_path)
        ori_num = labels.shape[0]

        # Step 2: Load encoder and decoder
        self.encoder = Sage_En(nfeat=features.shape[1], nhid=64, nembed=64, dropout=0.1)
        decoder = Decoder(nembed=64, dropout=0.1)
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        self.encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        self.encoder.eval()
        decoder.eval()

        with torch.no_grad():
            embed = self.encoder(features, adj)

            # Step 3: Apply GraphSMOTE upsampling
            idx_train = torch.arange(
                labels.shape[0]
            )  # usa todos os nós como treinamento
            embed_up, labels_up, _, adj_up = utils.recon_upsample(
                embed,
                labels,
                idx_train=idx_train,
                adj=adj.to_dense(),
                portion=1,
                im_class_num=1,
            )

            # Step 4: Create new edge_index
            adj_up[:ori_num, :ori_num] = adj.to_dense()  # restore original connections
            adj_np = adj_up.detach().numpy()
            rows, cols = np.where(adj_np > 0.5)
            edge_index = torch.from_numpy(np.array([rows, cols], dtype=np.int64))

        # Step 5: Create PyG Data object with expanded nodes and labels
        data = Data(
            x=embed_up,  # includes original + synthetic nodes
            edge_index=edge_index,
            y=labels_up,  # includes new labels
        )
        return data
