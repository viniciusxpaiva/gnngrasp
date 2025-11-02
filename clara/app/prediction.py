import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
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


class Prediction:
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
        self.embeddings_dim = None
        self.property_dim = None
        self.edge_dim = None

    def prepare_dataset(self, input_protein):
        """
        Select and load similar templates to build the training dataset based on dynamic similarity threshold.

        This function runs BLAST to find similar proteins, loads each as a PyG Data object,
        and sets dimensional metadata for embeddings and edge features.

        Args:
            input_protein: Protein sequence or identifier to query templates against.

        Returns:
            DataLoader: PyTorch Geometric DataLoader with prepared training data.
        """
        print("[!] Preparing dataset...")

        # Step 1: Run BLAST to select similar templates
        blast = BLAST(self.dirs["data"]["blast"])
        selected_templates = blast.run(input_protein, self.top_n_templates)

        # selected_templates = {"1a8t_A"}

        # Step 2: Build graphs from selected templates
        raw_data_list = [
            self._build_protein_graph(tpt_id) for tpt_id in selected_templates
        ]

        # Step 3: Filter out invalid graphs
        data_list = [d for d in raw_data_list if d is not None]
        total = 0
        for d in data_list:
            total = total + d.y.shape[0]
        print(f"Total de nós: {total}")
        exit()

        if not data_list:
            raise ValueError(
                "[ERROR] No valid templates found after filtering invalid properties."
            )

        # Step 4: Extract dimensional metadata from the first valid graph
        sample = data_list[0]
        self.embeddings_dim = sample.x_embeddings.shape[1]
        self.property_dim = sample.x_properties.shape[1]
        self.edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

        # Step 5: Create DataLoader
        self.loader = DataLoader(data_list, batch_size=2, shuffle=True)

        print(f"[+] Loaded {len(data_list)} selected training proteins")
        return self.loader

    def initialize_model(self):
        """
        Initialize a DeepGraspModel for end-to-end training (GNN → CNN → MLP)
        using the input dimensions and model configuration defined in prediction_params.
        """
        if not hasattr(self, "loader") or self.loader is None:
            raise ValueError(
                "[ERROR] Data loader not prepared. Please run prepare_dataset() first."
            )

        print(
            f"[!] Initializing DeepGraspModel with GNN type: {self.prediction_params['layer_type']} "
            f"({self.prediction_params['num_gnn_layers']} layers)"
        )

        # Step 1: Build the GNN module
        gnn_model = ProteinGNN(
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
        )

        # Step 2: Wrap in DeepGraspModel with CNN+MLP
        self.model = DeepGraspModel(
            gnn_module=gnn_model,
            cnn_channels=self.prediction_params["cnn_params"]["cnn_channels"],
            cnn_kernel_size=self.prediction_params["cnn_params"]["cnn_kernel_size"],
            cnn_dropout=self.prediction_params["cnn_params"]["cnn_dropout"],
            mlp_hidden=self.prediction_params["mlp_params"]["mlp_hidden"],
            mlp_dropout=self.prediction_params["mlp_params"]["mlp_dropout"],
        ).to(self._device)

    def train_model(self):
        """
        Train a DeepGraspModel using protein graphs with message passing (GNN) followed by CNN + MLP.
        Handles class imbalance using weighted cross-entropy loss.
        """
        # --- Step 1: Validate model and data ---
        if not hasattr(self, "model"):
            raise ValueError(
                "Model not initialized. Please call initialize_model() first."
            )

        if not hasattr(self, "loader") or self.loader is None:
            raise ValueError("Expected a single DataLoader in self.loader.")

        print(f"[!] Starting DeepGraspModel training | Device: {self._device}")
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
            f"[!] Class weights | 0 (non-binding): {class_weights[0]:.4f} | 1 (binding): {class_weights[1]:.4f}"
        )

        # --- Step 3: Optimizer and loss function ---
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.prediction_params["gnn_params"]["lr"],
            weight_decay=self.prediction_params["gnn_params"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

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

                # Forward pass through full model (GNN → CNN → MLP)
                output = self.model(
                    batch.x_embeddings,
                    batch.x_properties,
                    batch.edge_index,
                    batch.edge_attr if self.model.gnn.use_edge_attr else None,
                )

                # Compute loss and backpropagation
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Collect predictions
                preds = torch.argmax(output, dim=1).detach().cpu().numpy()
                labels = batch.y.detach().cpu().numpy()
                pred_labels.extend(preds)
                true_labels.extend(labels)

            # Epoch metrics
            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)
            f1 = f1_score(true_labels, pred_labels)

            if f1 >= best_f1:
                best_f1 = f1
                best_model_state = self.model.state_dict()

            if epoch % 10 == 0 or epoch == self.prediction_params["epochs"] - 1:
                print(f"[Epoch {epoch+1:>3}] Loss: {avg_loss:.4f} | F1: {best_f1:.3f}")

        # --- Step 5: Load best model ---
        self.model.load_state_dict(best_model_state)
        print(
            f"[✓] Best model (F1={best_f1:.4f}) loaded into memory for evaluation/prediction."
        )

        # --- Step 6: Plot training loss ---
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"], "training_loss_dashboard.html"
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def predict_binding_sites(self, input_protein, verbose: bool = False):
        """
        Predict binding sites for an input protein using a trained DeepGraspModel.

        Args:
            input_protein (EmbedProtein or FeatProtein): Input protein with node and edge data.
            verbose (bool): Whether to print debug information.

        Returns:
            DataFrame: residue_id + predicted_label (0 or 1)
        """
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Please train the model first.")

        # --- Step 1: Prepare model input tensors ---
        (
            x_embeddings,
            x_properties,
            edge_index_input,
            edge_features_input,
            batch,
            node_list,
        ) = self._prepare_inputs(input_protein)

        x_embeddings = x_embeddings.to(self._device)
        x_properties = x_properties.to(self._device)
        edge_index_input = edge_index_input.to(self._device)
        edge_features_input = edge_features_input.to(self._device)

        if verbose:
            print("[DEBUG] Running prediction on:")
            print(f"  - x_embeddings: {x_embeddings.shape}")
            print(f"  - x_properties: {x_properties.shape}")
            print(f"  - edge_index: {edge_index_input.shape}")
            if edge_features_input is not None:
                print(f"  - edge_features: {edge_features_input.shape}")

        # --- Step 2: Run forward pass ---
        print("[!] Running prediction with trained DeepGraspModel...")
        print(
            f"[!] use_edge_attr = {self.model.gnn.use_edge_attr} | use_embd_projection = {self.model.gnn.use_embd_projection} "
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                x_embeddings,
                x_properties,
                edge_index_input,
                edge_features_input if self.model.gnn.use_edge_attr else None,
            )

            probs = torch.softmax(logits, dim=1)[
                :, 1
            ]  # probability of class 1 (binding)

        # --- Step 3: Apply threshold ---
        threshold = self.prediction_params["prediction_threshold"]
        print(f"[!] Thresholding predictions at {threshold:.2f}...")
        predicted_labels = (probs >= threshold).long()

        cleanup_cuda(self.model)
        cleanup_cuda(self.model.gnn)

        # --- Step 4: Return results as DataFrame ---
        return pd.DataFrame(
            {
                "residue_id": node_list,
                "predicted_label": predicted_labels.cpu().numpy(),
            }
        )

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

                # --- Step 4a: Forward pass with new input format ---
                if getattr(self.model, "use_edge_features", True) and hasattr(
                    batch, "edge_attr"
                ):
                    output = self.model(
                        batch.x_embeddings,
                        batch.x_properties,
                        batch.edge_index,
                        batch.edge_attr if self.model.use_edge_attr else None,
                    )

                else:
                    output = self.model(
                        batch.x_embeddings, batch.x_properties, batch.edge_index
                    )

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
        (
            x_embeddings,
            x_properties,
            edge_index_input,
            edge_features_input,
            batch,
            node_list,
        ) = self._prepare_inputs(input_protein)

        if verbose:
            print("[DEBUG] Input tensors before GAT:")
            self._print_prediction_debug_info(x_embeddings, edge_index_input, node_list)

        # --- Step 2: Run prediction ---
        print("[!] Running prediction with trained GNN model...")

        with torch.no_grad():
            if self.model.use_edge_attr and edge_features_input is not None:
                probs = self._predict_single_model(
                    self.model,
                    x_embeddings.to(self._device),
                    edge_index_input.to(self._device),
                    edge_features_input.to(self._device),
                )
            else:
                probs = self._predict_single_model(
                    self.model,
                    x_embeddings.to(self._device),
                    x_properties.to(self._device),
                    edge_index_input.to(self._device),
                )

        if verbose:
            print("[DEBUG] Post-prediction graph overview:")
            x_proj = self.model.embedding_projector(x_embeddings)
            x_concat = torch.cat([x_proj, x_properties], dim=1)
            self._print_prediction_debug_info(x_concat, edge_index_input, node_list)

        # --- Step 3: Apply classification threshold ---
        threshold = self.prediction_params["prediction_threshold"]
        print(f"[!] Thresholding predictions at {threshold:.2f}...")
        predicted_labels = (probs >= threshold).long()

        cleanup_cuda(self.model)

        # --- Step 4: Return DataFrame with results ---
        return pd.DataFrame(
            {
                "residue_id": node_list,
                "predicted_label": predicted_labels.cpu().numpy(),
            }
        )

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _create_data_object(self, x, y, edge_index, edge_attr):
        """
        Create a PyG Data object with standardized attribute names for consistency.

        Args:
            x (Tensor): Node feature matrix (embeddings).
            y (Tensor): Labels for each node.
            edge_index (Tensor): Graph connectivity.
            edge_attr (Tensor): Edge feature matrix.

        Returns:
            Data: PyG Data object with standardized attributes.
        """
        batch = torch.zeros(y.size(0), dtype=torch.long)

        data = Data(
            x=x,  # Node features
            y=y,  # Node labels
            edge_index=edge_index,  # Graph edges
            edge_attr=edge_attr,  # Edge features
            batch=batch,  # Batch indices (single graph)
        )

        return data

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
        edge_index_input, edge_features_input = prepare_edge_data(
            edge_df, embeddings_df
        )
        edge_index_input = edge_index_input.to(self._device)

        # --- Optional: properties ---
        if self.use_node_properties:
            properties_df = input_protein.node_properties.fillna(0)

            # Normalize properties BEFORE aligning or converting to tensors
            properties_df = self._normalize_node_properties(properties_df)
            x_properties = prepare_node_tensor(properties_df).to(self._device)
        else:
            # If not used, create an empty tensor with 0 columns
            x_properties = torch.empty((x_embeddings.size(0), 0), device=self._device)

        # --- Edge features ---
        if edge_features_input is None or edge_features_input.numel() == 0:
            edge_features_input = torch.zeros(
                (edge_index_input.size(1), self.model.gnn.edge_dim), device=self._device
            )
        else:
            edge_features_input = edge_features_input.to(self._device)

        # --- Batch vector ---
        batch = torch.zeros(x_embeddings.size(0), dtype=torch.long, device=self._device)

        return (
            x_embeddings,
            x_properties,
            edge_index_input,
            edge_features_input,
            batch,
            embeddings_df["residue_id"],
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
        edge_attr = (
            self._normalize_edge_features(edge_features)
            if use_edge_attr and edge_features
            else None
        )

        # --- Step 5: Create Data object ---
        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            # batch=torch.zeros(y.size(0), dtype=torch.long),  # optional, for batching
        )
        # Attach custom attributes
        data.x_embeddings = x_embeddings
        data.x_properties = x_properties
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
        x_embeddings,
        x_properties,
        edge_index_input,
        edge_features_input=None,
    ):
        """
        Run a forward pass through a trained GNN model and return probabilities for each node.

        Args:
            model (ProteinGNN): The GNN model instance.
            x_embeddings (Tensor): Node embeddings.
            x_properties (Tensor): Node physicochemical properties.
            edge_index_input (Tensor): Graph edge indices.
            edge_features_input (Tensor, optional): Edge feature matrix.

        Returns:
            Tensor: Predicted probability (class 1) for each node.
        """
        model.eval()
        with torch.no_grad():
            if model.use_edge_attr and edge_features_input is not None:
                out = model(
                    x_embeddings, x_properties, edge_index_input, edge_features_input
                )
            else:
                out = model(x_embeddings, x_properties, edge_index_input)
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

    def _print_prediction_debug_info(
        self, x, edge_index, node_list, message="[DEBUG]", max_nodes=3
    ):
        """
        Print debug information after prediction, focusing on node feature shapes
        and neighborhood (edges) of selected nodes.

        Args:
            x (Tensor): Node features (after projection or full concatenation).
            edge_index (Tensor): Edge indices [2, num_edges].
            node_list (pd.Series): Residue IDs in order.
            message (str): Prefix for printed lines.
            max_nodes (int): Number of high-degree nodes to display.
        """
        print(f"{message} Node feature shape: {x.shape}")
        print(
            f"{message} Edge index shape: {edge_index.shape} (num_edges = {edge_index.shape[1]})"
        )

        edge_list = edge_index.t().tolist()
        edge_counts = {i: 0 for i in range(x.size(0))}
        edge_map = {i: [] for i in range(x.size(0))}

        for idx, (src, tgt) in enumerate(edge_list):
            edge_counts[src] += 1
            edge_counts[tgt] += 1
            edge_map[src].append((src, tgt, idx))
            edge_map[tgt].append((tgt, src, idx))

        top_nodes = sorted(edge_counts.keys(), key=lambda i: -edge_counts[i])[
            :max_nodes
        ]

        for node_idx in top_nodes:
            node_name = node_list.iloc[node_idx]
            print(
                f"{message} Residue: {node_name} (index {node_idx}) | Degree: {edge_counts[node_idx]}"
            )
            for src_idx, tgt_idx, edge_i in edge_map[node_idx]:
                src_name = node_list.iloc[src_idx]
                tgt_name = node_list.iloc[tgt_idx]
                print(f"{message}   {src_name} --> {tgt_name}")
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

    def _normalize_edge_features_standard(self, edge_features: list) -> torch.Tensor:
        """
        Normalize edge features using StandardScaler (mean=0, std=1).

        Args:
            edge_features (list): List of edge feature arrays (float).

        Returns:
            torch.Tensor: Scaled edge features as a FloatTensor.
        """
        edge_features_array = np.vstack(edge_features)
        scaler = StandardScaler()
        edge_features_scaled = scaler.fit_transform(edge_features_array)
        return torch.from_numpy(edge_features_scaled).float()

    def _normalize_edge_features(self, edge_features: list) -> torch.Tensor:
        """
        Normalize edge features to [-1, 1] range across all edges.

        Args:
            edge_features (list): List of feature arrays (one per edge).

        Returns:
            torch.Tensor: Tensor of normalized edge features.
        """
        array = np.array(edge_features)  # shape: (num_edges, num_features)

        min_vals = array.min(axis=0)
        max_vals = array.max(axis=0)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        norm_array = 2 * (array - min_vals) / range_vals - 1

        return torch.tensor(norm_array, dtype=torch.float)

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
