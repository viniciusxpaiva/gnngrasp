import os
import random
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from app.nn.hier.focal_loss import FocalLoss
from app.nn.hier.gat_node_classifier import GATNodeClassifier
from app.nn.hier.gin_node_classifier import GINNodeClassifier
from app.nn.hier.inter_subgraph_classifier import IntermediateSubgraphClassifier
from app.nn.hier.gatB_subgraph_classifier import GATBiasSubgraphClassifier
from app.nn.hier.gin_subgraph_classifier import GINSubgraphClassifier
from app.utils.build_subgraphs_from_neighbors import (
    build_template_subgraphs_from_neighbors,
)
from app.utils.plotting import plot_training_loss_curve
from app.utils.prediction_utils import cleanup_cuda
from data.blast.blast import BLAST
from sklearn.metrics import f1_score, matthews_corrcoef


class HierPrediction:
    """
    Class to handle template selection, model initialization, training, and prediction.

    It loads node/edge features and global embeddings, builds and trains the GNN
    (GAT or basic), and generates binding site predictions using parameters from a
    centralized configuration dictionary.
    """

    def __init__(self, dirs, params):
        """
        Initialize the Prediction object with configured hyperparameters.

        Args:
            dirs (dict): Dictionary with predefined folder structure and file paths.
            prediction_params (dict): Dictionary containing model and training parameters,
                                      including architecture configuration and prediction threshold.
        """
        self.dirs = dirs
        self.prediction_params = params
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = "cpu"

        # Attributes to be initialized during dataset loading and model setup
        self.subgraph_model = None
        self.node_model = None
        self.train_loader = None
        self.node_embd_dim = None
        self.edge_prop_dim = None

    def prepare_hierarchical_graph_dataset(self, input_protein):
        """
        Prepare a dataset for hierarchical prediction using a fixed number of subgraphs
        per template protein. Subgraphs are selected randomly or based on heuristics
        from the global graph (e.g., half the number of residues).

        Args:
            input_protein (str): Input protein file or ID.

        """
        print("[!] Preparing hierarchical subgraph dataset...")

        params = self.prediction_params
        subgraph_method = params["subg_gen_method"]
        neighbor_layers = params["subg_neighbor_layers"]
        top_n_templates = params["top_n_templates"]
        exposure_percent = params["asa_exposure_percent"]

        # Run BLAST to find similar templates
        blast = BLAST(self.dirs["data"]["blast"])
        selected_templates = blast.run(input_protein, top_n_templates)

        # selected_templates = {"1a8t_A"}

        print(f"[!] Selected {len(selected_templates)} templates")
        subgraphs = []
        for tpt_id in selected_templates:
            if subgraph_method == "color":
                local_subgraphs = self._build_subgraphs_from_template_color(
                    template_id=tpt_id,
                    num_layers=neighbor_layers,
                )
            elif subgraph_method == "anchor":
                local_subgraphs = self._build_subgraphs_from_template_anchor(
                    template_id=tpt_id,
                    num_layers=neighbor_layers,
                )
            else:
                local_subgraphs = self._build_subgraphs_from_template_asa(
                    template_id=tpt_id,
                    num_layers=neighbor_layers,
                    exposure_percent=exposure_percent,
                )
            subgraphs.extend(local_subgraphs)

        if not subgraphs:
            raise ValueError("[ERROR] No subgraphs found for selected templates.")

        # === Analyze subgraph label distribution ===
        num_positive = sum(g.y.item() == 1 for g in subgraphs)
        num_negative = len(subgraphs) - num_positive
        positive_ratio = num_positive / len(subgraphs)

        print(f"[INFO] Subgraph label distribution:")
        print(f"       Positive (binding): {num_positive}")
        print(f"       Negative (non-binding): {num_negative}")
        print(f"       Positive ratio: {positive_ratio:.2%}")

        site_ratios = [g.site_ratio.item() for g in subgraphs]
        avg_site_ratio = sum(site_ratios) / len(site_ratios)
        print(f"       Avg. site ratio in subgraphs: {avg_site_ratio:.3f}")

        self.node_embd_dim = subgraphs[0].x.shape[1]
        self.edge_prop_dim = (
            subgraphs[0].edge_attr.shape[1] if subgraphs[0].edge_attr is not None else 0
        )

        self.train_loader = DataLoader(subgraphs, batch_size=64, shuffle=True)
        print(
            f"[+] {subgraph_method.upper()} method {exposure_percent}% | Loaded {len(subgraphs)} subgraphs from {len(selected_templates)} templates."
        )

    def initialize_subgraph_classifier(self):
        """
        Instantiate a subgraph-level GNN classifier for subgraph classification.

        Uses parameters specified under the 'subgraph_classifier' block in HIER_PARAMS.
        """
        params = self.prediction_params["subgraph_classifier"]

        print(
            f"Initializing subgraph classifier | {params["gnn_type"]} | {params["num_layers"]} layers | {params["norm_type"]} norm | {params["pool_type"]} pooling"
        )

        if params["gnn_type"] == "GAT":
            self.subgraph_model = GATBiasSubgraphClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                num_heads=params["num_heads"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)
        elif params["gnn_type"] == "GIN":
            self.subgraph_model = GINSubgraphClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

    def initialize_node_classifier(self):
        """
        Instantiate a node-level GNN classifier for residue prediction within subgraphs.

        Uses parameters specified under the 'node_classifier' block in HIER_PARAMS.
        """
        params = self.prediction_params["node_classifier"]

        print(
            f"Initializing node classifier | {params["gnn_type"]} | {params["num_layers"]} layers | {params["norm_type"]} norm"
        )

        if params["gnn_type"] == "GAT":
            self.node_model = GATNodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                num_heads=params["num_heads"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
            ).to(self._device)
        elif params["gnn_type"] == "GIN":
            self.node_model = GINNodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
            ).to(self._device)
        else:
            raise ValueError("[ERROR] No valid GNN type for node classifier.")

    def train_subgraph_classifier(self):
        """
        Train the subgraph-level GNN classifier (GNN1) to distinguish between
        binding site–containing subgraphs and non-binding ones, and plot the training loss.
        """
        print(f"[!] Starting subgraph-level training... | {self._device}")
        params = self.prediction_params["subgraph_classifier"]

        self.subgraph_model.train()

        optimizer = torch.optim.AdamW(
            self.subgraph_model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        loss_fn = nn.BCEWithLogitsLoss()

        loss_history = []

        for epoch in range(1, params["epochs"] + 1):
            epoch_loss = 0.0
            y_true, y_pred = [], []

            for batch in self.train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                # === Forward pass ===
                logits = self.subgraph_model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.float().unsqueeze(1)

                # === Compute loss ===
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                # === Record loss and predictions ===
                epoch_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)

                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(preds)

            # === Epoch metrics ===
            mcc = matthews_corrcoef(y_true, y_pred)
            f1_s = f1_score(y_true, y_pred)
            loss_history.append(epoch_loss / len(self.train_loader))

            if epoch % 20 == 0 or epoch == params["epochs"]:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Loss: {loss_history[-1]:.4f} | "
                    f"F1 Score: {f1_s:.4f} | MCC: {mcc:.4f}"
                )

        # === Plot loss curve ===
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            "subgraph_classifier_training_loss_dashboard.html",
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def train_node_classifier(self):
        """
        Train the node-level GNN classifier (GNN2) on residues within subgraphs.

        Configurable flags in baseline params:
        - use_only_positive_subgraphs_for_node_train:
                True  -> train only on positive subgraphs (subgraphs containing at least one positive node)
                False -> train on all subgraphs
        - node_labeling_mode:
                "anchor"    -> supervise only the anchor node in each subgraph
                "all_nodes" -> supervise every node in the subgraph

        Notes:
        - This is a binary classification problem (labels already 0/1).
        - Always uses BCEWithLogitsLoss with pos_weight adjustment.
        """
        print(f"[!] Starting node-level training... | {self._device} | ", end="")

        # ---------------------------
        # 1) Read config
        # ---------------------------
        params = self.prediction_params["node_classifier"]
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        use_only_pos_for_train = bool(params.get("all_or_pos_sg_node_training", "all"))

        batch_size = params.get("batch_size", 16)
        lr = params.get("lr", 1e-3)
        wd = params.get("weight_decay", 0.0)
        epochs = params.get("epochs", 100)

        # ---------------------------
        # 2) Collect subgraphs
        # ---------------------------
        if self.train_loader is None:
            raise ValueError(
                "Train subgraph DataLoader (self.train_loader) not initialized."
            )

        all_train_subgraphs = list(self.train_loader.dataset)

        if use_only_pos_for_train == "pos":
            subgraphs = [
                sg for sg in all_train_subgraphs if sg.node_labels.sum().item() > 0
            ]
            print(f"Using ONLY positive subgraphs for training: {len(subgraphs)}")
        else:
            subgraphs = all_train_subgraphs
            print(f"Using ALL subgraphs for training: {len(subgraphs)}")

        loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)

        # ---------------------------
        # 3) Compute pos_weight for BCE
        # ---------------------------
        if node_labeling_mode == "anchor":
            pos = sum(
                int(sg.node_labels[int(sg.ego_center_index)].item() == 1)
                for sg in subgraphs
            )
            neg = len(subgraphs) - pos
        else:
            pos = sum(sg.node_labels.sum().item() for sg in subgraphs)
            neg = sum((1 - sg.node_labels).sum().item() for sg in subgraphs)

        pos_weight = torch.tensor(
            [neg / max(pos, 1)], device=self._device, dtype=torch.float
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(
            f"[INFO] Loss=BCEWithLogits | pos={pos} neg={neg} | pos_weight/IR={pos_weight.item():.3f}"
        )

        # ---------------------------
        # 4) Optimizer
        # ---------------------------
        optimizer = torch.optim.Adam(
            self.node_model.parameters(), lr=lr, weight_decay=wd
        )

        # ---------------------------
        # 5) Training loop (proteins)
        # ---------------------------
        loss_history = []
        for epoch in range(1, epochs + 1):
            self.node_model.train()
            total_loss, correct, total = 0.0, 0, 0

            for batch in loader:
                # Move batch to device
                batch = batch.to(self._device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass: per-node logits for the whole batch graph
                logits = self.node_model(batch.x, batch.edge_index).view(-1)

                if node_labeling_mode == "anchor":
                    # ================================
                    # Supervise ONLY the anchor residue
                    # ================================
                    # ego_center_index is already stored in each subgraph
                    centers = batch.ego_center_index.view(-1).long()

                    # Select logits and labels only for the anchors
                    logit_center = logits[centers]  # [B_graphs]
                    y_center = batch.node_labels[centers].float()

                    # Compute loss and predictions
                    loss = loss_fn(logit_center, y_center)
                    preds = (torch.sigmoid(logit_center) >= 0.5).long()

                    # Update metrics
                    correct += (preds == y_center.long()).sum().item()
                    total += y_center.numel()

                else:
                    # ================================
                    # Supervise ALL residues in the subgraph
                    # ================================
                    y = batch.node_labels.float().to(self._device)

                    # Compute loss and predictions for all nodes
                    loss = loss_fn(logits, y)
                    preds = (torch.sigmoid(logits) >= 0.5).long()

                    # Update metrics
                    correct += (preds == y.long()).sum().item()
                    total += y.numel()

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item()

            # Epoch summary
            avg_loss = total_loss / max(1, len(loader))
            acc = correct / max(1, total)
            loss_history.append(avg_loss)

            if epoch % 20 == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

        # ---------------------------
        # 6) Save training curve
        # ---------------------------
        out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            "node_classifier_training_loss_dashboard.html",
        )
        plot_training_loss_curve({1: loss_history}, output_path=out_path)

    def predict_binding_sites(self, input_subgraphs):
        """
        Hierarchical inference for the *inductive* Deep-GRaSP variant.

        Expected per-subgraph fields (inductive):
        - x:            [num_nodes_sub, feat_dim]
        - edge_index:   [2, num_edges_sub]
        - ego_nodes:    List[str]  (ordered residue identifiers local to this subgraph)
        - ego_center:   str        (anchor residue id; optional if 'ego_center_index' exists)
        - ego_center_index: int    (optional; local index of anchor; preferred when present)
        - (node_labels / y are NOT required for inference)

        Config (read with priority from section-specific blocks, then fallback to baseline):
        subgraph_classifier:
            - use_subgraph_classifier: bool (gate on/off)
            - prediction_threshold:    float in [0,1] (gate threshold)
            - batch_size:              int   (mini-batch for gate)
        node_classifier:
            - node_labeling_mode:      "anchor" | "all_nodes"
            - batch_size:              int   (mini-batch for node-level forward)
        baseline (fallbacks only):
            - use_subgraph_classifier, gate_threshold, node_labeling_mode
        """
        print(f"[!] Starting hierarchical binding site prediction... | {self._device}")

        # ---------------------------
        # 0) Read configuration with section-first priority
        # ---------------------------

        params = self.prediction_params
        sg_cfg = self.prediction_params.get("subgraph_classifier", {})
        nd_cfg = self.prediction_params.get("node_classifier", {})

        # Gate on/off (prefer subgraph_classifier flag)
        use_gate = bool(params.get("use_subgraph_classifier", True))

        # Gate threshold (prefer subgraph_classifier key 'prediction_threshold';
        gate_thr = float(sg_cfg.get("prediction_threshold", 0.5))

        # Batch sizes (section-specific defaults)
        gate_bs = int(sg_cfg.get("batch_size", 32))  # for GNN1 gate
        node_bs = int(nd_cfg.get("batch_size", 32))  # for GNN2 node pass

        # Node supervision/eval mode (prefer node_classifier)
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        # ---------------------------
        # 1) Optional subgraph gate (GNN1)
        # ---------------------------
        if use_gate:
            if not hasattr(self, "subgraph_model"):
                raise RuntimeError(
                    "use_subgraph_classifier=True but 'self.subgraph_model' is not initialized."
                )
            self.subgraph_model.eval()

        self.node_model.eval()

        retained_subgraphs = input_subgraphs
        if use_gate:
            retained_subgraphs = []
            gate_loader = DataLoader(input_subgraphs, batch_size=gate_bs, shuffle=False)

            with torch.no_grad():
                for batch in gate_loader:
                    batch = batch.to(self._device)

                    # One logit per subgraph in the mini-batch (shape [B] or [B,1])
                    logits = self.subgraph_model(batch.x, batch.edge_index, batch.batch)
                    probs = torch.sigmoid(logits).view(-1).cpu().numpy()

                    # Keep only the subgraphs with probability >= threshold
                    for i, sg in enumerate(batch.to_data_list()):
                        if probs[i] >= gate_thr:
                            retained_subgraphs.append(sg)

            print(
                f"[INFO] Retained {len(retained_subgraphs)}/{len(input_subgraphs)} "
                f"subgraphs after GNN1 gate @ {gate_thr:.2f}"
            )

            if len(retained_subgraphs) == 0:
                print(
                    "[WARN] No subgraphs retained by GNN1; returning empty predictions."
                )
                return {}
        else:
            print(
                f"[INFO] Skipping GNN1 gate; using all {len(retained_subgraphs)} input subgraphs."
            )

        # ---------------------------
        # 2) Node-level pass (GNN2) on retained subgraphs
        # ---------------------------
        residue_scores = (
            {}
        )  # Dict[str, List[float]] ; Aggregate multiple votes per residue

        node_loader = DataLoader(retained_subgraphs, batch_size=node_bs, shuffle=False)
        with torch.no_grad():
            for batch in node_loader:
                batch = batch.to(self._device)

                # Per-node logits over the *concatenated* batch graph → shape [sum_nodes_in_batch]
                logits_nodes = self.node_model(batch.x, batch.edge_index).view(-1)
                probs_nodes = torch.sigmoid(logits_nodes).cpu().numpy()

                # Delimit each subgraph inside this batch using ptr (preferred) or derive from batch.batch
                if hasattr(batch, "ptr") and batch.ptr is not None:
                    # ptr: prefix sums of node counts; len(ptr) = num_graphs_in_batch + 1
                    ptr = batch.ptr.long()
                else:
                    # Derive ptr from 'batch.batch' (graph ids per node)
                    num_nodes_per_graph = torch.bincount(batch.batch)
                    ptr = torch.cat(
                        [
                            num_nodes_per_graph.new_zeros(1),
                            num_nodes_per_graph.cumsum(0),
                        ]
                    ).long()

                graphs = batch.to_data_list()  # list[Data]; len == (len(ptr)-1)

                for g_idx, sg in enumerate(graphs):
                    start, end = int(ptr[g_idx].item()), int(ptr[g_idx + 1].item())
                    probs_g = probs_nodes[
                        start:end
                    ]  # probabilities for nodes in this subgraph

                    # ego_nodes is a List[str] with stable residue ids in *local* order
                    ego_nodes = list(sg.ego_nodes)

                    if node_labeling_mode == "anchor":
                        # Select *only* the anchor node from this subgraph
                        if hasattr(sg, "ego_center_index"):
                            c_local = int(sg.ego_center_index)
                        else:
                            # If we only have the anchor name, resolve its local index
                            if not hasattr(sg, "ego_center"):
                                raise RuntimeError(
                                    "anchor mode requires 'ego_center_index' or 'ego_center'."
                                )
                            try:
                                c_local = ego_nodes.index(sg.ego_center)
                            except ValueError:
                                raise RuntimeError(
                                    f"ego_center '{sg.ego_center}' was not found in ego_nodes (len={len(ego_nodes)})."
                                )

                        res_id = str(ego_nodes[c_local])  # stable residue name/id
                        p = float(probs_g[c_local])  # predicted prob for the anchor
                        residue_scores.setdefault(res_id, []).append(p)

                    else:  # "all_nodes": add a probability for every residue in this subgraph
                        for j, res_id in enumerate(ego_nodes):
                            residue_scores.setdefault(str(res_id), []).append(
                                float(probs_g[j])
                            )

        # ---------------------------
        # 3) Final aggregation per residue id (mean of votes across subgraphs)
        # ---------------------------
        averaged_scores = {
            res_id: float(np.mean(scores)) for res_id, scores in residue_scores.items()
        }

        print(
            f"[✓] Inference complete. Predicted scores for {len(averaged_scores)} residues."
        )
        if use_gate:
            cleanup_cuda(self.subgraph_model)
        cleanup_cuda(self.node_model)
        return averaged_scores

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _build_subgraphs_from_template_anchor(
        self, template_id: str, num_layers: int
    ) -> List[Data]:
        """
        Build k-hop ego-subgraphs for a template using **all residues as anchors** (root per residue).
        Coverage policy: only the root is marked as covered (one subgraph per residue, overlaps allowed).

        Files used:
            - ESM node embeddings:  {dirs.data.esm_templates.node_embeddings}/{template_id}_node_embeddings.csv.zip
            - Precomputed neighbors: {dirs.data.ego_templates}/{template_id}_neighbors.csv.zip

        Args:
            template_id: Template identifier, e.g., "1abc_A".
            num_layers: Number of BFS layers (k-hop expansion) from each root residue.

        Returns:
            List[Data]: One PyG Data per constructed subgraph. If required files are missing, returns [].
        """
        node_embd_path = os.path.join(
            self.dirs["data"]["esm_templates"]["node_embeddings"],
            f"{template_id}_node_embeddings.csv.zip",
        )
        neighbors_path = os.path.join(
            self.dirs["data"]["ego_templates"],
            f"{template_id}_neighbors.csv.zip",
        )

        if not (os.path.exists(node_embd_path) and os.path.exists(neighbors_path)):
            print(f"[WARNING] Missing files for {template_id}")
            return []

        node_embd_df = pd.read_csv(node_embd_path, compression="zip")
        neighbors_df = pd.read_csv(neighbors_path, compression="zip")
        cutoff_residues = list(node_embd_df["residue_id"])

        return build_template_subgraphs_from_neighbors(
            template_id=template_id,
            node_embd_df=node_embd_df,
            neighbors_df=neighbors_df,
            cutoff_residues=cutoff_residues,
            num_layers=num_layers,
            subgraph_type="anchor",
        )

    def _build_subgraphs_from_template_asa(
        self, template_id: str, num_layers: int, exposure_percent: float
    ) -> List[Data]:
        """
        Build k-hop ego-subgraphs for a template using **only solvent-exposed residues as anchors**.
        Filtering is done by `acc_all > exposure_percent`. Coverage policy: only the root is covered
        (same as anchor mode), i.e., one subgraph per selected residue.

        Files used:
            - ESM node embeddings:   {dirs.data.esm_templates.node_embeddings}/{template_id}_node_embeddings.csv.zip
            - Node properties (ASA): {dirs.data.prop_templates.node_properties}/{template_id}_node_properties.csv.zip
            - Precomputed neighbors: {dirs.data.ego_templates}/{template_id}_neighbors.csv.zip

        Args:
            template_id: Template identifier, e.g., "1abc_A".
            num_layers: Number of BFS layers (k-hop expansion).
            exposure_percent: Threshold for solvent accessibility; residues with acc_all > threshold are roots.

        Returns:
            List[Data]: One PyG Data per constructed subgraph. If required files are missing, returns [].
        """
        node_embd_path = os.path.join(
            self.dirs["data"]["esm_templates"]["node_embeddings"],
            f"{template_id}_node_embeddings.csv.zip",
        )
        node_prop_path = os.path.join(
            self.dirs["data"]["prop_templates"]["node_properties"],
            f"{template_id}_node_properties.csv.zip",
        )
        neighbors_path = os.path.join(
            self.dirs["data"]["ego_templates"],
            f"{template_id}_neighbors.csv.zip",
        )

        if not (
            os.path.exists(node_embd_path)
            and os.path.exists(neighbors_path)
            and os.path.exists(node_prop_path)
        ):
            print(f"[WARNING] Missing files for {template_id}")
            return []

        node_prop_df = pd.read_csv(node_prop_path, compression="zip")
        node_prop_df = node_prop_df[node_prop_df["acc_all"] > exposure_percent]

        node_embd_df = pd.read_csv(node_embd_path, compression="zip")
        neighbors_df = pd.read_csv(neighbors_path, compression="zip")
        cutoff_residues = list(node_prop_df["residue_id"])

        return build_template_subgraphs_from_neighbors(
            template_id=template_id,
            node_embd_df=node_embd_df,
            neighbors_df=neighbors_df,
            cutoff_residues=cutoff_residues,
            num_layers=num_layers,
            subgraph_type="anchor",  # ASA follows anchor-style coverage
        )

    def _build_subgraphs_from_template_color(
        self, template_id: str, num_layers: int
    ) -> List[Data]:
        """
        Build k-hop ego-subgraphs for a template using a **coloring/coverage scheme**.
        All residues are candidates as anchors, but after creating each subgraph,
        **all nodes in that subgraph are marked as covered**, reducing redundancy and
        typically producing fewer, broader subgraphs that still cover the whole protein.

        Files used:
            - ESM node embeddings:  {dirs.data.esm_templates.node_embeddings}/{template_id}_node_embeddings.csv.zip
            - Precomputed neighbors:{dirs.data.ego_templates}/{template_id}_neighbors.csv.zip

        Args:
            template_id: Template identifier, e.g., "1abc_A".
            num_layers: Number of BFS layers (k-hop expansion).

        Returns:
            List[Data]: Set of PyG Data objects that cover the template with minimal redundancy.
                        If required files are missing, returns [].
        """
        node_embd_path = os.path.join(
            self.dirs["data"]["esm_templates"]["node_embeddings"],
            f"{template_id}_node_embeddings.csv.zip",
        )
        neighbors_path = os.path.join(
            self.dirs["data"]["ego_templates"],
            f"{template_id}_neighbors.csv.zip",
        )

        if not (os.path.exists(node_embd_path) and os.path.exists(neighbors_path)):
            print(f"[WARNING] Missing files for {template_id}")
            return []

        node_embd_df = pd.read_csv(node_embd_path, compression="zip")
        neighbors_df = pd.read_csv(neighbors_path, compression="zip")
        cutoff_residues = list(node_embd_df["residue_id"])

        return build_template_subgraphs_from_neighbors(
            template_id=template_id,
            node_embd_df=node_embd_df,
            neighbors_df=neighbors_df,
            cutoff_residues=cutoff_residues,
            num_layers=num_layers,
            subgraph_type="color",
        )

    def _filter_positive_subgraphs(self):
        """
        Filter subgraphs labeled as positive (binding site present).
        Returns:
            list[Data]: Subgraphs with y == 1
        """
        if self.train_loader is None:
            raise ValueError("Subgraph DataLoader (self.train_loader) not initialized.")

        # Access raw subgraph list (not batched yet)
        positive_subgraphs = [g for g in self.train_loader.dataset if g.y.item() == 1]
        print(
            f"[INFO] Retained {len(positive_subgraphs)} positive subgraphs for node-level training."
        )
        return positive_subgraphs

    def _filter_positive_sub_subgraphs(self):
        """
        Filter sub-subgraphs labeled as positive (i.e., containing at least one binding residue).
        Returns:
            list[Data]: Sub-subgraphs with y == 1
        """
        if self.intermediate_loader is None:
            raise ValueError(
                "Intermediate subgraph DataLoader (self.intermediate_loader) not initialized."
            )

        # Access raw sub-subgraph list (not batched yet)
        positive_sub_subgraphs = [
            g for g in self.intermediate_loader.dataset if g.y.item() == 1
        ]
        print(
            f"[INFO] Retained {len(positive_sub_subgraphs)} positive sub-subgraphs for node-level training."
        )
        return positive_sub_subgraphs

    def _build_sub_subgraphs_from_positive_graphs(
        self,
        positive_graphs,
        num_layers,
        use_edge_attr=False,
        inference=False,
    ):
        """
        Generate non-overlapping sub-subgraphs from positive subgraphs by covering
        all nodes through iterative ego-centric neighborhood expansion.

        Args:
            positive_graphs (list[Data]): List of positive subgraphs (label = 1).
            num_layers (int): Number of hops (k) to define each neighborhood.
            use_edge_attr (bool): Whether to include edge attributes (if present).
            inference (bool): If True, disables label-related computations.

        Returns:
            list[Data]: List of sub-subgraph Data objects.
        """
        sub_subgraphs = []

        for parent_graph in positive_graphs:
            x = parent_graph.x  # [N, F]
            edge_index = parent_graph.edge_index
            num_nodes = x.size(0)
            ego_nodes = parent_graph.ego_nodes
            edge_attr = getattr(parent_graph, "edge_attr", None)

            if not inference:
                node_labels = parent_graph.node_labels
                template_id = parent_graph.template_id
            else:
                node_labels = torch.zeros(num_nodes)
                template_id = "input"

            covered = set()
            all_nodes = set(range(num_nodes))

            while covered < all_nodes:
                # Choose a new ego center that hasn't been covered yet
                candidates = list(all_nodes - covered)
                ego_idx = random.choice(candidates)

                # Extract k-hop neighborhood
                node_ids, sub_edge_index, _, edge_mask = k_hop_subgraph(
                    ego_idx, num_layers, edge_index, relabel_nodes=True
                )

                sub_x = x[node_ids]
                sub_labels = node_labels[node_ids.to(node_labels.device)]
                sub_edge_attr = (
                    edge_attr[edge_mask]
                    if use_edge_attr and edge_attr is not None
                    else None
                )

                if not inference:
                    sub_label = int(any(sub_labels))
                    site_ratio = float(sub_labels.sum()) / len(sub_labels)
                else:
                    sub_label = 0
                    site_ratio = 0.0

                ego_residues = [ego_nodes[i] for i in node_ids]

                subgraph = Data(
                    x=sub_x,
                    edge_index=sub_edge_index,
                    edge_attr=sub_edge_attr,
                    y=torch.tensor([sub_label], dtype=torch.long),
                    node_labels=sub_labels,
                    site_ratio=torch.tensor(site_ratio, dtype=torch.float),
                    ego_center=ego_nodes[ego_idx],
                    parent_id=template_id,
                    ego_nodes=ego_residues,
                )

                sub_subgraphs.append(subgraph)
                covered.update(node_ids.tolist())

        print(
            f"[INFO] Generated {len(sub_subgraphs)} sub-subgraphs from {len(positive_graphs)} positive subgraphs using coverage strategy."
        )
        return sub_subgraphs

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
                if hasattr(g, "node_labels"):
                    print(f"         node_labels: {g.node_labels.tolist()}")
                if hasattr(g, "edge_attr") and g.edge_attr is not None:
                    edge_feat = g.edge_attr[edge_idx].tolist()
                    print(
                        f"           {src_name} <--> {tgt_name} | features: {edge_feat}"
                    )
                else:
                    print(f"           {src_name} <--> {tgt_name}")

            print(f"         x:\n{g.x}\n")
