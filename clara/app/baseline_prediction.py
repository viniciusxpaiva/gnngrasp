import copy
import os
import random
import torch.nn as nn
import torch
import numpy as np
from torch_geometric.data import Data as PyGData
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from app.nn.hier.gat_context_node_classifier import GATContextNodeClassifier
from app.nn.hier.gat_node_classifier import GATNodeClassifier
from app.nn.hier.gcnB_node_classifier import GCNBiasNodeClassifier
from app.nn.hier.gcn_context_subgraph_classifier import GCNContextSubgraphClassifier
from app.nn.hier.gcn_context_node_classifier import GCNContextNodeClassifier
from app.nn.hier.gcn_node_classifier import GCNNodeClassifier
from app.nn.hier.gin_context_subgraph_classifier import GINContextSubgraphClassifier
from app.nn.hier.gin_node_classifier import GINNodeClassifier
from app.nn.hier.gat_context_subgraph_classifier import GATContextSubgraphClassifier
from app.nn.hier.gin_subgraph_classifier import GINSubgraphClassifier
from app.nn.hier.pna_context_subgraph_classifier import PNAContextSubgraphClassifier
from app.nn.hier.sage_node_classifier import SAGENodeClassifier
from app.utils.plotting import plot_training_loss_curve
from app.utils.prediction_utils import cleanup_cuda
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import k_hop_subgraph
from typing import List, Optional
from torch_geometric.utils import degree

from sklearn.metrics import (
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
)

from app.subgraphs_transform import (
    NodeStructTransform,
    NodeStructConfig,
    apply_subgraph_transforms,
)

from baselines.nodeimport.data_utils import make_longtailed_data_remove


class BaselinePrediction:
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
        self.loader = None
        self.train_data_class_weight = None
        self.edge_prop_dim = None
        self.output_dim = 1

        self.x_dim = None

    def train_node_classifier_old_no_context(self):
        """
        Train the node-level GNN classifier (GNN2) on nodes within subgraphs.

        Split-aware supervision to avoid leakage in 'coloring'+'all_nodes':
        - Message passing still uses all neighbors (transductive).
        - The loss is computed only on nodes that belong to train_mask.
        - In 'anchor' mode, we supervise only the anchor (no leakage by design).

        Binary one-vs-rest setup (always):
        * node_labeling_mode="anchor"    → supervise the anchor only (BATCHED)
        * node_labeling_mode="all_nodes" → supervise every node in the subgraph

        Flags in params["baseline"]:
        - node_labeling_mode: "anchor" | "all_nodes"
        - use_only_positive_subgraphs_for_node_train: str
            If True, use only subgraphs with sg.y == 1 for node-level training.
            (sg.y is set by `generate_binary_dataset_for_class` according to `mode=anchor|any_node`)
            If False (default), use all subgraphs (recommended baseline).

        Requirements:
        - self.train_loader exists (built from generate_binary_dataset_for_class)
        - Each subgraph carries:
            * node_labels (multiclass 0..C-1)
            * ego_center_local (tensor long; local anchor index)
            * global_node_ids (optional for training; needed for eval)
        - node_model returns per-node LOGITS (no sigmoid inside the model).
        """
        print(f"[!] Starting node-level training... | {self._device}")

        # --- Config ---
        params = self.prediction_params
        node_params = self.prediction_params["node_classifier"]
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        # use_only_pos = params.get("all_or_pos_subg_node_training", "all")
        batch_size = node_params.get("batch_size", 16)

        # --- Load train subgraphs from the pre-built train loader ---
        if self.train_loader is None:
            raise ValueError(
                "Train subgraph DataLoader (self.train_loader) not initialized."
            )
        subgraphs = list(self.train_loader.dataset)
        print(f"[INFO] Loaded all {len(subgraphs)} train subgraphs")
        self._assert_train_subgraphs_isolated(subgraphs)
        self._assert_train_edges_isolated(subgraphs)

        # --- DataLoader (batched for both modes) ---
        loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)

        # --- Loss (BCEWithLogits) and class-imbalance handling ---
        # Recompute pos_weight on the ACTUAL training pool (after any filtering).
        # In 'anchor' mode, positives = #anchors in target_class across the pool.
        # In 'all_nodes' mode, positives = #nodes with label==target_class across the pool.
        pos_weight = self._compute_pos_weight_for_node_loss(
            subgraphs, self.target_class, node_labeling_mode
        )

        # IMPORTANT:
        # For coloring+all_nodes with split-aware masking we need reduction='none'
        # so that we can zero-out loss for nodes outside train_mask, and then average.
        # For anchor mode, standard reduction='mean' is fine, but 'none' also works.
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self._device) if pos_weight is not None else None,
            reduction="none",  # we will handle the reduction manually
        )
        if pos_weight is not None:
            print(
                f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight={pos_weight.item():.3f}"
            )
        else:
            print(f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight=None")

        # --- Optimizer ---
        optimizer = torch.optim.Adam(
            self.node_model.parameters(),
            lr=node_params["lr"],
            weight_decay=node_params["weight_decay"],
        )

        loss_history = []

        # --- Training loop ---
        for epoch in range(1, node_params["epochs"] + 1):
            self.node_model.train()
            total_loss, correct, total = 0.0, 0, 0

            train_mask_global = self.data.train_mask.to(self._device)
            supervised_train_gids = set()

            for batch in loader:
                batch = batch.to(self._device)

                # Per-node logits on the concatenated batch graph. Shape → [sum_nodes_in_batch]
                logits = self.node_model(batch.x, batch.edge_index).view(-1)

                if node_labeling_mode == "anchor":
                    # --- Supervise ONLY the anchor of each graph in the batch (batched) ---
                    centers_global = self._gather_anchor_indices_in_batch(
                        batch
                    )  # [B_graphs]
                    logit_center = logits[centers_global]  # [B_graphs]

                    # Binary ground-truth for anchors (1 if anchor == target_class else 0)
                    y_center = (
                        batch.node_labels[centers_global] == self.target_class
                    ).float()

                    # loss_fn with reduction='none' → shape [B_graphs]; then mean
                    loss_vec = loss_fn(logit_center, y_center)
                    loss = loss_vec.mean()

                    # Metrics on anchors
                    with torch.no_grad():
                        preds = (torch.sigmoid(logit_center) >= 0.5).long()
                        correct += (preds == y_center.long()).sum().item()
                        total += y_center.numel()

                else:
                    # all_nodes supervision (COLORING):
                    # 1) Binarize labels vs target_class for ALL nodes in the batch.
                    y = (
                        (batch.node_labels == self.target_class)
                        .float()
                        .to(self._device)
                    )  # [sum_nodes]

                    # 2) Build a TRAIN mask aligned with the concatenated batch:
                    #    Map global ids back to the full-graph train_mask → avoids leakage.
                    if not hasattr(batch, "global_node_ids"):
                        raise RuntimeError(
                            "Subgraphs must carry 'global_node_ids' for split-aware masking."
                        )
                    train_mask_sub = train_mask_global[
                        batch.global_node_ids
                    ]  # [sum_nodes] bool

                    # 3) Per-node loss without reduction, then mask & average over train nodes only
                    loss_per_node = loss_fn(logits, y)  # [sum_nodes]
                    # sum over supervised nodes / count of supervised nodes
                    denom = max(1, int(train_mask_sub.sum().item()))
                    loss = (loss_per_node * train_mask_sub.float()).sum() / denom

                    gids = batch.global_node_ids.long().cpu().numpy()
                    mask = train_mask_sub.view(-1).cpu().numpy().astype(bool)
                    supervised_train_gids.update(gids[mask].tolist())

                    # Metrics only on supervised (train) nodes
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) >= 0.5).long()
                        correct += ((preds == y.long()) & train_mask_sub).sum().item()
                        total += train_mask_sub.sum().item()

                # --- Backprop & step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(loader))
            acc = correct / max(1, total)
            loss_history.append(avg_loss)

            if epoch % 20 == 0 or epoch == node_params["epochs"]:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

        # --- Plot training curve ---
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def _assert_train_subgraphs_isolated(self, train_subgraphs):
        """
        Ensure every node in each TRAIN subgraph belongs to data.train_mask.
        Prints a small report and raises AssertionError if leakage is found.
        """
        tm = self.data.train_mask
        total_nodes, leaked_nodes = 0, 0
        worst_leak = (None, 0)  # (idx, leaked_count)

        for i, sg in enumerate(train_subgraphs):
            if not hasattr(sg, "global_node_ids"):
                raise RuntimeError(
                    "Subgraphs must carry 'global_node_ids' for leakage checks."
                )
            gids = sg.global_node_ids
            allowed = tm[gids]
            total_nodes += gids.numel()
            leaks = int((~allowed).sum().item())
            leaked_nodes += leaks
            if leaks > worst_leak[1]:
                worst_leak = (i, leaks)

        leak_rate = 0.0 if total_nodes == 0 else 100.0 * leaked_nodes / total_nodes
        print(
            f"[LEAK-CHECK] train-subgraphs nodes={total_nodes} | leaked={leaked_nodes} ({leak_rate:.3f}%)"
        )
        if worst_leak[0] is not None and worst_leak[1] > 0:
            print(
                f"[LEAK-CHECK] worst subgraph idx={worst_leak[0]} leaked_nodes={worst_leak[1]}"
            )
        assert (
            leaked_nodes == 0
        ), "Leakage detected: some TRAIN subgraphs contain non-train nodes."

    def _assert_train_edges_isolated(self, train_subgraphs):
        """
        Ensure every edge in each TRAIN subgraph connects two TRAIN nodes.
        """
        tm = self.data.train_mask
        bad_edges = 0
        for i, sg in enumerate(train_subgraphs):
            gids = sg.global_node_ids
            # local->global map
            is_train_local = tm[gids]
            u, v = sg.edge_index
            # uma aresta é ruim se qualquer endpoint não for TRAIN
            bad = (~is_train_local[u]) | (~is_train_local[v])
            be = int(bad.sum().item())
            if be > 0:
                print(
                    f"[EDGE-LEAK] subgraph {i} has {be} edges touching non-train nodes."
                )
                bad_edges += be
        assert (
            bad_edges == 0
        ), "Leakage via edges: some TRAIN subgraphs have edges to non-train nodes."

    def prepare_dataset(self, ds_name):
        """
        Load a single global graph for the chosen dataset and build per-split ego-subgraphs.

        Splits:
            - Train centers: data.train_mask
            - Val   centers: data.val_mask
            - Test  centers: data.test_mask

        Center selection policy:
            - center_policy="anchor" → one subgraph per eligible center (no coverage constraint)
            - center_policy="color"  → greedy coverage: once a subgraph is built, all its nodes
                                    become ineligible as future centers.

        Notes:
            - Long-tailed shaping is applied ONLY on the train mask.
            - Transforms (e.g., E1 structural view) are applied to all splits.
            - gnn1_input_dim is resolved from the configured x_key on a probe subgraph.
        """
        dataset = self._get_pyg_dataset(ds_name)
        k = 0
        self.data = dataset[k]
        self.x_dim = self.data.x.size(1)

        # If using 'geom-gcn' or 'random' splits with [N,10] masks, select the k-th split and bool-ify.
        for attr in ["train_mask", "val_mask", "test_mask"]:
            m = getattr(self.data, attr)
            if m.dim() == 2:  # [N, 10]
                m = m[:, k].clone()  # -> [N]
                setattr(self.data, attr, m.to(torch.bool))

        override = False
        if override:
            print("Overriding mask splits")
            self._maybe_override_split(
                self.data,
                custom_split=override,
                # custom_split=self.prediction_params.get("custom_split", False),
                train_frac=self.prediction_params.get("split_train_frac", 0.60),
                val_frac=self.prediction_params.get("split_val_frac", 0.20),
                seed=self.prediction_params.get("split_seed", 0),
                stratified=self.prediction_params.get("split_stratified", True),
                min_per_class=self.prediction_params.get("split_min_per_class", 1),
            )

        # Optional: apply long-tailed shaping ONLY on train
        lt_info = self._apply_long_tailed_train_mask(self.data, ratio=10.0)

        # Ego-subgraph radius (k-hops)
        L = self.prediction_params["subg_neighbor_layers"]

        # Choose center selection policy
        center_policy = self.prediction_params.get("subg_gen_method", "color").lower()
        assert center_policy in (
            "anchor",
            "color",
        ), "center_policy must be 'anchor' or 'color'"

        subgraph_builder = (
            self._build_anchor_subraphs_from_mask
            if center_policy == "anchor"
            else self._build_coloring_subgraphs_from_mask
        )

        # ----------------------------
        # Build TRAIN / VAL / TEST raw subgraphs
        # ----------------------------
        self.train_subgraphs_raw = subgraph_builder(
            data=self.data,
            center_mask=self.data.train_mask,
            num_layers=L,
            shuffle=True,
            verbose=True,
            # allowed_nodes_mask=self.data.train_mask,
        )

        self.val_subgraphs_raw = subgraph_builder(
            data=self.data,
            center_mask=self.data.val_mask,
            num_layers=L,
            shuffle=False,  # keep val deterministic
            verbose=True,
        )

        self.test_subgraphs_raw = subgraph_builder(
            data=self.data,
            center_mask=self.data.test_mask,
            num_layers=L,
            shuffle=False,  # keep test deterministic
            verbose=True,
        )

        print(
            f"[✓] Subgraphs generated: "
            f"{len(self.train_subgraphs_raw)} train | {len(self.val_subgraphs_raw)} val | {len(self.test_subgraphs_raw)} test "
            f"| policy={center_policy}, k={L}"
        )

        # ----------------------------
        # Apply transforms (e.g., E1) to ALL splits
        # ----------------------------
        transforms = []
        if self.prediction_params.get("use_struct_view", True):
            # E1 config: local within the subgraph; normalization as configured.
            e1_cfg = NodeStructConfig(
                use_global=False,
                max_nodes_for_dense=self.prediction_params.get(
                    "struct_max_nodes_dense", 2000
                ),
                normalize=self.prediction_params.get(
                    "struct_normalize", {"method": "layernorm"}
                ),
                include=None,  # or a subset: ["deg","logdeg","cluster","pr","core"]
            )
            transforms.append(NodeStructTransform(cfg=e1_cfg))

        # Apply transforms and attach views (e.g., x_view_struct) to all splits
        self.train_subgraphs = apply_subgraph_transforms(
            self.train_subgraphs_raw, transforms
        )
        self.val_subgraphs = apply_subgraph_transforms(
            self.val_subgraphs_raw, transforms
        )
        self.test_subgraphs = apply_subgraph_transforms(
            self.test_subgraphs_raw, transforms
        )

        # ----------------------------
        # Resolve GNN1 input dim from the configured x_key
        # ----------------------------
        x_key = self.prediction_params.get(
            "gnn1_x_key", "x"
        )  # e.g., "x_view_struct" or "x_view_concat"
        # Choose a probe subgraph from train; fall back to val or test if needed
        if len(self.train_subgraphs) > 0:
            probe = self.train_subgraphs[0]
        elif len(self.val_subgraphs) > 0:
            probe = self.val_subgraphs[0]
        else:
            probe = self.test_subgraphs[0]

        self.gnn1_input_dim = self._resolve_x_dim_for_key(probe, x_key)

        print(
            f"[dims] original x_dim={self.x_dim} | gnn1_input_dim({x_key})={self.gnn1_input_dim}"
        )

        # Optional: quick E1 sanity check (on train if available)
        d0 = None
        if len(self.train_subgraphs) > 0:
            d0 = self.train_subgraphs[0]
        elif len(self.val_subgraphs) > 0:
            d0 = self.val_subgraphs[0]
        elif len(self.test_subgraphs) > 0:
            d0 = self.test_subgraphs[0]

        if d0 is not None:
            has_struct = hasattr(d0, "x_view_struct")
            print(
                f"[E1] x_view_struct attached? {has_struct} | shape={getattr(d0, 'x_view_struct', None).shape if has_struct else None}"
            )

        # Return number of classes
        return int(self.data.y.max().item() + 1)

    def generate_binary_dataset_for_class(
        self,
        target_class: int,
        mode: str,  # "anchor" | "color"
    ):
        """
        Build a binary subgraph dataset (one-vs-rest) for the given target_class.

        Labeling modes:
        - "anchor":  subgraph label = 1 iff the ANCHOR node belongs to target_class.
        - "color": subgraph label = 1 iff the subgraph CONTAINS at least one node of target_class.

        Notes:
        - Works for both center policies (anchor or coloring). The 'coloring' mode is
            commonly paired with coloring, but you can mix as you like.
        - Keeps original multiclass node labels (0..C-1) in `sg.node_labels` for node-level training.
        - Stores the subgraph binary label in `sg.y` as shape [1], dtype long {0,1}.
        """
        mode = mode.lower().strip()
        assert mode in ("anchor", "color"), "mode must be 'anchor' or 'coloring'"

        self.target_class = target_class
        print(
            f"[→] Generating binary subgraph dataset | target_class={self.target_class} | mode={mode}"
        )

        if not hasattr(self, "train_subgraphs_raw") or not hasattr(
            self, "test_subgraphs_raw"
        ):
            raise RuntimeError(
                "You must run `prepare_dataset()` first to generate train/test subgraphs."
            )

        def _label_subgraphs(subgraphs, split: str):
            labeled = []
            # choose which mask to use for this split
            if split == "train":
                split_mask_global = self.data.train_mask
            elif split == "val":
                split_mask_global = self.data.val_mask
            else:
                split_mask_global = self.data.test_mask
            for sg in subgraphs:
                sgc = copy.deepcopy(sg)

                if mode == "anchor":
                    # Positive if the anchor node is of target_class
                    anchor_gid = int(sgc.ego_center_global)
                    anchor_cls = int(self.data.y[anchor_gid].item())
                    sgc.y = torch.tensor(
                        [1 if anchor_cls == target_class else 0], dtype=torch.long
                    )
                else:
                    mask_sub = split_mask_global[sgc.global_node_ids]  # [|V_sub|] bool
                    labels_sub = sgc.node_labels  # [|V_sub|] (multiclass)
                    has_pos = ((labels_sub == target_class) & mask_sub).any().item()
                    sgc.y = torch.tensor([1 if has_pos else 0], dtype=torch.long)

                labeled.append(sgc)

            return labeled

        # --- Label each split ---
        train_labeled = _label_subgraphs(self.train_subgraphs_raw, split="train")
        val_labeled = _label_subgraphs(self.val_subgraphs_raw, split="val")
        test_labeled = _label_subgraphs(self.test_subgraphs_raw, split="test")

        def _count_pos(subgraphs):
            return sum(int(sg.y.item()) for sg in subgraphs)

        print("[DEBUG] Subgraph label counts:")
        print(
            f" - Train: {_count_pos(train_labeled)} positive | {len(train_labeled) - _count_pos(train_labeled)} negative"
        )
        print(
            f" - Val  : {_count_pos(val_labeled)} positive | {len(val_labeled) - _count_pos(val_labeled)} negative"
        )
        print(
            f" - Test : {_count_pos(test_labeled)} positive | {len(test_labeled) - _count_pos(test_labeled)} negative"
        )

        # --- Sanity checks ---
        for split_name, subgraphs in [
            ("train", train_labeled),
            ("val", val_labeled),
            ("test", test_labeled),
        ]:
            if len(subgraphs) == 0:
                print(
                    f"[⚠️] Empty {split_name} split after labeling (check masks/mode)."
                )
            if _count_pos(subgraphs) == 0:
                print(
                    f"[⚠️] No positive subgraphs in {split_name} for target_class={target_class}."
                )

        # --- Dataloaders ---
        self.train_loader = DataLoader(train_labeled, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_labeled, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(test_labeled, batch_size=16, shuffle=False)

        # Combined loader (train+test) — still optional, depends on downstream use
        self.loader = DataLoader(
            train_labeled + test_labeled, batch_size=16, shuffle=True
        )

        print(
            f"[✓] Dataset prepared for class {target_class}: "
            f"{len(train_labeled)} train | {len(val_labeled)} val | {len(test_labeled)} test"
        )

    def initialize_subgraph_classifier(self):
        """
        Instantiate a subgraph-level GNN classifier for subgraph classification.
        Uses parameters specified under the 'subgraph_classifier' block in HIER_PARAMS.

        Supported gnn_type (case-insensitive):
        - 'GAT' → GATContextSubgraphClassifier
        - 'PNA' → PNAContextSubgraphClassifier
        - 'GIN' → GINContextSubgraphClassifier
        - 'GCN' → GCNContextSubgraphClassifier
        """
        params = self.prediction_params["subgraph_classifier"]
        metric_mode = str(
            params.get("metric_mode", "f_beta")
        ).lower()  # "recall" or "f_beta"
        beta = float(params.get("beta", 2.0))  # only relevant for Fβ

        print(
            f"Initializing subgraph classifier | {params['gnn_type']} | "
            f"{params['num_layers']} layers | {params['norm_type']} norm | {params['pool_type']} pooling | {metric_mode} | {beta}"
        )

        gnn_type = params["gnn_type"].upper()

        if gnn_type == "GAT":
            self.subgraph_model = GATContextSubgraphClassifier(
                input_dim=self.x_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                num_heads=params["num_heads"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

        elif gnn_type == "PNA":
            # num_heads is accepted in signature for API compat but unused by PNA
            self.subgraph_model = PNAContextSubgraphClassifier(
                input_dim=self.x_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
                deg=self.deg_hist,
            ).to(self._device)

        elif gnn_type == "GIN":
            # Use the context-ready GIN variant
            self.subgraph_model = GINContextSubgraphClassifier(
                input_dim=self.x_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

        elif gnn_type == "GCN":
            self.subgraph_model = GCNContextSubgraphClassifier(
                input_dim=self.gnn1_input_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

        else:
            raise ValueError("[ERROR] Unsupported subgraph gnn_type.")

        # --- Sanity: ensure the model exposes `graph_emb_dim` (used by node-level context fusion).
        # For GAT with a single layer, width = hidden_dim * num_heads; otherwise it's hidden_dim.
        if not hasattr(self.subgraph_model, "graph_emb_dim"):
            if gnn_type == "GAT" and params["num_layers"] == 1:
                self.subgraph_model.graph_emb_dim = (
                    params["hidden_dim"] * params["num_heads"]
                )
            else:
                self.subgraph_model.graph_emb_dim = params["hidden_dim"]

    def initialize_node_classifier(self):
        """
        Instantiate a node-level GNN classifier for residue/node prediction within subgraphs.
        Uses parameters under 'node_classifier' in HIER_PARAMS.

        Supports bias for GCN via GCNBiasNodeClassifier when fusion_mode != 'none':
        - fusion_mode = 'none'   -> standard classifier (no context)
        - fusion_mode = 'concat' -> concatenates per-node context vectors to node embeddings
        """
        params = self.prediction_params["node_classifier"]
        use_subgraph_classifier = self.prediction_params.get(
            "use_subgraph_classifier", False
        )
        gnn_type = params["gnn_type"].upper()
        fusion_mode = params.get("fusion_mode", "concat")  # 'film' | 'concat'
        num_classes = params.get("num_classes", 1)  # keep 1 for BCE by default

        fusion_info = (
            f" | fusion_mode={fusion_mode} | context_norm={params.get('context_norm','none')}"
            if use_subgraph_classifier
            else ""
        )
        print(
            f"Initializing node classifier | {gnn_type} | {params['num_layers']} layers{fusion_info}"
        )

        # Helper: infer context_dim from subgraph model if needed (concat mode)
        def _infer_context_dim():
            explicit_ctx = params.get("context_dim", None)
            if explicit_ctx is not None:
                return explicit_ctx
            if getattr(self, "subgraph_model", None) is None:
                raise RuntimeError(
                    "subgraph_model must be initialized before node_model when using fusion_mode='concat'."
                )
            if not hasattr(self.subgraph_model, "graph_emb_dim"):
                raise RuntimeError(
                    "subgraph_model must expose 'graph_emb_dim' for fusion_mode='concat'."
                )
            return int(self.subgraph_model.graph_emb_dim)

        # === Instantiate per GNN type ===
        if gnn_type == "GCN":
            if not use_subgraph_classifier:
                # baseline
                self.node_model = GCNNodeClassifier(
                    input_dim=self.x_dim,
                    hidden_dim=params["hidden_dim"],
                    dropout=params["dropout"],
                    num_layers=params["num_layers"],
                    norm_type=params["norm_type"],
                ).to(self._device)
            else:
                # bias-capable version
                # context_dim = _infer_context_dim()
                if fusion_mode == "film":
                    self.node_model = GCNContextNodeClassifier(
                        input_dim=self.x_dim,
                        hidden_dim=params["hidden_dim"],
                        dropout=params["dropout"],
                        num_layers=params["num_layers"],
                        norm_type=params["norm_type"],
                        num_classes=num_classes,
                        fusion_mode=fusion_mode,  # 'none' | 'concat' | 'film'
                        context_dim=context_dim,  # >0 only if 'concat'
                        use_layer_dropout=True,
                        ctx_bottleneck_dim=64,
                        context_dropout_p=0.1,
                        film_bias=True,
                    ).to(self._device)

                else:
                    self.node_model = GCNContextNodeClassifier(
                        input_dim=self.x_dim,
                        hidden_dim=params["hidden_dim"],
                        dropout=params["dropout"],
                        num_layers=params["num_layers"],
                        norm_type=params["norm_type"],
                        num_classes=num_classes,
                        fusion_mode="none",  # 'none' | 'concat' | 'film'
                        # context_dim=context_dim,  # >0 only if 'concat'
                        use_layer_dropout=True,
                        ctx_bottleneck_dim=64,
                        context_dropout_p=0.1,
                        film_bias=False,
                    ).to(self._device)

        elif gnn_type == "GAT":
            # Ainda não criamos GATBiasNodeClassifier.
            # Use baseline quando fusion_mode='none'; caso contrário, dispare um aviso/erro.
            if not use_subgraph_classifier:
                self.node_model = GATNodeClassifier(
                    input_dim=self.x_dim,
                    hidden_dim=params["hidden_dim"],
                    num_heads=params["num_heads"],
                    dropout=params["dropout"],
                    num_layers=params["num_layers"],
                    norm_type=params["norm_type"],
                ).to(self._device)
            else:
                self.node_model = GATContextNodeClassifier(
                    input_dim=self.x_dim,
                    hidden_dim=params["hidden_dim"],
                    dropout=params["dropout"],
                    num_layers=params["num_layers"],
                    norm_type=params["norm_type"],
                    num_classes=num_classes,
                    num_heads=params["num_heads"],
                    fusion_mode="none",  # 'none' | 'concat' | 'film'
                    # context_dim=context_dim,  # >0 only if 'concat'
                    use_layer_dropout=True,
                    ctx_bottleneck_dim=64,
                    context_dropout_p=0.1,
                    film_bias=False,
                ).to(self._device)

        elif gnn_type == "GIN":
            if not use_subgraph_classifier:
                self.node_model = GINNodeClassifier(
                    input_dim=self.x_dim,
                    hidden_dim=params["hidden_dim"],
                    dropout=params["dropout"],
                    num_layers=params["num_layers"],
                    norm_type=params["norm_type"],
                ).to(self._device)
            else:
                raise NotImplementedError(
                    "fusion_mode is currently implemented only for GCN. "
                    "Create a GINBiasNodeClassifier to enable bias for GIN."
                )

        elif gnn_type in ("SAGE", "GRAPH_SAGE"):
            if not use_subgraph_classifier:
                self.node_model = SAGENodeClassifier(
                    input_dim=self.x_dim,
                    hidden_dim=params["hidden_dim"],
                    dropout=params["dropout"],
                    num_layers=params["num_layers"],
                    norm_type=params["norm_type"],
                    aggr=params.get("aggr", "mean"),
                ).to(self._device)
            else:
                raise NotImplementedError(
                    "fusion_mode  is currently implemented only for GCN. "
                    "Create a SAGEBiasNodeClassifier to enable bias for SAGE."
                )
        else:
            raise ValueError("[ERROR] No valid GNN type for node classifier.")

    def train_subgraph_classifier(self):
        """
        Train GNN1 (subgraph-level classifier) with f1-score-oriented strategy.

        Key features:
        - Optimizer: AdamW with weight decay
        - LR schedule: linear warmup + cosine decay
        - Loss: BCEWithLogits + class imbalance correction (pos_weight, with optional boost for recall)
        - Gradient clipping for stability
        - Validation-driven:
            * Early stopping (by F1) + best checkpoint restore
            * Threshold calibration (by F1)
        - Learning curves saved (train loss, val F1, val PR-AUC)
        """
        print(f"[!] Starting subgraph-level training... | Device: {self._device}")
        params = self.prediction_params["subgraph_classifier"]

        # Feature view selection for GNN1 (e.g., "x", "x_view_struct", "x_view_concat")
        x_key = self.prediction_params.get("gnn1_x_key", "x")
        if hasattr(self.subgraph_model, "x_key"):
            self.subgraph_model.x_key = x_key

        # Put model in train mode
        self.subgraph_model.train()

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.subgraph_model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = self._make_warmup_cosine_scheduler(
            optimizer,
            total_epochs=params["epochs"],
            warmup_epochs=None,  # default ~5% of epochs
            min_lr_scale=0.1,  # decay down to 10% of base LR
        )

        # Loss (with pos_weight computed on train)
        pos_weight = self._compute_pos_weight_from_loader(
            self.train_loader, self._device
        )
        loss_fn = self._make_bce_with_logits(pos_weight=pos_weight)

        # Early stopping config (by F1 on validation)
        use_val = getattr(self, "val_loader", None) is not None
        patience = int(params.get("early_stop_patience", 30))
        min_delta = float(params.get("early_stop_min_delta", 0.002))  # 0.2 p.p. in F1
        best_val_f1 = -1.0
        best_thresh = float(params.get("prediction_threshold", 0.5))
        best_state = None
        best_epoch = -1
        epochs_no_improve = 0

        # History for curves
        train_loss_hist = []
        val_f1_hist = []
        val_prauc_hist = []

        total_epochs = int(params["epochs"])

        for epoch in range(1, total_epochs + 1):
            epoch_loss = 0.0
            y_true_ep, y_prob_ep = [], []

            # ---------------------------
            # Train one epoch
            # ---------------------------
            for batch in self.train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                logits = self.subgraph_model.forward_from_data(batch)  # [B,1]
                labels = batch.y.float().unsqueeze(1)  # [B,1]

                loss = loss_fn(logits, labels)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.subgraph_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())

                # Quick train stats (at threshold 0.5 — for info only)
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                y_prob_ep.extend(probs.tolist())
                y_true_ep.extend(labels.detach().cpu().numpy().flatten().tolist())

            # LR schedule step (once per epoch)
            scheduler.step()

            # Save avg train loss for the epoch
            avg_train_loss = epoch_loss / max(1, len(self.train_loader))
            train_loss_hist.append(avg_train_loss)

            # Optional: quick train metrics @ 0.5 (informative, not used for ES)
            y_pred_ep = (np.array(y_prob_ep) >= 0.5).astype(int)
            f1_train = f1_score(y_true_ep, y_pred_ep, average="binary", zero_division=0)
            mcc_train = (
                matthews_corrcoef(y_true_ep, y_pred_ep)
                if len(set(y_true_ep)) > 1
                else 0.0
            )

            # ---------------------------
            # Validation (for ES + threshold)
            # ---------------------------
            this_val_f1 = None
            this_val_ap = None

            if use_val:
                self.subgraph_model.eval()
                vy_true, vy_prob = [], []
                with torch.no_grad():
                    for vbatch in self.val_loader:
                        vbatch = vbatch.to(self._device)
                        vlogits = self.subgraph_model.forward_from_data(vbatch)  # [B,1]
                        vprobs = torch.sigmoid(vlogits).cpu().numpy().flatten()
                        vy_prob.extend(vprobs.tolist())
                        vy_true.extend(vbatch.y.cpu().numpy().flatten().tolist())
                self.subgraph_model.train()

                vy_true = np.asarray(vy_true, dtype=int)
                vy_prob = np.asarray(vy_prob, dtype=float)

                # Choose threshold that maximizes F1 on validation
                t_opt, f1_val = self._best_threshold_by_f1(vy_true, vy_prob, beta=1.0)
                this_val_f1 = float(f1_val)
                val_f1_hist.append(this_val_f1)

                # PR-AUC on validation (probability-based)
                try:
                    this_val_ap = float(average_precision_score(vy_true, vy_prob))
                except Exception:
                    this_val_ap = None
                val_prauc_hist.append(
                    this_val_ap if this_val_ap is not None else np.nan
                )

                # Early stopping check (improvement in F1_val)
                if this_val_f1 > best_val_f1 + min_delta:
                    best_val_f1 = this_val_f1
                    best_thresh = float(t_opt)
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.subgraph_model.state_dict().items()
                    }
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"[EarlyStop] No F1_val improvement > {min_delta:.4f} for {patience} epochs. Stopping at epoch {epoch}."
                        )
                        break

            # ------------- Logging -------------
            if epoch % 20 == 0 or epoch == total_epochs:
                lr_now = scheduler.get_last_lr()[0]
                if use_val:
                    ap_str = f"{this_val_ap:.4f}" if this_val_ap is not None else "n/a"
                    print(
                        f"Epoch {epoch:03d} | "
                        f"TrainLoss: {avg_train_loss:.4f} | TrainF1@0.5: {f1_train:.4f} | TrainMCC@0.5: {mcc_train:.4f} | "
                        f"ValF1*: {this_val_f1:.4f} | ValPR-AUC: {ap_str} | "
                        f"LR: {lr_now:.2e}"
                    )
                else:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"TrainLoss: {avg_train_loss:.4f} | TrainF1@0.5: {f1_train:.4f} | TrainMCC@0.5: {mcc_train:.4f} | "
                        f"LR: {lr_now:.2e}"
                    )

        # ---------------------------
        # Restore best checkpoint & persist best threshold (if we had validation)
        # ---------------------------
        if use_val and best_state is not None:
            self.subgraph_model.load_state_dict(best_state)
            self.prediction_params["subgraph_classifier"][
                "prediction_threshold"
            ] = best_thresh
            print(
                f"[Best] Restored best subgraph model @ epoch {best_epoch} (F1_val={best_val_f1:.4f}, thr={best_thresh:.3f})"
            )

        # ---------------------------
        # Save learning curves
        # ---------------------------
        out_dir = self.dirs["output"]["prot_out_dir"]
        # 1) Train loss curve
        training_loss_curve_out_path = os.path.join(
            out_dir, f"subgraph_classifier_train_loss_class_{self.target_class}.html"
        )
        plot_training_loss_curve(
            {"train_loss": train_loss_hist}, output_path=training_loss_curve_out_path
        )

        # 2) Validation curves (if available)
        if use_val and len(val_f1_hist) > 0:
            val_f1_curve_out_path = os.path.join(
                out_dir, f"subgraph_classifier_val_f1_class_{self.target_class}.html"
            )
            plot_training_loss_curve(
                {"val_f1": val_f1_hist}, output_path=val_f1_curve_out_path
            )

            # PR-AUC curve (optional)
            if np.isfinite(np.nanmean(val_prauc_hist)):
                val_pr_curve_out_path = os.path.join(
                    out_dir,
                    f"subgraph_classifier_val_prauc_class_{self.target_class}.html",
                )
                plot_training_loss_curve(
                    {"val_pr_auc": val_prauc_hist}, output_path=val_pr_curve_out_path
                )

        # If there was no validation, keep whatever threshold was configured
        if not use_val:
            print(
                "[Info] No validation loader found: ran full training with fixed threshold "
                f"{self.prediction_params['subgraph_classifier'].get('prediction_threshold', 0.5):.3f}"
            )

    def train_node_classifier(self):
        """
        Train the node-level GNN classifier (GNN2) with:
        - Split-aware supervision (anchor | all_nodes).
        - Optional context from GNN1 (fusion_mode: 'none' | 'concat' | 'film').
        - Class-imbalance handling via pos_weight.
        - LR warmup + cosine decay (optional).
        - Validation at each epoch (val subgraphs).
        - Early stopping on validation F1 (with patience/min_delta).
        - Threshold calibration by validation F1 (store in params).

        Notes:
        * Validation follows the same supervision mode used in training.
        * For 'all_nodes', only nodes that belong to the split's mask are used
            to compute loss/metrics (avoiding leakage).
        * For 'anchor', supervision/metrics são feitos apenas no nó âncora.
        """
        print(
            f"[!] Starting node-level training (with validation) ... | Device: {self._device}"
        )

        # --- Config & bookkeeping ---
        params = self.prediction_params
        node_params = params["node_classifier"]
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        batch_size = node_params.get("batch_size", 16)
        epochs = node_params.get("epochs", 100)
        context_anneal = node_params.get(
            "context_anneal", "none"
        ).lower()  # 'none'|'linear'|'power'

        # Early stopping & threshold calibration
        patience = int(node_params.get("early_stop_patience", 20))
        min_delta = float(node_params.get("early_stop_min_delta", 1e-4))
        base_threshold = float(node_params.get("prediction_threshold", 0.5))
        best_val_f1, best_threshold = -1.0, base_threshold
        best_epoch = 0
        epochs_no_improve = 0

        # --- Derive 'use_context' ---
        fusion_mode = getattr(
            self.node_model, "fusion_mode", "concat"
        )  # 'none'|'concat'|'film'
        use_context = bool(
            params.get("use_subgraph_classifier", False)
        ) and fusion_mode in {"concat", "film"}

        if use_context:
            if not hasattr(self, "subgraph_model") or self.subgraph_model is None:
                print(
                    "[WARN] use_subgraph_classifier=True but subgraph_model is missing; proceeding WITHOUT bias."
                )
                use_context = False
            else:
                self.subgraph_model.eval()  # frozen context provider
                ctx_dim = getattr(self.node_model, "context_dim", None)
                gnn1_dim = getattr(self.subgraph_model, "graph_emb_dim", None)
                if ctx_dim is not None and gnn1_dim is not None and ctx_dim != gnn1_dim:
                    raise ValueError(
                        f"[ERROR] context_dim ({ctx_dim}) must match subgraph_model.graph_emb_dim ({gnn1_dim}) for '{fusion_mode}'."
                    )
                print(
                    f"[INFO] Using subgraph_model context | fusion_mode={fusion_mode} | context_norm={node_params.get('context_norm','none')}"
                )
        else:
            print("[INFO] Training node_model WITHOUT subgraph context/bias.")

        # --- Check loaders ---
        if self.train_loader is None:
            raise ValueError(
                "Train subgraph DataLoader (self.train_loader) not initialized."
            )
        if getattr(self, "val_loader", None) is None:
            print(
                "[WARN] val_loader is None. Proceeding without validation/early stopping/threshold calibration."
            )
        if getattr(self, "test_loader", None) is None:
            print("[WARN] test_loader missing (only matters for later evaluation).")

        # Materialize subgraphs to allow reusing the same pool consistently
        train_subgraphs = list(self.train_loader.dataset)
        print(f"[INFO] Using {len(train_subgraphs)} subgraphs for node training.")

        # DataLoaders
        train_loader = DataLoader(train_subgraphs, batch_size=batch_size, shuffle=True)
        val_loader = self.val_loader  # already batched

        # Split masks (global)
        train_mask_global = self.data.train_mask.to(self._device)
        val_mask_global = getattr(self.data, "val_mask", None)
        if val_loader is not None and val_mask_global is None:
            raise RuntimeError("val_loader provided but self.data.val_mask is missing.")

        # --- Loss & class imbalance ---
        pos_weight = self._compute_pos_weight_for_node_loss(
            train_subgraphs, self.target_class, node_labeling_mode
        )
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self._device) if pos_weight is not None else None,
            reduction="none",  # manual masking/averaging
        )
        if pos_weight is not None:
            print(
                f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight={pos_weight.item():.3f}"
            )
        else:
            print(f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight=None")

        # --- Optimizer & scheduler ---
        optimizer = torch.optim.Adam(
            self.node_model.parameters(),
            lr=node_params["lr"],
            weight_decay=node_params["weight_decay"],
        )
        # Optional LR warmup+cosine (set warmup_epochs=None → 5% default)
        scheduler = self._make_warmup_cosine_scheduler(
            optimizer,
            total_epochs=epochs,
            warmup_epochs=None,
            min_lr_scale=0.1,
        )

        # --- Histories for plotting ---
        train_loss_hist, val_loss_hist, val_f1_hist = [], [], []

        # ----------------------- Helpers -----------------------
        def _build_context_nodes(batch, alpha: float):
            """Build per-node context vectors from GNN1 (optional), with annealing/normalization."""
            if not use_context:
                return None
            with torch.no_grad():
                Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                    batch
                )  # [B_graphs, d]
            ctx_nodes = Z_s[batch.batch]  # broadcast to nodes [N, d]
            ctx_mode = node_params.get("context_norm", "none")
            ctx_nodes = self._normalize_context_vectors(ctx_nodes, mode=ctx_mode)
            # Trainer-side annealing for concat (FiLM handled via model method)
            return alpha * ctx_nodes if fusion_mode == "concat" else ctx_nodes

        def _epoch_forward(
            loader, split: str, threshold: float = 0.5, eval_mode: bool = False
        ):
            """
            One pass over a loader to compute:
            - Average BCE loss over supervised nodes of the split.
            - y_true (0/1) and y_prob (sigmoid) for F1-calibration.
            Masking/supervision matches node_labeling_mode.
            """
            if eval_mode:
                self.node_model.eval()
            else:
                self.node_model.train()

            if split == "train":
                split_mask_global = train_mask_global
            elif split == "val":
                split_mask_global = val_mask_global.to(self._device)
            else:
                raise ValueError("split must be 'train' or 'val'.")

            total_loss, denom_total = 0.0, 0
            y_true_all, y_prob_all = [], []

            with torch.set_grad_enabled(not eval_mode):
                for batch in loader:
                    if not eval_mode:
                        optimizer.zero_grad()
                    batch = batch.to(self._device)

                    # annealing α in [0,1]
                    # (only makes sense for training; at eval α=1.0)
                    if context_anneal == "linear":
                        alpha_curr = (
                            float(epoch) / float(max(1, epochs))
                            if not eval_mode
                            else 1.0
                        )
                    elif context_anneal == "power":
                        alpha_curr = (
                            (float(epoch) / float(max(1, epochs))) ** 1.5
                            if not eval_mode
                            else 1.0
                        )
                    else:
                        alpha_curr = 1.0

                    # FiLM: set context scale
                    if (
                        use_context
                        and fusion_mode == "film"
                        and hasattr(self.node_model, "set_context_scale")
                    ):
                        self.node_model.set_context_scale(alpha_curr)

                    # Build (optional) context
                    ctx_nodes = _build_context_nodes(batch, alpha=alpha_curr)

                    # Forward
                    if use_context:
                        logits = self.node_model(
                            batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                        ).view(-1)
                    else:
                        logits = self.node_model(batch.x, batch.edge_index).view(-1)

                    # Supervision & masking
                    if node_labeling_mode == "anchor":
                        centers_global = self._gather_anchor_indices_in_batch(
                            batch
                        )  # [B_graphs]
                        logit_center = logits[centers_global]  # [B_graphs]
                        y_center = (
                            batch.node_labels[centers_global] == self.target_class
                        ).float()
                        loss_vec = loss_fn(logit_center, y_center)
                        loss = loss_vec.mean()

                        # For calibration: collect probs/labels for all anchors in this split
                        probs = (
                            torch.sigmoid(logit_center).detach().cpu().numpy().flatten()
                        )
                        y_true = y_center.detach().cpu().numpy().astype(int).flatten()

                        denom = y_center.numel()

                    else:
                        # all_nodes, split-aware masking
                        y = (
                            (batch.node_labels == self.target_class)
                            .float()
                            .to(self._device)
                        )  # [N]
                        if not hasattr(batch, "global_node_ids"):
                            raise RuntimeError(
                                "Subgraphs must carry 'global_node_ids' for split-aware masking."
                            )
                        split_mask_sub = split_mask_global[
                            batch.global_node_ids
                        ]  # [N] bool

                        loss_per_node = loss_fn(logits, y)  # [N]
                        denom = int(split_mask_sub.sum().item())
                        if denom > 0:
                            loss = (
                                loss_per_node * split_mask_sub.float()
                            ).sum() / denom
                        else:
                            # No supervised nodes for this mini-batch; skip loss contribution
                            loss = logits.new_tensor(0.0)
                            denom = 0

                        # For calibration: collect probs/labels ONLY for masked nodes (this split)
                        if denom > 0:
                            probs = (
                                torch.sigmoid(logits[split_mask_sub])
                                .detach()
                                .cpu()
                                .numpy()
                                .flatten()
                            )
                            y_true = (
                                y[split_mask_sub]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(int)
                                .flatten()
                            )
                        else:
                            probs, y_true = np.array([]), np.array([])

                    # Optimize (train only)
                    if not eval_mode:
                        loss.backward()
                        optimizer.step()

                    # Accumulate
                    total_loss += float(loss.item()) if denom > 0 else 0.0
                    denom_total += max(1, denom)  # avoid /0
                    if probs.size > 0:
                        y_prob_all.extend(probs.tolist())
                        y_true_all.extend(y_true.tolist())

            avg_loss = total_loss / max(1, denom_total)
            y_true_all = np.array(y_true_all, dtype=int)
            y_prob_all = np.array(y_prob_all, dtype=float)

            # F1 at a *fixed* threshold (puremente informativo)
            if y_true_all.size > 0:
                from sklearn.metrics import f1_score

                y_pred_all = (y_prob_all >= threshold).astype(int)
                f1_fixed = f1_score(
                    y_true_all, y_pred_all, average="binary", zero_division=0
                )
            else:
                f1_fixed = 0.0

            return avg_loss, y_true_all, y_prob_all, f1_fixed

        # ----------------------- Training loop -----------------------
        for epoch in range(1, epochs + 1):
            # 1) Train epoch
            train_loss, _, _, f1_train_fixed = _epoch_forward(
                train_loader, split="train", threshold=base_threshold, eval_mode=False
            )
            train_loss_hist.append(train_loss)

            # 2) Step scheduler (once per epoch)
            scheduler.step()

            # 3) Validation epoch (if available)
            if val_loader is not None:
                with torch.no_grad():
                    val_loss, vy_true, vy_prob, f1_val_fixed = _epoch_forward(
                        val_loader,
                        split="val",
                        threshold=base_threshold,
                        eval_mode=True,
                    )
                val_loss_hist.append(val_loss)

                # 4) Threshold calibration by F1 on validation
                if vy_true.size > 0:
                    t_opt, val_f1 = self._best_threshold_by_f1(
                        vy_true, vy_prob, beta=1.0
                    )
                else:
                    t_opt, val_f1 = base_threshold, 0.0

                val_f1_hist.append(val_f1)

                # 5) Early stopping on val F1
                improved = (val_f1 - best_val_f1) > min_delta
                if improved:
                    best_val_f1 = val_f1
                    best_threshold = float(t_opt)
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Logging
                if epoch % 20 == 0 or epoch == epochs:
                    curr_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else node_params["lr"]
                    )
                    print(
                        f"Epoch {epoch:03d} | "
                        f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                        f"ValF1*(best-thr): {val_f1:.4f} (thr={t_opt:.3f}) | "
                        f"LR: {curr_lr:.2e}"
                    )

                if epochs_no_improve >= patience:
                    print(
                        f"[EarlyStop] No val F1 improvement > {min_delta} for {patience} epochs. Stopping at epoch {epoch}."
                    )
                    break
            else:
                # No validation: only log train
                if epoch % 20 == 0 or epoch == epochs:
                    curr_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else node_params["lr"]
                    )
                    print(
                        f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | LR: {curr_lr:.2e}"
                    )

        # ----------------------- End of training -----------------------
        # Plot training/validation loss curves
        out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
        )
        series = {1: train_loss_hist}
        if val_loader is not None and len(val_loss_hist) > 0:
            series[2] = val_loss_hist
        plot_training_loss_curve(series, output_path=out_path)

        # Persist calibrated threshold (if validation existed)
        if val_loader is not None and best_val_f1 >= 0:
            self.prediction_params["node_classifier"][
                "prediction_threshold"
            ] = best_threshold
            print(
                f"[Calib] Best Val F1={best_val_f1:.4f} at threshold={best_threshold:.3f} (epoch={best_epoch})"
            )
        else:
            print(
                f"[Calib] No validation available. Keeping configured threshold={base_threshold:.3f}."
            )

    def train_node_classifier_old_no_context(self):
        """
        Train the node-level GNN classifier (GNN2) on nodes within subgraphs.

        Split-aware supervision to avoid leakage in 'coloring'+'all_nodes':
        - Message passing still uses all neighbors (transductive).
        - The loss is computed only on nodes that belong to train_mask.
        - In 'anchor' mode, we supervise only the anchor (no leakage by design).

        Binary one-vs-rest setup (always):
        * node_labeling_mode="anchor"    → supervise the anchor only (BATCHED)
        * node_labeling_mode="all_nodes" → supervise every node in the subgraph

        Flags in params["baseline"]:
        - node_labeling_mode: "anchor" | "all_nodes"
        - use_only_positive_subgraphs_for_node_train: str
            If True, use only subgraphs with sg.y == 1 for node-level training.
            (sg.y is set by `generate_binary_dataset_for_class` according to `mode=anchor|any_node`)
            If False (default), use all subgraphs (recommended baseline).

        Requirements:
        - self.train_loader exists (built from generate_binary_dataset_for_class)
        - Each subgraph carries:
            * node_labels (multiclass 0..C-1)
            * ego_center_local (tensor long; local anchor index)
            * global_node_ids (optional for training; needed for eval)
        - node_model returns per-node LOGITS (no sigmoid inside the model).
        """
        print(f"[!] Starting node-level training... | {self._device}")

        # --- Config ---
        params = self.prediction_params
        node_params = self.prediction_params["node_classifier"]
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        # use_only_pos = params.get("all_or_pos_subg_node_training", "all")
        batch_size = node_params.get("batch_size", 16)

        # --- Load train subgraphs from the pre-built train loader ---
        if self.train_loader is None:
            raise ValueError(
                "Train subgraph DataLoader (self.train_loader) not initialized."
            )
        subgraphs = list(self.train_loader.dataset)
        print(f"[INFO] Loaded all {len(subgraphs)} train subgraphs")

        # --- DataLoader (batched for both modes) ---
        loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)

        # --- Loss (BCEWithLogits) and class-imbalance handling ---
        # Recompute pos_weight on the ACTUAL training pool (after any filtering).
        # In 'anchor' mode, positives = #anchors in target_class across the pool.
        # In 'all_nodes' mode, positives = #nodes with label==target_class across the pool.
        pos_weight = self._compute_pos_weight_for_node_loss(
            subgraphs, self.target_class, node_labeling_mode
        )

        # IMPORTANT:
        # For coloring+all_nodes with split-aware masking we need reduction='none'
        # so that we can zero-out loss for nodes outside train_mask, and then average.
        # For anchor mode, standard reduction='mean' is fine, but 'none' also works.
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self._device) if pos_weight is not None else None,
            reduction="none",  # we will handle the reduction manually
        )
        if pos_weight is not None:
            print(
                f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight={pos_weight.item():.3f}"
            )
        else:
            print(f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight=None")

        # --- Optimizer ---
        optimizer = torch.optim.Adam(
            self.node_model.parameters(),
            lr=node_params["lr"],
            weight_decay=node_params["weight_decay"],
        )

        loss_history = []

        # --- Training loop ---
        for epoch in range(1, node_params["epochs"] + 1):
            self.node_model.train()
            total_loss, correct, total = 0.0, 0, 0

            train_mask_global = self.data.train_mask.to(self._device)
            supervised_train_gids = set()

            for batch in loader:
                batch = batch.to(self._device)

                # Per-node logits on the concatenated batch graph. Shape → [sum_nodes_in_batch]
                logits = self.node_model(batch.x, batch.edge_index).view(-1)

                if node_labeling_mode == "anchor":
                    # --- Supervise ONLY the anchor of each graph in the batch (batched) ---
                    centers_global = self._gather_anchor_indices_in_batch(
                        batch
                    )  # [B_graphs]
                    logit_center = logits[centers_global]  # [B_graphs]

                    # Binary ground-truth for anchors (1 if anchor == target_class else 0)
                    y_center = (
                        batch.node_labels[centers_global] == self.target_class
                    ).float()

                    # loss_fn with reduction='none' → shape [B_graphs]; then mean
                    loss_vec = loss_fn(logit_center, y_center)
                    loss = loss_vec.mean()

                    # Metrics on anchors
                    with torch.no_grad():
                        preds = (torch.sigmoid(logit_center) >= 0.5).long()
                        correct += (preds == y_center.long()).sum().item()
                        total += y_center.numel()

                else:
                    # all_nodes supervision (COLORING):
                    # 1) Binarize labels vs target_class for ALL nodes in the batch.
                    y = (
                        (batch.node_labels == self.target_class)
                        .float()
                        .to(self._device)
                    )  # [sum_nodes]

                    # 2) Build a TRAIN mask aligned with the concatenated batch:
                    #    Map global ids back to the full-graph train_mask → avoids leakage.
                    if not hasattr(batch, "global_node_ids"):
                        raise RuntimeError(
                            "Subgraphs must carry 'global_node_ids' for split-aware masking."
                        )
                    train_mask_sub = train_mask_global[
                        batch.global_node_ids
                    ]  # [sum_nodes] bool

                    # 3) Per-node loss without reduction, then mask & average over train nodes only
                    loss_per_node = loss_fn(logits, y)  # [sum_nodes]
                    # sum over supervised nodes / count of supervised nodes
                    denom = max(1, int(train_mask_sub.sum().item()))
                    loss = (loss_per_node * train_mask_sub.float()).sum() / denom

                    gids = batch.global_node_ids.long().cpu().numpy()
                    mask = train_mask_sub.view(-1).cpu().numpy().astype(bool)
                    supervised_train_gids.update(gids[mask].tolist())

                    # Metrics only on supervised (train) nodes
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) >= 0.5).long()
                        correct += ((preds == y.long()) & train_mask_sub).sum().item()
                        total += train_mask_sub.sum().item()

                # --- Backprop & step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(loader))
            acc = correct / max(1, total)
            loss_history.append(avg_loss)

            if epoch % 20 == 0 or epoch == node_params["epochs"]:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

        # --- Plot training curve ---
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def _assert_centers_belong_to_split(self, loader, split: str):
        split_mask = {
            "train": self.data.train_mask,
            "val": self.data.val_mask,
            "test": self.data.test_mask,
        }[split].to(self._device)
        bad = total = 0
        for batch in loader:
            batch = batch.to(self._device)
            if not hasattr(batch, "ego_center_local"):
                continue  # se não carrega centros, pula
            # recupera índices locais dos centros por gráfico no batch
            if hasattr(batch, "ptr") and batch.ptr is not None:
                ptr = batch.ptr.long()
            else:
                num_nodes_per_graph = torch.bincount(batch.batch)
                ptr = torch.cat(
                    [num_nodes_per_graph.new_zeros(1), num_nodes_per_graph.cumsum(0)]
                ).long()
            centers_local = batch.ego_center_local.view(-1).long()
            centers_concat = centers_local + ptr[:-1]  # idx no tensor concatenado
            centers_global = batch.global_node_ids[centers_concat]  # ids globais
            total += centers_global.numel()
            bad += int((~split_mask[centers_global]).sum().item())
        print(f"[SPLIT-CHECK] centers in '{split}' | total={total} | offsplit={bad}")
        assert bad == 0, f"Há centros fora da máscara de {split} no loader!"

    def _assert_eval_only_split_nodes(self, node_truth: dict, split: str):
        split_mask = {
            "train": self.data.train_mask,
            "val": self.data.val_mask,
            "test": self.data.test_mask,
        }[split].cpu()
        bad = 0
        for gid in node_truth.keys():
            if not bool(split_mask[gid]):
                bad += 1
        print(
            f"[SPLIT-CHECK] evaluated nodes in '{split}' | total={len(node_truth)} | offsplit={bad}"
        )
        assert bad == 0, f"Foram avaliados nós fora do split '{split}'."

    @torch.no_grad()
    def _report_test_node_appearances(self):
        from collections import Counter

        if self.test_loader is None:
            print("[WARN] test_loader missing")
            return
        test_mask = self.data.test_mask.to(self._device)
        cnt = Counter()
        for batch in self.test_loader:
            batch = batch.to(self._device)
            gids = batch.global_node_ids
            is_test = test_mask[gids]
            for gid in gids[is_test].tolist():
                cnt[gid] += 1
        if not cnt:
            print("[INFO] No test nodes found in test_loader.")
            return
        vals = torch.tensor(list(cnt.values()), dtype=torch.float)
        print(
            f"[COVERAGE] TEST nodes covered: {len(cnt)} "
            f"| mean appearances: {vals.mean():.2f} | p90: {vals.kthvalue(int(0.9*len(vals)))[0].item():.0f} "
            f"| max: {int(vals.max().item())}"
        )

    def evaluate_subgraph_classifier_on_test(self, split: str = "test"):
        """
        Evaluate the subgraph-level classifier (GNN1) on a given split (default 'test').

        This function *does not* run any node-level model. It measures how well GNN1
        distinguishes positive vs. negative subgraphs (binary classification).

        Assumptions:
            - A DataLoader for the chosen split exists (self.test_loader or self.val_loader).
            - Each batch is a PyG Batch with:
                - .y           : [B] 0/1 subgraph labels
                - .edge_index  : [2, E]
                - .batch       : [N] node->graph mapping
                - feature attr : one of (batch.<x_key>, batch.x)
            - self.subgraph_model implements `forward_from_data(batch)` and has `x_key`.

        Metrics returned:
            - pos/neg counts
            - accuracy, precision, recall, F1, MCC, balanced accuracy
            - ROC-AUC, PR-AUC (if probabilities for both classes exist)
            - confusion matrix (tn, fp, fn, tp)

        Args:
            split: 'test' (default) or 'val'

        Returns:
            dict with rounded metrics and counts.
        """
        assert split in {"test", "val"}, "split must be 'test' or 'val'"

        # ---- 0) Resolve loader & threshold ----
        if split == "test":
            loader = getattr(self, "test_loader", None)
        else:
            loader = getattr(self, "val_loader", None)

        if loader is None:
            raise RuntimeError(
                f"{split}_loader not initialized. Prepare dataset first."
            )

        params = self.prediction_params.get("subgraph_classifier", {})
        threshold = float(params.get("prediction_threshold", 0.5))

        # Make sure the model is in eval mode and has the right feature key
        self.subgraph_model.eval()
        x_key = self.prediction_params.get("gnn1_x_key", "x")
        if hasattr(self.subgraph_model, "x_key"):
            self.subgraph_model.x_key = x_key

        # ---- 1) Accumulate probs and labels across the split ----
        all_probs, all_true = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                # Forward → logits per subgraph [B, 1]
                logits = self.subgraph_model.forward_from_data(batch)
                probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy().astype(int)

                all_probs.extend(probs.tolist())
                all_true.extend(y.tolist())

        if len(all_true) == 0:
            print(f"[WARN] No subgraphs found in {split} split.")
            return {
                "split": split,
                "num_pos": 0,
                "num_neg": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "balanced_accuracy": 0.0,
                "roc_auc": None,
                "pr_auc": None,
                "confusion_matrix": (0, 0, 0, 0),
                "threshold_used": threshold,
            }

        all_probs = np.asarray(all_probs, dtype=float)
        all_true = np.asarray(all_true, dtype=int)

        # ---- 2) Hard predictions using the configured threshold ----
        preds = (all_probs >= threshold).astype(int)

        # ---- 3) Basic counts ----
        num_pos = int((all_true == 1).sum())
        num_neg = int((all_true == 0).sum())

        # ---- 4) Core metrics (robust to edge cases) ----
        # Precision/Recall/F1 (binary); zero_division=0 avoids warnings on empty preds
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, preds, average="binary", zero_division=0
        )
        # MCC (handle single-class y_true)
        mcc = (
            matthews_corrcoef(all_true, preds) if len(np.unique(all_true)) > 1 else 0.0
        )
        # Balanced accuracy (macro avg of sensitivity & specificity)
        bal_acc = balanced_accuracy_score(all_true, preds)

        # Accuracy
        acc = float((preds == all_true).mean())

        # Confusion matrix → (tn, fp, fn, tp)
        tn, fp, fn, tp = confusion_matrix(all_true, preds, labels=[0, 1]).ravel()

        # ---- 5) Probability-based metrics (AUCs) ----
        roc_auc = None
        pr_auc = None
        # Only defined if both classes are present
        if len(np.unique(all_true)) == 2:
            try:
                roc_auc = float(roc_auc_score(all_true, all_probs))
            except Exception:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(all_true, all_probs))
            except Exception:
                pr_auc = None

        # ---- 6) Pretty print summary ----
        print(f"\n[📈 Subgraph-level evaluation — {split.upper()} split]")
        print(
            f"Positives: {num_pos} | Negatives: {num_neg} | Threshold: {threshold:.3f}"
        )
        print(f"Accuracy   : {acc * 100:5.2f}%")
        print(f"BalancedAcc: {bal_acc * 100:5.2f}%")
        print(f"Precision  : {precision * 100:5.2f}%")
        print(f"Recall     : {recall * 100:4.2f}%")
        print(f"F1         : {f1 * 100:4.2f}%")
        print(f"MCC        : {mcc:6.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC    : {roc_auc:6.4f}")
        if pr_auc is not None:
            print(f"PR-AUC     : {pr_auc:6.4f}")
        print(f"Confusion  : tn={tn} fp={fp} fn={fn} tp={tp}\n")

        # ---- 7) Return a dict of metrics ----
        return {
            "split": split,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "accuracy": round(acc, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "mcc": round(float(mcc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "roc_auc": None if roc_auc is None else round(roc_auc, 4),
            "pr_auc": None if pr_auc is None else round(pr_auc, 4),
            "confusion_matrix": (int(tn), int(fp), int(fn), int(tp)),
            "threshold_used": round(threshold, 3),
        }

    def evaluate_node_classifier_on_test(self, return_node_probs=False):
        """
        Node-level evaluation on the TEST split for one-vs-rest target_class.

        This version NEVER filters subgraphs. If `use_subgraph_classifier=True`,
        GNN1 is used strictly as CONTEXT (bias) for GNN2:
            - fusion_mode='concat' : broadcast subgraph embeddings to node embeddings (concat)
            - fusion_mode='film'   : FiLM modulation (if implemented)
        If `use_subgraph_classifier=False`, runs vanilla node inference (no bias).

        Pipeline:
        0) Build a DataLoader over TEST subgraphs (no filtering).
        1) (Optional) Build per-node context from GNN1 and normalize/ablate if requested.
        2) Run GNN2 forward pass to obtain per-node logits → sigmoid → probabilities.
        3) Restrict to TEST nodes (split-aware).
        4) Aggregate multiple appearances of the same global node id:
        - average probability across appearances
        - true label is unique (binary target vs rest)
        - final hard prediction via calibrated threshold
        5) Compute node-level binary metrics:
        - Accuracy, Precision, Recall, F1, MCC, Balanced Accuracy
        - Macro F1 (for reporting consistency)
        - ROC-AUC, PR-AUC (se ambas as classes aparecem)
        - Confusion matrix (tn, fp, fn, tp)
        """

        print(f"[!] Starting node-level TEST evaluation... | {self._device}")
        self.node_model.eval()
        self._assert_centers_belong_to_split(self.test_loader, "test")
        self._report_test_node_appearances()

        params = self.prediction_params
        node_params = params["node_classifier"]
        subg_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_method == "anchor" else "all_nodes"

        # --- Use the calibrated decision threshold (fallback to 0.5) ---
        threshold = float(node_params.get("prediction_threshold", 0.5))

        # --- Derive 'use_context' exactly as in training ---
        fusion_mode = getattr(self.node_model, "fusion_mode", "concat")
        use_context = bool(
            params.get("use_subgraph_classifier", False)
        ) and fusion_mode in {"concat", "film"}

        if use_context:
            if not hasattr(self, "subgraph_model") or self.subgraph_model is None:
                print(
                    "[WARN] use_subgraph_classifier=True but subgraph_model is missing; proceeding WITHOUT bias."
                )
                use_context = False
            else:
                self.subgraph_model.eval()
                ctx_dim = getattr(self.node_model, "context_dim", None)
                gnn1_dim = getattr(self.subgraph_model, "graph_emb_dim", None)
                if ctx_dim is not None and gnn1_dim is not None and ctx_dim != gnn1_dim:
                    raise ValueError(
                        f"[ERROR] context_dim ({ctx_dim}) must match subgraph_model.graph_emb_dim ({gnn1_dim}) for '{fusion_mode}'."
                    )

        # --- Safety: need global TEST mask from the full graph ---
        if not hasattr(self, "data") or not hasattr(self.data, "test_mask"):
            raise RuntimeError(
                "self.data.test_mask is required to restrict metrics to TEST nodes."
            )
        test_mask_global = self.data.test_mask.to(self._device)

        # ---- 0) TEST subgraphs ----
        if self.test_loader is None:
            raise RuntimeError("test_loader not initialized. Prepare dataset first.")
        test_subgraphs = list(self.test_loader.dataset)
        node_loader = DataLoader(test_subgraphs, batch_size=16, shuffle=False)

        fusion_info = (
            f" | fusion_mode={fusion_mode} | context_norm={node_params.get('context_norm', 'none')}"
            if use_context
            else ""
        )
        ab_cfg = params.get("context_ablation", "none")
        print(
            f"[INFO] Evaluation config | mode={node_labeling_mode} | use_context={use_context}{fusion_info} | ablation={ab_cfg} | thr={threshold:.3f}]"
        )

        # ---- 1–2) Node-level inference, collecting probabilities per global node ----
        # We aggregate by global node id:
        #   - votes_prob[gid] : list of probabilities (sigmoid) for that node across subgraphs
        #   - node_truth[gid] : single binary ground-truth for that node
        votes_prob = {}  # gid -> list[float] of probabilities
        node_truth = {}  # gid -> int (0/1)

        with torch.no_grad():
            for batch in node_loader:
                batch = batch.to(self._device)

                # Optional context from GNN1 (as during training)
                ctx_nodes = None
                if use_context:
                    Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                        batch
                    )  # [B,d]
                    ctx_nodes = Z_s[batch.batch]  # [N,d]
                    ctx_mode = node_params.get("context_norm", "none")
                    ctx_nodes = self._normalize_context_vectors(
                        ctx_nodes, mode=ctx_mode
                    )
                    ctx_nodes = self._apply_context_ablation(ctx_nodes, ab_cfg)

                # Forward pass → logits per-node → probabilities
                if use_context:
                    logits_nodes = self.node_model(
                        batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                    ).view(-1)
                else:
                    logits_nodes = self.node_model(batch.x, batch.edge_index).view(-1)
                probs_nodes = torch.sigmoid(logits_nodes)  # [N]

                # --- Utilities to index anchors inside the batch (if needed) ---
                if hasattr(batch, "ptr") and batch.ptr is not None:
                    ptr = batch.ptr.long()  # [B+1] prefix sums
                else:
                    num_nodes_per_graph = torch.bincount(batch.batch)
                    ptr = torch.cat(
                        [
                            num_nodes_per_graph.new_zeros(1),
                            num_nodes_per_graph.cumsum(0),
                        ]
                    ).long()

                # Global node ids, labels and TEST mask aligned to concatenated nodes
                gids_concat = batch.global_node_ids.view(-1).cpu().numpy()
                labels_concat = batch.node_labels.view(-1)
                test_mask_sub = test_mask_global[batch.global_node_ids.long()].view(
                    -1
                )  # [N] bool

                if node_labeling_mode == "anchor":
                    # Evaluate ONLY anchors that belong to the TEST split
                    centers_local = batch.ego_center_local.view(-1).long()  # [B_graphs]
                    offsets = ptr[:-1]  # [B_graphs]
                    centers_idx = (
                        centers_local + offsets
                    )  # [B_graphs] indices in concatenated tensors
                    centers_in_test = test_mask_sub[centers_idx]  # [B_graphs] bool
                    sel = torch.where(centers_in_test)[0]
                    if sel.numel() == 0:
                        continue

                    # Collect probabilities for anchors
                    probs_center = probs_nodes[centers_idx[sel]].cpu().numpy()
                    for k in range(sel.numel()):
                        node_idx = int(centers_idx[sel[k]].item())
                        gid = int(gids_concat[node_idx])
                        true_bin = int(
                            labels_concat[node_idx].item() == self.target_class
                        )
                        votes_prob.setdefault(gid, []).append(float(probs_center[k]))
                        node_truth[gid] = true_bin

                else:
                    # Evaluate ALL nodes that are in the TEST split
                    test_idx = np.where(test_mask_sub.cpu().numpy())[0]
                    if test_idx.size == 0:
                        continue

                    probs_all = probs_nodes.cpu().numpy()
                    trues_all = (
                        labels_concat.cpu().numpy() == self.target_class
                    ).astype(int)

                    for j in test_idx:
                        gid = int(gids_concat[j])
                        p = float(probs_all[j])
                        true_bin = int(trues_all[j])
                        votes_prob.setdefault(gid, []).append(p)
                        node_truth[gid] = true_bin

        # ---- 3) Aggregate probabilities per global node → mean prob; then hard decision by threshold ----
        self._assert_eval_only_split_nodes(node_truth, "test")

        if len(node_truth) == 0:
            print("[WARN] No TEST nodes evaluated; returning zeros.")
            return {
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "roc_auc": None,
                "pr_auc": None,
                "confusion_matrix": (0, 0, 0, 0),
                "threshold_used": float(node_params.get("prediction_threshold", 0.5)),
            }

        final_true, final_prob, final_pred = [], [], []
        for gid, true_bin in node_truth.items():
            ps = votes_prob.get(gid, [])
            # If for some reason we got no probs (shouldn't happen), default to 0.0
            p_mean = float(np.mean(ps)) if len(ps) > 0 else 0.0
            y_hat = 1 if p_mean >= threshold else 0
            final_true.append(int(true_bin))
            final_prob.append(p_mean)
            final_pred.append(y_hat)

        final_true = np.asarray(final_true, dtype=int)
        final_prob = np.asarray(final_prob, dtype=float)
        final_pred = np.asarray(final_pred, dtype=int)

        # ---- 4) Metrics ----
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            f1_score,
            matthews_corrcoef,
            balanced_accuracy_score,
            confusion_matrix,
            roc_auc_score,
            average_precision_score,
        )

        acc = accuracy_score(final_true, final_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            final_true, final_pred, average="binary", zero_division=0
        )
        macro_f1 = f1_score(final_true, final_pred, average="macro")
        mcc = (
            matthews_corrcoef(final_true, final_pred)
            if len(np.unique(final_true)) > 1
            else 0.0
        )
        bal_acc = balanced_accuracy_score(final_true, final_pred)
        tn, fp, fn, tp = confusion_matrix(final_true, final_pred, labels=[0, 1]).ravel()

        roc_auc = None
        pr_auc = None
        if len(np.unique(final_true)) == 2:
            try:
                roc_auc = float(roc_auc_score(final_true, final_prob))
            except Exception:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(final_true, final_prob))
            except Exception:
                pr_auc = None

        print(f"\n[📊 Evaluation on {len(test_subgraphs)} subgraphs — Node Level]")
        print(f"Threshold     : {threshold:.3f}")
        print(f"Accuracy      : {acc * 100:.2f}%")
        print(f"Balanced Acc. : {bal_acc * 100:.2f}%")
        print(f"Precision     : {precision * 100:.2f}%")
        print(f"Recall        : {recall * 100:.2f}%")
        print(f"F1 (binary)   : {f1 * 100:.2f}%")
        print(f"Macro F1      : {macro_f1 * 100:.2f}%")
        print(f"MCC           : {mcc:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC       : {roc_auc:.4f}")
        if pr_auc is not None:
            print(f"PR-AUC        : {pr_auc:.4f}")
        print(f"Confusion     : tn={tn} fp={fp} fn={fn} tp={tp}\n")

        if return_node_probs:
            # retorna também os dicionários crús para uso multiclasse
            return {
                "metrics": {  # suas métricas atuais
                    "accuracy": round(float(acc), 4),
                    "balanced_accuracy": round(float(bal_acc), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "macro_f1": round(float(macro_f1), 4),
                    "mcc": round(float(mcc), 4),
                    "roc_auc": None if roc_auc is None else round(float(roc_auc), 4),
                    "pr_auc": None if pr_auc is None else round(float(pr_auc), 4),
                },
                "node_truth_binary": node_truth,  # dict[int gid] -> 0/1 para ESTA classe
                "node_prob_binary": {
                    gid: float(np.mean(ps)) for gid, ps in votes_prob.items()
                },  # gid -> p(class)
            }

        return {
            # todos em fração (0–1), não em %
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),  # F1 binário
            "macro_f1": round(float(macro_f1), 4),
            "mcc": round(float(mcc), 4),
            "roc_auc": None if roc_auc is None else round(float(roc_auc), 4),
            "pr_auc": None if pr_auc is None else round(float(pr_auc), 4),
        }

        print(f"\n[📊 Evaluation on {len(test_subgraphs)} subgraphs — Node Level]")
        print(f"Threshold   : {threshold:.3f}")
        print(f"Accuracy    : {acc * 100:6.2f}%")
        print(f"BalancedAcc : {bal_acc * 100:6.2f}%")
        print(f"Precision   : {precision * 100:6.2f}%")
        print(f"Recall      : {recall * 100:6.2f}%")
        print(f"F1          : {f1 * 100:6.2f}%")
        print(f"Macro F1    : {macro_f1 * 100:6.2f}%")
        print(f"MCC         : {mcc:8.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC     : {roc_auc:8.4f}")
        if pr_auc is not None:
            print(f"PR-AUC      : {pr_auc:8.4f}")
        print(f"Confusion   : tn={tn} fp={fp} fn={fn} tp={tp}\n")

        return {
            "balanced_accuracy": round(bal_acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            # Extras to igualar com GNN1:
            "accuracy": round(acc, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "mcc": round(float(mcc), 4),
            "roc_auc": None if roc_auc is None else round(roc_auc, 4),
            "pr_auc": None if pr_auc is None else round(pr_auc, 4),
            "confusion_matrix": (int(tn), int(fp), int(fn), int(tp)),
            "threshold_used": round(threshold, 3),
        }

    def evaluate_node_classifier_on_test_filter(self, ovr_evaluation: bool = True):
        """
        Node-level evaluation on the TEST split (one-vs-rest for self.target_class).

        New: supports FILTERING + CONTEXT
        --------------------------------
        - Filtering (optional, params['use_subgraph_filtering']=True):
            • Run GNN1 over ALL test subgraphs to obtain per-subgraph probabilities.
            • Keep ONLY subgraphs predicted as positive w.r.t. the calibrated threshold
            (params['subgraph_classifier']['prediction_threshold']; fallback to 0.5).
            • If no subgraph passes the threshold, fallback to using ALL subgraphs (warn).

        - Context (optional, same as before):
            • If params['use_subgraph_classifier'] is True and node_model.fusion_mode in {'concat','film'},
            broadcast the subgraph embedding from GNN1 to nodes and feed it to GNN2.

        Pipeline (this function):
        0) Resolve flags/thresholds and build the base test loader.
        1) (Optional) Filtering: score test subgraphs with GNN1 and keep only positives.
        2) (Optional) Build per-node context from GNN1 for the remaining subgraphs.
        3) Run GNN2 → logits → sigmoid(probs) for nodes in TEST mask.
        4) Aggregate multiple occurrences of the same global node id (mean prob).
        5) Hard decision by the calibrated node threshold and compute metrics.

        Returns:
            dict with:
                accuracy, balanced_accuracy, precision, recall, f1 (binary), macro_f1,
                mcc, roc_auc (if defined), pr_auc (if defined),
                plus some bookkeeping: 'threshold_used' (node), 'g1_threshold_used',
                'num_test_subgraphs', 'num_kept_subgraphs'.
        """

        print(f"[!] Starting node-level TEST evaluation... | {self._device}")
        self.node_model.eval()

        # ------------------------- Resolve config & thresholds -------------------------
        params = self.prediction_params
        node_params = params["node_classifier"]
        subg_params = params.get("subgraph_classifier", {})

        subg_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_method == "anchor" else "all_nodes"

        use_filtering = bool(params.get("use_subgraph_filtering", False))  # NEW FLAG
        g1_threshold = float(
            subg_params.get("prediction_threshold", 0.5)
        )  # threshold for GNN1 filtering
        node_threshold = float(
            node_params.get("prediction_threshold", 0.5)
        )  # decision boundary for GNN2

        # Context usage (unchanged logic)
        fusion_mode = getattr(self.node_model, "fusion_mode", "concat")
        use_context = bool(
            params.get("use_subgraph_classifier", False)
        ) and fusion_mode in {"concat", "film"}

        # Sanity for context/filtering dependencies
        if (use_filtering or use_context) and (
            not hasattr(self, "subgraph_model") or self.subgraph_model is None
        ):
            print(
                "[WARN] Filtering/Context requested but subgraph_model is missing; disabling both."
            )
            use_filtering = False
            use_context = False

        if use_context:
            # Ensure dimensions match when using context
            self.subgraph_model.eval()
            ctx_dim = getattr(self.node_model, "context_dim", None)
            gnn1_dim = getattr(self.subgraph_model, "graph_emb_dim", None)
            if ctx_dim is not None and gnn1_dim is not None and ctx_dim != gnn1_dim:
                raise ValueError(
                    f"[ERROR] context_dim ({ctx_dim}) must match subgraph_model.graph_emb_dim ({gnn1_dim}) for '{fusion_mode}'."
                )

        # Need TEST mask from the full graph
        if not hasattr(self, "data") or not hasattr(self.data, "test_mask"):
            raise RuntimeError(
                "self.data.test_mask is required to restrict metrics to TEST nodes."
            )
        test_mask_global = self.data.test_mask.to(self._device)

        # Base test subgraphs
        if self.test_loader is None:
            raise RuntimeError("test_loader not initialized. Prepare dataset first.")
        test_subgraphs_all = list(self.test_loader.dataset)
        num_test_sg = len(test_subgraphs_all)

        # ------------------------- (Optional) FILTERING with GNN1 -------------------------
        # If enabled, score each subgraph with GNN1 and keep only those with prob >= g1_threshold.
        if use_filtering:
            self.subgraph_model.eval()
            x_key = params.get("gnn1_x_key", "x")
            if hasattr(self.subgraph_model, "x_key"):
                self.subgraph_model.x_key = x_key

            kept = []
            # Small, deterministic loader for scoring subgraphs (no shuffle)
            filter_loader = DataLoader(test_subgraphs_all, batch_size=32, shuffle=False)
            cursor = (
                0  # how many subgraphs we've consumed so far from test_subgraphs_all
            )

            with torch.no_grad():
                for batch in filter_loader:
                    batch = batch.to(self._device)
                    logits = self.subgraph_model.forward_from_data(batch)  # [B,1]
                    probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()

                    # Split the batch back into single graphs to filter by prob
                    # We map each prob to the corresponding item from the original list
                    # by iterating in order (DataLoader preserves order when shuffle=False).
                    # Collect indices for this minibatch window:
                    # (We rely on the fact that DataLoader yields samples in the same order)

                    B = len(probs)
                    for j in range(B):
                        idx_global = cursor + j
                        # safety check (shouldn't happen with shuffle=False)
                        if idx_global >= len(test_subgraphs_all):
                            break
                        if float(probs[j]) >= g1_threshold:
                            kept.append(test_subgraphs_all[idx_global])

            if len(kept) == 0:
                print(
                    "[WARN] Filtering kept 0 subgraphs. Falling back to ALL test subgraphs."
                )
                filtered_subgraphs = test_subgraphs_all
            else:
                filtered_subgraphs = kept
            num_kept = len(filtered_subgraphs)
        else:
            filtered_subgraphs = test_subgraphs_all
            num_kept = len(filtered_subgraphs)

        # Final test loader (possibly filtered)
        print(
            f"[INFO] {len(filtered_subgraphs)}/{len(test_subgraphs_all)} subgraphs remained after filtering"
        )
        node_loader = DataLoader(filtered_subgraphs, batch_size=16, shuffle=False)

        fusion_info = (
            f" | fusion_mode={fusion_mode} | context_norm={node_params.get('context_norm', 'none')}"
            if use_context
            else ""
        )
        ab_cfg = params.get("context_ablation", "none")
        print(
            f"[INFO] Eval config | mode={node_labeling_mode} | context={use_context}{fusion_info} "
            f"| filtering={use_filtering} (g1_thr={g1_threshold:.3f}) | node_thr={node_threshold:.3f} "
            f"| kept={num_kept}/{num_test_sg} | ablation={ab_cfg}"
        )

        # ------------------------- Inference with GNN2 (on remaining subgraphs) -------------------------
        # Aggregate per-node probabilities by global node id across multiple subgraph appearances.
        votes_prob = {}  # gid -> list[float]
        node_truth = {}  # gid -> int (0/1)

        with torch.no_grad():
            for batch in node_loader:
                batch = batch.to(self._device)

                # Optional context from GNN1 (same normalization/ablation as training)
                ctx_nodes = None
                if use_context:
                    Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                        batch
                    )  # [B, d]
                    ctx_nodes = Z_s[batch.batch]  # broadcast to nodes [N, d]
                    ctx_mode = node_params.get("context_norm", "none")
                    ctx_nodes = self._normalize_context_vectors(
                        ctx_nodes, mode=ctx_mode
                    )
                    ctx_nodes = self._apply_context_ablation(ctx_nodes, ab_cfg)

                # Forward GNN2 → node logits → probabilities
                if use_context:
                    logits_nodes = self.node_model(
                        batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                    ).view(-1)
                else:
                    logits_nodes = self.node_model(batch.x, batch.edge_index).view(-1)
                probs_nodes = torch.sigmoid(logits_nodes)  # [N]

                # Prepare ptr to recover anchor positions if needed
                if hasattr(batch, "ptr") and batch.ptr is not None:
                    ptr = batch.ptr.long()  # [B+1]
                else:
                    num_nodes_per_graph = torch.bincount(batch.batch)
                    ptr = torch.cat(
                        [
                            num_nodes_per_graph.new_zeros(1),
                            num_nodes_per_graph.cumsum(0),
                        ]
                    ).long()

                # Align global ids, labels and TEST mask to concatenated nodes
                gids_concat = batch.global_node_ids.view(-1).cpu().numpy()
                labels_concat = batch.node_labels.view(-1)
                test_mask_sub = test_mask_global[batch.global_node_ids.long()].view(
                    -1
                )  # [N] bool

                if node_labeling_mode == "anchor":
                    # Evaluate ONLY anchors that are in TEST split
                    centers_local = batch.ego_center_local.view(-1).long()  # [B]
                    offsets = ptr[:-1]  # [B]
                    centers_idx = (
                        centers_local + offsets
                    )  # [B] positions in concatenated tensors
                    centers_in_test = test_mask_sub[centers_idx]  # [B] bool
                    sel = torch.where(centers_in_test)[0]
                    if sel.numel() == 0:
                        continue

                    probs_center = probs_nodes[centers_idx[sel]].cpu().numpy()
                    for k in range(sel.numel()):
                        node_idx = int(centers_idx[sel[k]].item())
                        gid = int(gids_concat[node_idx])
                        true_bin = int(
                            labels_concat[node_idx].item() == self.target_class
                        )
                        votes_prob.setdefault(gid, []).append(float(probs_center[k]))
                        node_truth[gid] = true_bin
                else:
                    # Evaluate ALL nodes that are in TEST split
                    test_idx = np.where(test_mask_sub.cpu().numpy())[0]
                    if test_idx.size == 0:
                        continue

                    probs_all = probs_nodes.cpu().numpy()
                    trues_all = (
                        labels_concat.cpu().numpy() == self.target_class
                    ).astype(int)

                    for j in test_idx:
                        gid = int(gids_concat[j])
                        p = float(probs_all[j])
                        true_bin = int(trues_all[j])
                        votes_prob.setdefault(gid, []).append(p)
                        node_truth[gid] = true_bin

        # ------------------------- Aggregate per-node and compute metrics -------------------------
        if len(node_truth) == 0:
            print("[WARN] No TEST nodes evaluated; returning zeros.")
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "macro_f1": 0.0,
                "mcc": 0.0,
                "roc_auc": None,
                "pr_auc": None,
                "threshold_used": node_threshold,
                "g1_threshold_used": g1_threshold if use_filtering else None,
                "num_test_subgraphs": num_test_sg,
                "num_kept_subgraphs": num_kept,
            }

        final_true, final_prob, final_pred = [], [], []
        for gid, true_bin in node_truth.items():
            ps = votes_prob.get(gid, [])
            p_mean = float(np.mean(ps)) if len(ps) > 0 else 0.0
            y_hat = 1 if p_mean >= node_threshold else 0
            final_true.append(int(true_bin))
            final_prob.append(p_mean)
            final_pred.append(y_hat)

        final_true = np.asarray(final_true, dtype=int)
        final_prob = np.asarray(final_prob, dtype=float)
        final_pred = np.asarray(final_pred, dtype=int)

        # Metrics (binary)
        acc = accuracy_score(final_true, final_pred)
        precision, recall, _, _ = precision_recall_fscore_support(
            final_true, final_pred, average="binary", zero_division=0
        )
        f1 = f1_score(final_true, final_pred)
        macro_f1 = f1_score(final_true, final_pred, average="macro")
        mcc = (
            matthews_corrcoef(final_true, final_pred)
            if len(np.unique(final_true)) > 1
            else 0.0
        )
        bal_acc = balanced_accuracy_score(final_true, final_pred)
        tn, fp, fn, tp = confusion_matrix(final_true, final_pred, labels=[0, 1]).ravel()

        roc_auc = None
        pr_auc = None
        if len(np.unique(final_true)) == 2:
            try:
                roc_auc = float(roc_auc_score(final_true, final_prob))
            except Exception:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(final_true, final_prob))
            except Exception:
                pr_auc = None

        # Pretty print
        print(f"\n[📊 Evaluation on {num_kept} subgraphs — Node Level]")
        if use_filtering:
            print(
                f"(Filtering active | GNN1 thr={g1_threshold:.3f} | kept {num_kept}/{num_test_sg})"
            )
        print(f"Node threshold : {node_threshold:.3f}")
        print(f"Accuracy       : {acc * 100:.2f}%")
        print(f"Balanced Acc.  : {bal_acc * 100:.2f}%")
        print(f"Precision      : {precision * 100:.2f}%")
        print(f"Recall         : {recall * 100:.2f}%")
        print(f"F1 (binary)    : {f1 * 100:.2f}%")
        print(f"Macro F1       : {macro_f1 * 100:.2f}%")
        print(f"MCC            : {mcc:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC        : {roc_auc:.4f}")
        if pr_auc is not None:
            print(f"PR-AUC         : {pr_auc:.4f}")
        print(f"Confusion      : tn={tn} fp={fp} fn={fn} tp={tp}\n")

        if ovr_evaluation:
            # Return everything as fractions (0–1)
            return {
                "accuracy": round(float(acc), 4),
                "balanced_accuracy": round(float(bal_acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "macro_f1": round(float(macro_f1), 4),
                "mcc": round(float(mcc), 4),
                "roc_auc": None if roc_auc is None else round(float(roc_auc), 4),
                "pr_auc": None if pr_auc is None else round(float(pr_auc), 4),
                "threshold_used": node_threshold,
                "g1_threshold_used": g1_threshold if use_filtering else None,
                "num_test_subgraphs": num_test_sg,
                "num_kept_subgraphs": num_kept,
            }

        return {
            "metrics": {  # suas métricas atuais
                "accuracy": round(float(acc), 4),
                "balanced_accuracy": round(float(bal_acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "macro_f1": round(float(macro_f1), 4),
                "mcc": round(float(mcc), 4),
                "threshold_used": node_threshold,
                "roc_auc": None if roc_auc is None else round(float(roc_auc), 4),
                "pr_auc": None if pr_auc is None else round(float(pr_auc), 4),
            },
            "node_truth_binary": node_truth,  # dict[int gid] -> 0/1 para ESTA classe
            "node_prob_binary": {
                gid: float(np.mean(ps)) for gid, ps in votes_prob.items()
            },  # gid -> p(class)
        }

    #############################################################
    # Auxiliary Functions
    #############################################################

    def _build_coloring_subgraphs_from_mask2(
        self,
        data: PyGData,
        center_mask: torch.Tensor,
        num_layers: int,
        shuffle: bool = True,
        limit_centers: Optional[int] = None,
        verbose: bool = True,
        *,
        allowed_nodes_mask: Optional[torch.Tensor] = None,
    ) -> List[PyGData]:
        """
        Build k-hop ego-subgraphs using a coloring/coverage policy.

        Coloring policy (coverage):
        - Candidate centers = nodes with center_mask == True.
        - Iterate candidates (optionally shuffled). For each center c:
            * If c is already 'colored', skip (it was covered before).
            * Extract the k-hop ego-subgraph around c.
            * (Optional) If `allowed_nodes_mask` is provided, drop any nodes that are not allowed.
            * Append the subgraph.
            * Mark every node in THIS subgraph as 'colored' (ineligible as future centers).
        - Stop when all eligible centers are covered or `limit_centers` is reached.

        Inductive train-mode switch:
        - Pass `allowed_nodes_mask=data.train_mask` for TRAIN subgraphs
            to ensure neighbors/edges are restricted to the training split.
        - For VAL/TEST, call without `allowed_nodes_mask` to keep natural connectivity
            (neighbors can include train + current split).

        Args:
        data: Full-graph PyG Data.
        center_mask: Boolean [N], True = eligible center.
        num_layers: k-hop radius for ego-subgraphs.
        shuffle: Shuffle candidate centers before coloring.
        limit_centers: Cap number of produced subgraphs (after coloring).
        verbose: Print a summary line.
        allowed_nodes_mask: Optional boolean [N]. If provided, only nodes with True
                            are kept in each subgraph (center MUST satisfy mask).

        Returns:
        List[PyGData], one subgraph per selected center, carrying:
            - x, edge_index
            - node_labels
            - ego_center_local (Long scalar)
            - ego_center_global (int)
            - global_node_ids (Tensor[|V_sub|])
        """
        assert center_mask.dtype == torch.bool and center_mask.numel() == data.num_nodes
        if allowed_nodes_mask is not None:
            assert (
                allowed_nodes_mask.dtype == torch.bool
                and allowed_nodes_mask.numel() == data.num_nodes
            )

        # Prepare candidate centers (only those marked by center_mask)
        centers = torch.nonzero(center_mask, as_tuple=True)[0].tolist()
        if shuffle:
            random.shuffle(centers)

        subgraphs: List[PyGData] = []
        colored: set[int] = set()  # nodes already covered by previous subgraphs
        n = data.num_nodes

        for c in centers:
            # Skip if this center is already covered by a previous subgraph
            if c in colored:
                continue

            # If we're restricting nodes (e.g., train-only), centers must also be allowed
            if allowed_nodes_mask is not None and not bool(allowed_nodes_mask[c]):
                continue

            # Extract k-hop ego-subgraph (local relabeling to [0..m-1])
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                c,
                num_layers,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=n,
            )

            # Optional restriction: keep only allowed nodes within the subgraph
            if allowed_nodes_mask is not None:
                keep_local = allowed_nodes_mask[subset]  # [|V_sub|] bool over local IDs
                if not keep_local.any():
                    # No allowed nodes (shouldn't happen because center is allowed)
                    continue

                # Map from old local IDs to new compact local IDs
                # Build a local reindexing for kept nodes
                kept_idx_local = torch.nonzero(keep_local, as_tuple=True)[0]
                old_to_new = -torch.ones(subset.size(0), dtype=torch.long)
                old_to_new[kept_idx_local] = torch.arange(
                    kept_idx_local.numel(), dtype=torch.long
                )

                # Filter edges whose endpoints remain after restriction
                u, v = sub_edge_index
                keep_e = keep_local[u] & keep_local[v]
                sub_edge_index = sub_edge_index[:, keep_e]
                # Reindex edge endpoints to the compacted local space
                sub_edge_index = torch.stack(
                    [old_to_new[u[keep_e]], old_to_new[v[keep_e]]], dim=0
                )

                # Shrink subset and features/labels accordingly
                subset = subset[kept_idx_local]
                # Remap center's local index
                ego_center_local_old = int(mapping[0].item())
                # If the original center got dropped (should not happen), skip
                if old_to_new[ego_center_local_old] < 0:
                    # Safety guard; practically unreachable because center is allowed
                    continue
                ego_center_local = int(old_to_new[ego_center_local_old].item())
            else:
                ego_center_local = int(mapping[0].item())

            # Skip degenerate subgraphs: less than 2 nodes or no edges
            if subset.numel() < 2 or sub_edge_index.numel() == 0:
                continue

            # Build PyG subgraph object
            sg = PyGData(
                x=data.x[subset],
                edge_index=sub_edge_index,
                node_labels=data.y[subset],
                ego_center_local=torch.tensor(ego_center_local, dtype=torch.long),
                ego_center_global=int(c),
                global_node_ids=subset,  # map back to global graph
            )
            subgraphs.append(sg)

            # Color ONLY nodes that survived in this subgraph (respecting allowed_nodes_mask if used)
            colored.update(subset.tolist())

            # Optional cap on produced subgraphs
            if limit_centers is not None and len(subgraphs) >= int(limit_centers):
                break

        if verbose:
            print(
                f"[✓] Built {len(subgraphs)} k-hop coloring subgraphs "
                f"(k={num_layers}) from {center_mask.sum().item()} eligible centers | "
                f"centers used: {len(subgraphs)} | covered nodes: {len(colored)} "
                f"| restricted={allowed_nodes_mask is not None}"
            )

        return subgraphs

    def _build_coloring_subgraphs_from_mask(
        self,
        data: PyGData,
        center_mask: torch.Tensor,
        num_layers: int,
        shuffle: bool = True,
        limit_centers: Optional[int] = None,
        verbose: bool = True,
    ) -> List[PyGData]:
        """
        Build k-hop ego-subgraphs on the *full* graph using a coloring/coverage policy:

        Policy:
        - Candidate centers are nodes with `center_mask == True`.
        - Iterate candidates (optionally shuffled). For each candidate c:
            * If c is already 'colored' (covered by a previous subgraph), skip.
            * Extract the k-hop ego-subgraph around c (using ALL nodes/edges reached).
            * Append the subgraph.
            * 'Color' (cover) EVERY node in this subgraph so it can no longer be a center later.
        - Neighbors may be colored; only the *center* is constrained to be uncolored.
        - Stop when all eligible centers are covered or `limit_centers` is reached.

        Notes:
        - We DO NOT filter neighbors by any split mask: all nodes/edges reached within k hops are included.
        - We DO keep edgeless subgraphs (rare; e.g., isolated nodes under small k); you may skip them if desired.

        Args:
            data: Full PyG Data (entire graph).
            center_mask: Boolean [num_nodes], True = eligible center.
            num_layers: Number of hops (k) for ego-subgraph.
            shuffle: Shuffle candidate centers before coloring.
            limit_centers: Cap the number of produced subgraphs (after coloring).
            verbose: Print a summary.

        Returns:
            List[PyGData]: One subgraph per *selected* center. Each subgraph contains:
                - x, edge_index
                - node_labels  (multiclass 0..C-1 for subgraph nodes)
                - ego_center_local (LongTensor scalar)  index of the center inside the subgraph
                - ego_center_global (int)               center global id
                - global_node_ids (Tensor[|V_sub|])     global ids of subgraph nodes
        """
        assert center_mask.dtype == torch.bool
        assert center_mask.numel() == data.num_nodes

        # Candidate centers = nodes where mask is True
        centers = torch.nonzero(center_mask, as_tuple=True)[0].tolist()
        if shuffle:
            random.shuffle(centers)

        subgraphs: List[PyGData] = []
        colored: set[int] = set()  # set of *center-ineligible* nodes
        n = data.num_nodes

        for c in centers:
            if c in colored:
                continue

            # k-hop ego-subgraph around center 'c', relabeled to local [0..m-1]
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                c,
                num_layers,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=n,
            )

            # Skip subgraphs that are too small (singleton) or edgeless:
            if subset.numel() < 2 or sub_edge_index.numel() == 0:
                continue

            ego_center_local = int(mapping[0].item())

            sg = PyGData(
                x=data.x[subset],
                edge_index=sub_edge_index,
                node_labels=data.y[subset],
                ego_center_local=torch.tensor(ego_center_local, dtype=torch.long),
                ego_center_global=int(c),
                global_node_ids=subset,
            )
            subgraphs.append(sg)

            # Color (cover) every node in this subgraph so they can no longer be centers
            colored.update(subset.tolist())

            # Optional cap on the number of produced subgraphs
            if limit_centers is not None and len(subgraphs) >= int(limit_centers):
                break

        if verbose:
            print(
                f"[✓] Built {len(subgraphs)} k-hop coloring subgraphs "
                f"(k={num_layers}) from {center_mask.sum().item()} eligible centers | "
                f"centers used: {len(subgraphs)} | covered nodes: {len(colored)}"
            )

        return subgraphs

    def _build_anchor_subraphs_from_mask(
        self,
        data: PyGData,
        center_mask: torch.Tensor,
        num_layers: int,
        shuffle: bool = True,
        limit_centers: Optional[int] = None,
        verbose: bool = True,
    ) -> List[PyGData]:
        """
        Build k-hop ego-subgraphs on the *full* graph (no partitioning).
        Each subgraph is centered at one node selected by `center_mask`.

        - We DO NOT filter neighbors by any split mask: all nodes/edges
        reached within k hops are included (train/val/test/unlabeled).
        - We DO NOT apply coloring/coverage: each True in `center_mask`
        becomes a subgraph center (optionally shuffled/limited).

        Args:
            data: Full PyG Data object (entire Cora graph).
            center_mask: Boolean tensor of shape [num_nodes], True = eligible center.
            num_layers: Number of hops (k) for the ego-subgraph.
            shuffle: Whether to randomize the order of centers.
            limit_centers: If set, cap the number of centers used (after shuffling).
            verbose: Print a short summary at the end.

        Returns:
            List[PyGData]: One subgraph per center. Each subgraph contains:
                - x: node features of the subgraph
                - edge_index: relabeled edges within the subgraph
                - node_labels: original multi-class labels (0..C-1) for subgraph nodes
                - ego_center_local: index (0..|V_sub|-1) of the center inside the subgraph
                - ego_center_global: original global node id of the center
                - global_node_ids: original global node ids for all nodes in the subgraph

            Training tip:
            - Compute supervised loss only on the center node:
            logits_center = model(subgraph).logits[ subgraph.ego_center_local ]
            loss += criterion(logits_center, label_of_center)
        """
        assert center_mask.dtype == torch.bool
        assert center_mask.numel() == data.num_nodes

        # Candidate centers are global node indices where mask == True
        centers = torch.nonzero(center_mask, as_tuple=True)[0].tolist()
        if shuffle:
            random.shuffle(centers)
        if limit_centers is not None:
            centers = centers[: int(limit_centers)]

        subgraphs: List[PyGData] = []
        n = data.num_nodes

        for c in centers:
            # k-hop ego-subgraph around center 'c', relabeled to local [0..m-1]
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                c,
                num_layers,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=n,
            )

            # Skip subgraphs that are too small (singleton) or edgeless:
            if subset.numel() < 2 or sub_edge_index.numel() == 0:
                continue

            # If the ego-subgraph is edgeless (rare; e.g., isolated center under current k),
            # we can still keep it (a single node) or skip it. Here we keep it.
            ego_center_local = int(mapping[0].item())

            sg = PyGData(
                x=data.x[subset],
                edge_index=sub_edge_index,
                node_labels=data.y[subset],  # keep original labels (0..C-1)
                ego_center_local=torch.tensor(
                    ego_center_local, dtype=torch.long
                ),  # local index of the anchor
                ego_center_global=int(c),  # global id of the anchor
                global_node_ids=subset,  # global ids of all nodes in this subgraph
            )
            subgraphs.append(sg)

        if verbose:
            print(
                f"[✓] Built {len(subgraphs)} k-hop subgraphs "
                f"(k={num_layers}) from {center_mask.sum().item()} centers "
                f"{'(limited)' if limit_centers is not None else ''}"
            )

        return subgraphs

    def _apply_long_tailed_train_mask(
        self,
        data,
        ratio: float,  # e.g., 50.0 means imbalance ratio = 50
    ):
        """
        Apply long-tailed distribution logic to the *training* set (multiclass),
        following the NodeImport approach:
        - Count the number of nodes per class within the original train split
        - Call make_longtailed_data_remove to adjust the train distribution
        according to the imbalance ratio (IR)
        - Update data.train_mask with the new mask (validation/test remain untouched)
        - Print class distribution before and after LT for verification
        - Optionally return detailed info (idx_info, masks) for inspection
        """
        n_cls = int(data.y.max().item() + 1)
        device = data.y.device

        # Count samples per class ONLY inside the original train split
        train_idx = torch.nonzero(data.train_mask, as_tuple=True)[0]
        y_train = data.y[train_idx]
        n_data = torch.bincount(
            y_train, minlength=n_cls
        ).tolist()  # list of counts per class

        print("[LT] Original train class counts:", n_data, "| sum:", sum(n_data))

        # Construct long-tailed training set
        class_num_list, new_train_mask, idx_info, node_mask, edge_mask = (
            make_longtailed_data_remove(
                edge_index=data.edge_index.to(device),
                label=data.y.to(device),
                n_data=n_data,
                n_cls=n_cls,
                ratio=ratio,
                train_mask=data.train_mask.to(device),
            )
        )

        # Update the training mask (validation/test remain intact)
        data.train_mask = new_train_mask.cpu()

        # Inspect how many samples per class were kept
        kept_counts = [
            len(idx) for idx in idx_info
        ]  # global indices of training nodes per class
        print("[LT] New train class counts:", kept_counts, "| sum:", sum(kept_counts))

        return {
            "class_num_list": class_num_list,  # desired number of nodes per class under LT
            "kept_idx_per_class": idx_info,  # global indices of training nodes kept per class
            "node_mask": node_mask,  # boolean mask over all nodes (True = kept)
            "edge_mask": edge_mask,  # boolean mask over edges (True = kept)
        }

    def _compute_pos_weight_for_node_loss(
        self, subgraphs, target_class: int, node_labeling_mode: str
    ) -> torch.Tensor:
        """
        Compute a sensible pos_weight for BCEWithLogitsLoss according to the chosen node labeling mode.
        - 'anchor': count positives/negatives at ANCHOR-level (one label per subgraph).
        - 'all_nodes': count positives/negatives at NODE-level (every node in every subgraph).
        Returns a 1D tensor [pos_weight] on self._device.
        """
        if node_labeling_mode == "anchor":
            pos = sum(
                int(sg.node_labels[int(sg.ego_center_local)].item() == target_class)
                for sg in subgraphs
            )
            neg = len(subgraphs) - pos
        else:  # 'all_nodes'
            pos = sum((sg.node_labels == target_class).sum().item() for sg in subgraphs)
            neg = sum((sg.node_labels != target_class).sum().item() for sg in subgraphs)

        pos_weight = torch.tensor(
            [neg / max(pos, 1)], device=self._device, dtype=torch.float
        )
        return pos_weight

    def _gather_anchor_indices_in_batch(self, batch) -> torch.Tensor:
        """
        Map each subgraph's local anchor index to the concatenated node index in the batched tensors.
        Works with PyG Batch objects:
        - Prefer Batch.ptr (PyG >= 2), which stores cumulative node counts per graph.
        - Fallback to computing offsets via batch.batch (graph id per node).
        Returns a LongTensor of shape [num_graphs_in_batch] with global indices.
        """
        centers_local = batch.ego_center_local.view(-1).long()  # one anchor per graph

        if hasattr(batch, "ptr") and batch.ptr is not None:
            # ptr: [B+1], offsets = ptr[:-1]
            offsets = batch.ptr[:-1].long()
        else:
            # Fallback: offsets from bincount on batch.batch
            num_nodes_per_graph = torch.bincount(batch.batch)
            offsets = torch.cat(
                [num_nodes_per_graph.new_zeros(1), num_nodes_per_graph.cumsum(0)[:-1]]
            ).long()

        centers_global = centers_local + offsets
        return centers_global

    def _get_pyg_dataset(self, ds_name):
        """
        Load a PyG dataset (Amazon or Planetoid family) with normalized features.

        Args:
            ds_name (str): Dataset name. Supported values:
                - "Photo" or "Computers": load from Amazon dataset family.
                - Otherwise: load from Planetoid dataset family (e.g., "Cora", "CiteSeer", "PubMed").

        Returns:
            dataset (torch_geometric.data.Dataset): Loaded PyG dataset with normalized features.
        """
        if ds_name == "Photo" or ds_name == "Computers":
            # --- Amazon datasets ---
            # Amazon-Photo and Amazon-Computers come from the "Amazon" benchmark in PyG.
            # They are citation/product co-purchase graphs with "geom-gcn" split.
            print("[!] Preparing dataset with Amazon public split...")
            dataset = Amazon(
                root=f"data/{ds_name}",  # where to store/cache the dataset
                name=ds_name,  # which Amazon dataset to load
                split="geom-gcn",  # standard split used in PyG benchmarks
                transform=NormalizeFeatures(),  # normalize node features (unit variance, zero mean)
            )
        else:
            # --- Planetoid datasets ---
            # Includes Cora, CiteSeer, PubMed (standard citation networks).
            print("[!] Preparing dataset with Planetoid public split...")
            dataset = Planetoid(
                root=f"data/{ds_name}",  # where to store/cache the dataset
                name=ds_name,  # which Planetoid dataset to load
                split="geom-gcn",  # public split from Geom-GCN paper
                transform=NormalizeFeatures(),  # normalize node features
            )

        return dataset

    def _normalize_context_vectors(
        self, ctx_nodes, mode: str = "none", eps: float = 1e-6
    ):
        """
        Normalize per-node context vectors [N, d] using a single selected mode.
        mode: 'none' | 'layernorm' | 'batch_zscore' | 'l2'
        """
        mode = (mode or "none").lower()
        if mode == "none":
            return ctx_nodes
        if mode in ("layernorm", "ln"):
            return F.layer_norm(ctx_nodes, (ctx_nodes.size(1),))
        if mode in ("batch_zscore", "zscore", "standardize"):
            mean = ctx_nodes.mean(dim=0, keepdim=True)
            std = ctx_nodes.std(dim=0, keepdim=True).clamp_min(eps)
            return (ctx_nodes - mean) / std
        if mode in ("l2", "l2norm"):
            return ctx_nodes / (ctx_nodes.norm(dim=1, keepdim=True) + eps)
        raise ValueError(f"Unknown context_norm mode: {mode}")

    def _resolve_x_dim_for_key(self, data_obj, x_key: str) -> int:
        """
        Return the feature dimension for the chosen x_key on a PyG Data object.
        Falls back to `.x` if x_key is missing. Raises if none are present.
        """
        x = getattr(data_obj, x_key, None)
        if x is None:
            x = getattr(data_obj, "x", None)
        if x is None:
            raise AttributeError(f"No features found for '{x_key}' or 'x'.")
        return int(x.size(1))

    def _compute_pos_weight_from_loader(self, loader, device) -> torch.Tensor:
        """
        Compute 'pos_weight' for BCEWithLogitsLoss given a DataLoader of subgraph batches.

        Why:
            For imbalanced binary classification, BCEWithLogitsLoss can upweight positive
            examples by 'pos_weight = #neg / #pos', which balances the loss contributions.

        Args:
            loader: Iterable of PyG Batches with attribute `y` ∈ {0,1} per subgraph.
            device: Torch device for the returned tensor.

        Returns:
            pos_weight: shape [1] tensor to pass into BCEWithLogitsLoss(pos_weight=...).
        """
        pos = 0
        total = 0
        for batch in loader:
            y = batch.y.view(-1).long()
            pos += int((y == 1).sum())
            total += int(y.numel())
        neg = total - pos
        # Avoid division by zero when there are no positives in the split
        pw = neg / max(pos, 1)
        return torch.tensor([pw], device=device, dtype=torch.float32)

    def _make_warmup_cosine_scheduler(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int | None = None,
        min_lr_scale: float = 0.1,
    ):
        """
        Create a LambdaLR scheduler that:
        1) linearly warms up LR to its base value in the first 'warmup_epochs';
        2) applies cosine decay from 1.0 down to 'min_lr_scale' afterwards.

        Notes:
            - This operates on a *scale* applied to each param group's base LR.
            - Keep 'min_lr_scale' > 0 to avoid LR=0 near the end.

        Args:
            optimizer: torch.optim.Optimizer
            total_epochs: total number of epochs you plan to run
            warmup_epochs: number of warmup epochs (default = 5% of total)
            min_lr_scale: final scaling factor (e.g., 0.1 keeps LR at 10% of base)

        Returns:
            torch.optim.lr_scheduler.LambdaLR
        """
        if warmup_epochs is None:
            warmup_epochs = max(5, int(0.05 * total_epochs))

        def lr_lambda(epoch_idx: int) -> float:
            # Epoch indices start at 0 inside schedulers; map to [1..]
            e = epoch_idx + 1
            if e <= warmup_epochs:
                # Linear warmup from ~0 to 1 over 'warmup_epochs'
                return float(e) / float(warmup_epochs)

            # Cosine decay from 1.0 → min_lr_scale across the remaining epochs
            progress = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))  # ∈ [0,1]
            return min_lr_scale + (1.0 - min_lr_scale) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -----------------------------------------------------------------------------
    # Utility: apply threshold and compute quick metrics
    # -----------------------------------------------------------------------------
    def _apply_threshold(self, y_prob: np.ndarray, threshold: float) -> np.ndarray:
        """
        Convert probabilities to hard labels using the given threshold.
        """
        return (np.asarray(y_prob) >= float(threshold)).astype(int)

    # -----------------------------------------------------------------------------
    # Loss factory with pos_weight
    # -----------------------------------------------------------------------------
    def _make_bce_with_logits(
        self, pos_weight: torch.Tensor | None = None
    ) -> nn.Module:
        """
        Create BCEWithLogitsLoss with an optional pos_weight.
        """
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- helper (coloque na sua classe/pipeline) ---
    def _apply_context_ablation(
        self, ctx_nodes: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """
        Ablate the context to test whether GNN2 gains come from capacity or real information.
        mode = "none" | "shuffle" | "gauss" | "zeros"
        """
        if mode == "none":
            return ctx_nodes
        if mode == "shuffle":
            idx = torch.randperm(ctx_nodes.size(0), device=ctx_nodes.device)
            return ctx_nodes[idx]
        if mode == "gauss":
            mu = ctx_nodes.mean(dim=0, keepdim=True)
            std = ctx_nodes.std(dim=0, keepdim=True) + 1e-6
            return torch.randn_like(ctx_nodes) * std + mu
        if mode == "zeros":
            return torch.zeros_like(ctx_nodes)
        return ctx_nodes  # fallback

    def _best_threshold_by_f1(self, y_true, y_prob, beta=1.0, num_thresh=200):
        """
        Select the probability threshold that maximizes F1 (or F-beta) on a validation split.
        Strategy:
        - If the number of unique probabilities is small, try them all.
        - Otherwise, evaluate a linspace grid (e.g., 0.05..0.95) to keep it fast.

        Args:
            y_true: ground-truth labels, shape [N], values in {0,1}
            y_prob: predicted probabilities (sigmoid(logits)), shape [N], values in [0,1]
            beta:  F-beta score (beta=1 -> F1; beta>1 weights recall more, beta<1 weights precision)
            num_thresh: cap on how many thresholds to evaluate (speed/robustness tradeoff)

        Returns:
            (best_threshold, best_f1)
        """
        best_t, best_f = 0.5, -1.0
        thresholds = np.linspace(0.05, 0.95, num_thresh)

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f = fbeta_score(
                y_true, y_pred, beta=beta, average="binary", zero_division=0
            )
            if f > best_f:
                best_f, best_t = f, t

        return best_t, best_f

    def _maybe_override_split(
        self,
        data,
        custom_split: bool,
        *,
        train_frac: float = 0.60,
        val_frac: float = 0.20,
        seed: int = 0,
        stratified: bool = True,
        min_per_class: int = 1,
    ):
        """
        Optionally override dataset masks with a 60/20/20 split.

        - Saves original masks in data._orig_* for safety.
        - Produces 1-D boolean masks: data.train_mask, data.val_mask, data.test_mask
        - Also sets convenience index tensors: data.train_idx, data.val_idx, data.test_idx
        - Stratified per class by default (keeps at least `min_per_class` samples/class in train/val if possível)

        Args:
            data: PyG Data with attributes x, y, (edge_index), and possibly train/val/test_mask.
            custom_split: if False, do nothing.
            train_frac, val_frac: fractions; test = 1 - train - val
            seed: RNG seed for reproducibility
            stratified: split per class if True; else global random split
            min_per_class: enforce at least this many nodes/class in train/val when viável
        """
        if not custom_split:
            return

        assert hasattr(data, "y"), "[split] data.y is required for stratified split."
        num_nodes = data.num_nodes
        y = data.y.view(-1)

        # Save originals once (if present)
        for attr in ("train_mask", "val_mask", "test_mask"):
            if hasattr(data, attr) and not hasattr(data, f"_orig_{attr}"):
                setattr(data, f"_orig_{attr}", getattr(data, attr).clone())

        # Helper RNG
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        # Prepare containers
        train_idx_list, val_idx_list, test_idx_list = [], [], []

        if stratified:
            classes = torch.unique(y).tolist()
            for c in classes:
                cls_idx = (y == c).nonzero(as_tuple=False).view(-1)
                # Shuffle indices of this class
                perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]

                n = perm.numel()
                n_train = max(min_per_class, int(round(train_frac * n)))
                n_val = max(min_per_class, int(round(val_frac * n)))
                # Ensure we don't exceed n; adjust greedily
                if n_train + n_val > n:
                    overflow = n_train + n_val - n
                    # reduce val first, then train if needed
                    reduce_val = min(overflow, n_val - min_per_class)
                    n_val -= reduce_val
                    overflow -= reduce_val
                    if overflow > 0:
                        reduce_train = min(overflow, n_train - min_per_class)
                        n_train -= reduce_train
                        overflow -= reduce_train
                n_test = max(0, n - n_train - n_val)

                tr = perm[:n_train]
                va = perm[n_train : n_train + n_val]
                te = perm[n_train + n_val :]

                train_idx_list.append(tr)
                val_idx_list.append(va)
                test_idx_list.append(te)
        else:
            # Global (non-stratified) split
            perm = torch.randperm(num_nodes, generator=g)
            n_train = int(round(train_frac * num_nodes))
            n_val = int(round(val_frac * num_nodes))
            n_test = max(0, num_nodes - n_train - n_val)

            train_idx_list = [perm[:n_train]]
            val_idx_list = [perm[n_train : n_train + n_val]]
            test_idx_list = [perm[n_train + n_val :]]

        # Concatenate and ensure disjointness
        train_idx = (
            torch.cat(train_idx_list)
            if len(train_idx_list)
            else torch.empty(0, dtype=torch.long)
        )
        val_idx = (
            torch.cat(val_idx_list)
            if len(val_idx_list)
            else torch.empty(0, dtype=torch.long)
        )
        test_idx = (
            torch.cat(test_idx_list)
            if len(test_idx_list)
            else torch.empty(0, dtype=torch.long)
        )

        # Deduplicate in the unlikely event of overlaps due to rounding
        def unique_keep(t):
            return torch.unique(t, sorted=False)

        train_idx = unique_keep(train_idx)
        # Remove any accidental overlaps
        val_idx = unique_keep(val_idx[~torch.isin(val_idx, train_idx)])
        test_idx = unique_keep(
            test_idx[~torch.isin(test_idx, train_idx) & ~torch.isin(test_idx, val_idx)]
        )

        # Build boolean masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # Sanity: full coverage (if any leftover due to rounding, push to test)
        leftover = ~(train_mask | val_mask | test_mask)
        if leftover.any():
            test_mask[leftover] = True
            test_idx = torch.cat([test_idx, leftover.nonzero(as_tuple=False).view(-1)])

        # Attach new masks & indices
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_idx = train_idx
        data.val_idx = val_idx
        data.test_idx = test_idx

        # Convenience: record meta
        data._custom_split = dict(
            seed=int(seed),
            train_frac=float(train_frac),
            val_frac=float(val_frac),
            test_frac=float(max(0.0, 1.0 - train_frac - val_frac)),
            stratified=bool(stratified),
            min_per_class=int(min_per_class),
            num_nodes=int(num_nodes),
            per_class_counts={
                int(c): int((y == c).sum()) for c in torch.unique(y).tolist()
            },
        )

        print(
            f"[split] Custom 60/20/20 applied (stratified={stratified}, seed={seed}) "
            f"| train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
        )
