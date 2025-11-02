import os
import random
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from typing import List
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from app.nn.hier.focal_loss import FocalLoss
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
from app.utils.build_subgraphs_from_neighbors import (
    build_template_subgraphs_from_neighbors,
)
from app.utils.plotting import plot_training_loss_curve
from app.utils.prediction_utils import cleanup_cuda
from data.blast.blast import BLAST
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
)


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

    def prepare_dataset(self, input_protein: str):
        """
        Orchestrates dataset build and splitting for hierarchical prediction:
        1) Build subgraphs per template and attach template ids.
        2) Print global stats.
        3) Split into train/val(/test) using template-wise (default) or subgraph-wise split.
        4) Print split stats and create loaders.

        After this:
        - self.train_loader / self.val_loader / self.test_loader will be available.
        - self.node_embd_dim / self.edge_prop_dim set for model construction.
        """
        print("[!] Preparing hierarchical subgraph dataset...")
        params = self.prediction_params

        # 1) Build all subgraphs (template-by-template)
        subgraphs_all = self._build_template_subgraphs(input_protein)

        # 2) Global stats
        self._compute_label_stats(subgraphs_all)

        # 3) Split
        split_by = params.get("protein_split_by", "template").lower()
        val_ratio = float(params.get("val_ratio", 0.30))
        train_set, val_set, test_set = self._split_subgraphs(
            split_by=split_by, val_ratio=val_ratio, subgraphs_all=subgraphs_all
        )

        # 4) Report and make loaders
        print("[INFO] Split summary:")
        self._report_split("train", train_set)
        self._report_split("val", val_set)
        if test_set:
            self._report_split("test", test_set)

        self._make_dataloaders(train_set, val_set, test_set)

        # Dataset logs
        subgraph_method = self.prediction_params.get("subg_gen_method", "color")
        exposure_percent = self.prediction_params.get("asa_exposure_percent", 60)
        n_templates = len({g.template_id for g in subgraphs_all})
        split_by = self.prediction_params.get("protein_split_by", "template").upper()
        print(
            f"[+] {subgraph_method.upper()} method {exposure_percent}% | "
            f"Loaded {len(subgraphs_all)} subgraphs from {n_templates} templates. "
            f"→ {split_by} split | train={len(train_set)} | val={len(val_set)}"
            + (f" | test={len(test_set)}" if test_set else "")
        )

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

        Supported gnn_type (case-insensitive):
        - 'GAT' → GATContextSubgraphClassifier
        - 'PNA' → PNAContextSubgraphClassifier
        - 'GIN' → GINContextSubgraphClassifier
        - 'GCN' → GCNContextSubgraphClassifier
        """
        params = self.prediction_params["subgraph_classifier"]

        print(
            f"Initializing subgraph classifier | {params['gnn_type']} | "
            f"{params['num_layers']} layers | {params['norm_type']} norm | {params['pool_type']} pooling"
        )

        gnn_type = params["gnn_type"].upper()

        if gnn_type == "GAT":
            self.subgraph_model = GATContextSubgraphClassifier(
                input_dim=self.node_embd_dim,
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
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

        elif gnn_type == "GIN":
            # Use the context-ready GIN variant
            self.subgraph_model = GINContextSubgraphClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                pool_type=params["pool_type"],
            ).to(self._device)

        elif gnn_type == "GCN":
            self.subgraph_model = GCNContextSubgraphClassifier(
                input_dim=self.node_embd_dim,
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
        """
        params = self.prediction_params["node_classifier"]
        gnn_type = params["gnn_type"].upper()
        print(f"Initializing node classifier | {gnn_type} | {params['num_layers']}")

        # === Instantiate per GNN type ===
        if gnn_type == "GCN":
            self.node_model = GCNNodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
            ).to(self._device)

        elif gnn_type == "GAT":
            self.node_model = GATNodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                num_heads=params["num_heads"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
            ).to(self._device)

        elif gnn_type == "GIN":
            self.node_model = GINNodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
            ).to(self._device)

        elif gnn_type in ("SAGE", "GRAPH_SAGE"):
            self.node_model = SAGENodeClassifier(
                input_dim=self.node_embd_dim,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                num_layers=params["num_layers"],
                norm_type=params["norm_type"],
                aggr=params.get("aggr", "mean"),
            ).to(self._device)
        else:
            raise ValueError("[ERROR] No valid GNN type for node classifier.")

    def train_subgraph_classifier(self):
        """
        Train the subgraph-level classifier (GNN1) and calibrate its probabilities.

        What this function does:
        1) Train with AdamW + BCEWithLogits, optional pos_weight for imbalance.
        2) Validate each epoch, compute PR-AUC on VAL and keep an EMA(PR-AUC).
        3) Early stopping when EMA(PR-AUC) stops improving for a patience window
            (after a cooldown), then restore the best checkpoint.
        4) (If enabled) Calibrate logits with Temperature Scaling on the VAL split
            → set a stable decision threshold at 0.5 for calibrated probabilities.

        Why PR-AUC for early stop?
        - PR-AUC does not depend on a specific threshold and is robust when class
            imbalance and prevalence vary across validation folds/splits.

        Why temperature scaling + 0.5?
        - Temperature scaling calibrates the *shape/scale* of logits, so that
            sigmoid(logit/T) becomes well-calibrated. With well-calibrated probs,
            a fixed threshold of 0.5 is both simple and stable across runs/splits.

        Params expected in self.prediction_params['subgraph_classifier']:
        - lr, weight_decay, epochs
        - early_stop_patience, early_stop_min_delta, early_stop_cooldown
        - use_temperature_calibration (bool, default True)
        - ema_alpha (optional, default 0.9) → smoothing factor for EMA(PR-AUC)

        Side effects:
        - Saves best checkpoint in-memory (restores state_dict).
        - Persists to params:
            * 'temperature' (float, if calibration is used)
            * 'prediction_threshold' (float; 0.5 if calibration is used, otherwise unchanged)

        Notes:
        - Requires self.train_loader and (optionally) self.val_loader.
        - If self.val_loader is missing, we train full and skip ES+calibration.
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.metrics import average_precision_score

        print(f"[!] Starting subgraph-level training... | Device: {self._device}")
        P = self.prediction_params.get("subgraph_classifier", {})
        use_temp_calib = bool(P.get("use_temperature_calibration", True))

        # ---------------------------
        # Helpers
        # ---------------------------
        def _compute_pos_weight_safe(loader, device):
            """
            Optional class-imbalance compensation: pos_weight for BCEWithLogitsLoss.
            Uses self._compute_pos_weight_from_loader if available; otherwise None.
            """
            if hasattr(self, "_compute_pos_weight_from_loader"):
                try:
                    return self._compute_pos_weight_from_loader(loader, device)
                except Exception:
                    return None
            return None

        @torch.no_grad()
        def _collect_split_logits_and_labels(loader):
            """
            Forward the current model on a loader and return:
                logits: np.array [num_graphs]
                labels: np.array [num_graphs] in {0,1}
            """
            self.subgraph_model.eval()
            all_logits, all_labels = [], []
            for batch in loader:
                batch = batch.to(self._device)
                # Forward: model must output per-graph logits [B, 1] (no activation)
                logits = self.subgraph_model.forward_from_data(batch).view(-1)  # [B]
                all_logits.append(logits.detach().cpu())
                all_labels.append(batch.y.view(-1).detach().cpu())
            if len(all_logits) == 0:
                return np.array([]), np.array([])
            logits = torch.cat(all_logits, dim=0).numpy()
            labels = torch.cat(all_labels, dim=0).numpy().astype(int)
            return logits, labels

        def _fit_temperature_on_val(logits_val, y_val, max_iter=50):
            """
            1-parameter temperature scaling: find T>0 minimizing NLL on validation.
            Returns float(T). If val is degenerate, returns T=1.0.
            """
            import torch
            import torch.optim as optim

            if logits_val.size == 0 or len(np.unique(y_val)) < 2:
                return 1.0  # nothing to calibrate

            z = torch.tensor(logits_val, dtype=torch.float32).view(-1, 1)
            y = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

            T = torch.nn.Parameter(torch.ones(1))
            bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
            opt = optim.LBFGS([T], lr=0.1, max_iter=max_iter)

            def closure():
                opt.zero_grad()
                loss = bce(z / T.clamp_min(1e-3), y)
                loss.backward()
                return loss

            opt.step(closure)
            return float(T.detach().clamp_min(1e-3))

        def _apply_temperature(logits, T):
            """Return calibrated probabilities: sigmoid(logits / T)."""
            z = torch.as_tensor(logits, dtype=torch.float32) / float(T)
            return torch.sigmoid(z).cpu().numpy()

        # ---------------------------
        # Setup
        # ---------------------------
        self.subgraph_model.train()

        optimizer = torch.optim.AdamW(
            self.subgraph_model.parameters(),
            lr=P["lr"],
            weight_decay=P["weight_decay"],
        )

        # Optional pos_weight for BCE
        pos_weight = _compute_pos_weight_safe(self.train_loader, self._device)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=(pos_weight.to(self._device) if pos_weight is not None else None)
        )

        # Early-stopping by EMA(PR-AUC) on validation
        use_val = getattr(self, "val_loader", None) is not None
        patience = int(P.get("early_stop_patience", 25))
        min_delta = float(P.get("early_stop_min_delta", 0.003))
        cooldown = int(P.get("early_stop_cooldown", 5))  # allow some warmup before ES
        ema_alpha = float(P.get("ema_alpha", 0.9))  # higher = smoother

        best_state = None
        best_epoch = -1
        best_ema_pr = -np.inf
        epochs_no_improve = 0

        ema_pr = None
        train_loss_hist = []

        total_epochs = int(P["epochs"])

        # ---------------------------
        # Training loop
        # ---------------------------
        for epoch in range(1, total_epochs + 1):
            self.subgraph_model.train()
            ep_loss = 0.0

            # --- Train one epoch ---
            for batch in self.train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                logits = self.subgraph_model.forward_from_data(batch)  # [B,1]
                labels = batch.y.float().unsqueeze(1)  # [B,1]

                loss = loss_fn(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.subgraph_model.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += float(loss.item())

            avg_train_loss = ep_loss / max(1, len(self.train_loader))
            train_loss_hist.append(avg_train_loss)

            # --- Validation: compute PR-AUC and update EMA ---
            this_pr = None
            if use_val:
                logits_val, y_val = _collect_split_logits_and_labels(self.val_loader)
                if logits_val.size > 0 and len(np.unique(y_val)) > 1:
                    try:
                        probs_val = torch.sigmoid(torch.tensor(logits_val)).numpy()
                        this_pr = float(average_precision_score(y_val, probs_val))
                    except Exception:
                        this_pr = None

                # Update EMA(PR-AUC)
                if this_pr is not None:
                    ema_pr = (
                        this_pr
                        if (ema_pr is None)
                        else (ema_alpha * ema_pr + (1 - ema_alpha) * this_pr)
                    )

                    # Best-checkpoint logic after cooldown
                    if epoch > cooldown:
                        improved = (ema_pr - best_ema_pr) > min_delta
                        if improved:
                            best_ema_pr = ema_pr
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
                                    f"[EarlyStop] No EMA(PR-AUC)_val improvement > {min_delta:.4f} "
                                    f"for {patience} epochs after cooldown. Stopping at epoch {epoch}."
                                )
                                break

            # --- Logging (every 20 epochs or last) ---
            if (epoch % 20 == 0) or (epoch == total_epochs):
                if use_val:
                    pr_str = f"{this_pr:.4f}" if this_pr is not None else "n/a"
                    ema_str = f"{ema_pr:.4f}" if ema_pr is not None else "n/a"
                    print(
                        f"Epoch {epoch:03d} | TrainLoss: {avg_train_loss:.4f} | "
                        f"Val PR-AUC: {pr_str} (EMA: {ema_str})"
                    )
                else:
                    print(f"Epoch {epoch:03d} | TrainLoss: {avg_train_loss:.4f}")

        # ---------------------------
        # Restore best checkpoint
        # ---------------------------
        if use_val and best_state is not None:
            self.subgraph_model.load_state_dict(best_state)
            print(
                f"[Best] Restored best GNN1 @ epoch {best_epoch} | EMA(PR-AUC)_val≈{best_ema_pr:.4f}"
            )
        elif not use_val:
            print(
                "[Info] No validation loader found: trained full model without early stopping/calibration."
            )

        # ---------------------------
        # Probability calibration (Temperature Scaling) + fixed threshold 0.5
        # ---------------------------
        if use_val and use_temp_calib:
            # 1) collect logits on validation using the *restored best* model
            logits_val, y_val = _collect_split_logits_and_labels(self.val_loader)
            # 2) fit temperature (robust to degenerate val)
            T = _fit_temperature_on_val(logits_val, y_val, max_iter=50)
            # 3) persist: calibrated inference will be sigmoid(logit / T)
            self.prediction_params["subgraph_classifier"]["temperature"] = float(T)
            # 4) stable decision rule: fixed threshold at 0.5 over calibrated probs
            self.prediction_params["subgraph_classifier"]["prediction_threshold"] = 0.5
            print(
                f"[Calib] Temperature scaling fitted: T={T:.3f} | threshold set to 0.5 (on calibrated probs)."
            )
        else:
            # No VAL or calibration disabled → keep configured threshold as-is (or default 0.5 if none)
            thr = float(
                self.prediction_params["subgraph_classifier"].get(
                    "prediction_threshold", 0.5
                )
            )
            if not use_val:
                print(
                    f"[Calib] Skipped calibration (no val). Keeping threshold={thr:.3f}."
                )
            elif not use_temp_calib:
                print(
                    f"[Calib] Calibration disabled by flag. Keeping threshold={thr:.3f}."
                )

        # ---------------------------
        # Save learning curve
        # ---------------------------
        out_dir = self.dirs["output"]["prot_out_dir"]
        training_loss_curve_out_path = os.path.join(
            out_dir, f"subgraph_classifier_train_loss_class.html"
        )
        plot_training_loss_curve(
            {"train_loss": train_loss_hist}, output_path=training_loss_curve_out_path
        )

    def train_subgraph_classifier_old(self):
        """
        Train the subgraph-level GNN classifier (GNN1).

        Training setup:
            - Optimizer: AdamW
            - Loss: BCEWithLogitsLoss (binary classification at subgraph level)
            - Metrics: F1-score and MCC (computed at the end of each epoch)

        The model learns to predict whether each subgraph contains at least one
        node of the target class. After training, this classifier can be used in
        two ways:
            (1) As a standalone subgraph predictor (evaluation at subgraph level).
            (2) As a context provider for GNN2 (node-level), where its logits or
                embeddings are broadcast to nodes for bias fusion.
        """

        print(f"[!] Starting subgraph-level training... | Device: {self._device}")
        params = self.prediction_params["subgraph_classifier"]

        self.subgraph_model.train()
        optimizer = torch.optim.AdamW(
            self.subgraph_model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        # Binary classification loss (logits vs. {0,1} labels)
        loss_fn = nn.BCEWithLogitsLoss()
        loss_history = []

        # === Training loop ===
        for epoch in range(1, params["epochs"] + 1):
            epoch_loss = 0.0
            y_true, y_pred = [], []

            for batch in self.train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                # Forward pass: predict logits for each subgraph in the batch
                logits = self.subgraph_model(batch.x, batch.edge_index, batch.batch)

                # Prepare labels: [B, 1] binary labels per subgraph
                labels = batch.y.float().unsqueeze(1)

                # Compute loss
                loss = loss_fn(logits, labels)

                # Compute predictions for metrics
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)

                # Accumulate ground-truth and predictions
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(preds)

                # Backward + update
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # === Metrics per epoch ===
            mcc = matthews_corrcoef(y_true, y_pred)
            f1_s = f1_score(y_true, y_pred, average="binary")
            loss_history.append(epoch_loss / len(self.train_loader))

            # Optional: print every 20 epochs
            if epoch % 20 == 0 or epoch == params["epochs"]:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Loss: {loss_history[-1]:.4f} | "
                    f"F1 Score: {f1_s:.4f} | MCC: {mcc:.4f}"
                )

        # === Plot training loss curve ===
        training_loss_curve_out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"subgraph_classifier_training_loss_dashboard_class.html",
        )
        plot_training_loss_curve(
            {1: loss_history}, output_path=training_loss_curve_out_path
        )

    def train_node_classifier_old(self):
        """
        Train the node-level classifier (GNN2) with optional context from GNN1.

        Behavior:
        - If use_subgraph_classifier=True AND the node model expects context
            (fusion_mode in {'concat','film'}), we build per-node context from GNN1
            and fuse it inside the node model.
        - Otherwise, we run vanilla node training (no context).

        Compatible with:
        - Planetoids (has data.train_mask; split-aware masking)
        - Proteins (no masks; binary task; target class is 1)

        Config keys used:
        HIER_PARAMS["use_subgraph_classifier"] : bool
        HIER_PARAMS["subg_gen_method"]         : "anchor" | "color" (defines node_labeling_mode)
        HIER_PARAMS["node_classifier"]         : dict with:
            - batch_size, lr, weight_decay, epochs
            - context_norm: 'none'|'layernorm'|'batch_zscore'|'l2'
            - context_anneal: 'none'|'linear'|'power'  (optional; default 'none')
        """
        print(f"[!] Starting node-level training (context-aware)... | {self._device}")

        # ---------------------------
        # 1) Read config
        # ---------------------------
        params_all = self.prediction_params
        node_params = params_all["node_classifier"]
        subg_method = params_all.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_method == "anchor" else "all_nodes"

        batch_size = node_params.get("batch_size", 16)
        lr = node_params.get("lr", 1e-3)
        wd = node_params.get("weight_decay", 0.0)
        epochs = node_params.get("epochs", 100)

        # ---------------------------
        # 2) Collect subgraphs
        # ---------------------------
        if self.train_loader is None:
            raise ValueError(
                "Train subgraph DataLoader (self.train_loader) not initialized."
            )

        subgraphs = list(self.train_loader.dataset)
        print(
            f"[INFO] Using {len(subgraphs)} train subgraphs | mode={node_labeling_mode}"
        )

        loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)

        # ---------------------------
        # 3) Loss & class imbalance
        # ---------------------------
        pos_weight = self._compute_pos_weight(subgraphs)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        print(f"[INFO] BCEWithLogits | pos_weight={pos_weight.item():.3f}")

        # ---------------------------
        # 4) Optimizer
        # ---------------------------
        optimizer = torch.optim.Adam(
            self.node_model.parameters(), lr=lr, weight_decay=wd
        )

        # ---------------------------
        # 5) Training loop
        # ---------------------------
        loss_history = []

        for epoch in range(1, epochs + 1):
            self.node_model.train()
            total_loss, correct, total = 0.0, 0, 0

            for batch in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                # ---------- Forward ----------
                logits = self.node_model(batch.x, batch.edge_index).view(-1)

                # ---------- Supervision ----------
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
                    loss = loss_fn(logits, y).mean()

                    # Update metrics
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) >= 0.5).long()
                        correct += (preds == y.long()).sum().item()
                        total += y.numel()

                # ---------- Optimize ----------
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # epoch summary
            avg_loss = total_loss / max(1, len(loader))
            acc = correct / max(1, total)
            loss_history.append(avg_loss)
            if epoch % 20 == 0 or epoch == node_params["epochs"]:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc(sup): {acc:.4f}")

        # ---------------------------
        # 6) Plot training curve
        # ---------------------------
        out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_training_loss_dashboard_class_{getattr(self, 'target_class', 'bin')}.html",
        )
        plot_training_loss_curve({1: loss_history}, output_path=out_path)

    def train_node_classifier(self):
        """
        Train the node-level classifier (GNN2) on PROTEIN datasets (no masks/inductive):
        - Uses the same train/val loaders you created for GNN1 (template split).
        - Optional context from GNN1 (fusion_mode in {'concat','film'}) exactly like at inference.
        - Early stopping driven by the EMA of PR-AUC on the validation split.
        - Decision threshold is calibrated on the validation split by maximizing F1.
        - The calibrated threshold is persisted into self.prediction_params['node_classifier']['prediction_threshold'].

        Why this design:
        • BCEWithLogits is optimized in training; we DO NOT optimize a threshold in the loss.
        • On each epoch we collect validation probabilities → compute PR-AUC (for stable ES)
            and scan thresholds to pick the one that maximizes F1 (for the final decision rule).
        • We smooth PR-AUC with EMA to avoid noisy stops in small/imbalanced validation sets.

        Requirements:
        - self.train_loader / self.val_loader (PyG DataLoader over subgraphs).
        - Each subgraph has:
            . node_labels: [N] 0/1 labels for residues (binary)
            . ego_nodes   : list[str] (stable residue ids in local order)   [only used in inference]
            . ego_center_index (int)  [required if using node_labeling_mode='anchor']
        - If using context:
            . self.subgraph_model.get_subgraph_embeddings_from_data(batch) → [B, d]
            . node_model accepts ctx_nodes=[N, d] when fusion_mode in {'concat','film'}

        Config keys used (self.prediction_params):
        - 'subg_gen_method': 'anchor' or 'color' → sets node_labeling_mode ('anchor'|'all_nodes')
        - 'use_subgraph_classifier': bool
        - 'node_classifier': {
                'batch_size', 'lr', 'weight_decay', 'epochs',
                'fusion_mode', 'context_norm' ('none'|'layernorm'|'batch_zscore'|'l2'),
                'context_anneal' ('none'|'linear'|'power'),   # optional
                # Early stop (EMA of PR-AUC):
                'ema_alpha' (default 0.9), 'early_stop_patience' (default 20),
                'early_stop_min_delta' (default 1e-3),
                # Threshold (will be overwritten at the end if val is available):
                'prediction_threshold' (default 0.5),
            }
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from sklearn.metrics import (
            precision_recall_fscore_support,
            f1_score,
            average_precision_score,  # PR-AUC
        )

        print(
            f"[!] Starting node-level training (proteins; val-driven ES+calib) ... | Device: {self._device}"
        )

        # ---------- 0) Resolve config ----------
        P = self.prediction_params
        ND = P["node_classifier"]
        subg_method = P.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_method == "anchor" else "all_nodes"

        batch_size = int(ND.get("batch_size", 32))
        epochs = int(ND.get("epochs", 100))
        lr = float(ND.get("lr", 1e-3))
        wd = float(ND.get("weight_decay", 0.0))

        # Early stopping / EMA(PR-AUC)
        ema_alpha = float(ND.get("ema_alpha", 0.90))
        patience = int(ND.get("early_stop_patience", 20))
        min_delta = float(ND.get("early_stop_min_delta", 3e-3))

        base_threshold = float(
            ND.get("prediction_threshold", 0.5)
        )  # fallback if no val

        # ---------- 1) Loaders ----------
        if self.train_loader is None:
            raise RuntimeError(
                "train_loader is required (use the same one created for GNN1)."
            )
        if getattr(self, "val_loader", None) is None:
            print(
                "[WARN] val_loader is None → training will run without ES/threshold calibration."
            )
        train_loader = self.train_loader
        val_loader = getattr(self, "val_loader", None)

        # ---------- 2) Context usage ----------
        fusion_mode = getattr(self.node_model, "fusion_mode", "none").lower()
        use_context = bool(P.get("use_subgraph_classifier", False)) and fusion_mode in {
            "concat",
            "film",
        }

        if use_context:
            if not hasattr(self, "subgraph_model") or self.subgraph_model is None:
                print(
                    "[WARN] use_subgraph_classifier=True but subgraph_model is missing; proceeding WITHOUT context."
                )
                use_context = False
            else:
                self.subgraph_model.eval()
                ctx_dim = getattr(self.node_model, "context_dim", None)
                gnn1_dim = getattr(self.subgraph_model, "graph_emb_dim", None)
                if ctx_dim is not None and gnn1_dim is not None and ctx_dim != gnn1_dim:
                    raise ValueError(
                        f"[ERROR] node.context_dim ({ctx_dim}) must match subgraph_model.graph_emb_dim ({gnn1_dim})."
                    )
                print(
                    f"[INFO] Using subgraph context | fusion_mode={fusion_mode} | context_norm={ND.get('context_norm','none')}"
                )

        # ---------- 3) Loss / class imbalance ----------
        # Reuse your helper that inspects node labels inside the subgraphs pool.
        # (If you only have a binary task 0/1, this will compute pos_weight for BCE.)
        pos_weight = self._compute_pos_weight(list(train_loader.dataset))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        print(
            f"[INFO] BCEWithLogitsLoss | pos_weight={pos_weight.item():.3f} | mode={node_labeling_mode} | context={use_context} | batch_size={self.train_loader.batch_size}"
        )

        # ---------- 4) Optimizer ----------
        optimizer = torch.optim.Adam(
            self.node_model.parameters(), lr=lr, weight_decay=wd
        )

        # ---------- 5) Small helpers ----------
        def _normalize_ctx(ctx: torch.Tensor, mode: str):
            return self._normalize_context_vectors(ctx, mode=mode)

        def _build_ctx_nodes(batch, scale: float = 1.0):
            """Build per-node context from GNN1: Z_s (per subgraph) → broadcast to nodes via batch.batch."""
            if not use_context:
                return None
            with torch.no_grad():
                Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                    batch
                )  # [B,d]
            ctx = Z_s[batch.batch]  # [N,d]
            ctx = _normalize_ctx(ctx, ND.get("context_norm", "none"))
            if fusion_mode == "concat":
                ctx = scale * ctx  # trainer-side anneal only for concat
            elif fusion_mode == "film" and hasattr(
                self.node_model, "set_context_scale"
            ):
                # For FiLM we use a model-side scale (see below)
                pass
            return ctx

        def _epoch_pass(loader: DataLoader, eval_mode: bool, epoch_idx: int):
            """
            One pass over a loader:
            - Returns avg BCE loss over supervised items,
            - Collects y_true, y_prob for metric computation / threshold scan.
            Supervision matches `node_labeling_mode`.
            """
            if eval_mode:
                self.node_model.eval()
            else:
                self.node_model.train()

            total_loss, denom_total = 0.0, 0
            y_true_all, y_prob_all = [], []

            for batch in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad(set_to_none=True)

                # (Optional) Anneal context influence across epochs (only meaningful in training)
                if (
                    fusion_mode == "film"
                    and use_context
                    and hasattr(self.node_model, "set_context_scale")
                ):
                    # simple linear anneal (can be changed to 'power' if you prefer)
                    scale = (
                        float(epoch_idx) / float(max(1, epochs))
                        if not eval_mode
                        else 1.0
                    )
                    self.node_model.set_context_scale(scale)
                    ctx_nodes = _build_ctx_nodes(
                        batch, scale=1.0
                    )  # scale handled in model
                else:
                    scale = (
                        float(epoch_idx) / float(max(1, epochs))
                        if not eval_mode
                        else 1.0
                    )
                    ctx_nodes = _build_ctx_nodes(batch, scale=scale)

                # Forward (with or without context)
                if use_context and ctx_nodes is not None:
                    logits = self.node_model(
                        batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                    ).view(-1)
                else:
                    logits = self.node_model(batch.x, batch.edge_index).view(-1)

                # Labels (binary 0/1 for proteins)
                y = batch.node_labels.float().to(self._device)

                # Supervision
                if node_labeling_mode == "anchor":
                    if not hasattr(batch, "ego_center_index"):
                        raise RuntimeError(
                            "Anchor mode requires 'ego_center_index' in each subgraph."
                        )
                    centers = batch.ego_center_index.view(-1).long()
                    logit_sup = logits[centers]
                    y_sup = y[centers]
                    loss_vec = loss_fn(logit_sup, y_sup)
                    loss = loss_vec.mean()
                    probs_sup = (
                        torch.sigmoid(logit_sup).detach().cpu().numpy().flatten()
                    )
                    y_sup_np = y_sup.detach().cpu().numpy().astype(int).flatten()
                    denom = y_sup.numel()
                else:
                    # all_nodes
                    loss_vec = loss_fn(logits, y)
                    loss = loss_vec.mean()
                    probs_sup = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                    y_sup_np = y.detach().cpu().numpy().astype(int).flatten()
                    denom = y.numel()

                # Optimize
                if not eval_mode:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item())
                denom_total += max(1, int(denom))
                y_true_all += y_sup_np.tolist()
                y_prob_all += probs_sup.tolist()

            avg_loss = total_loss / max(1, len(loader))
            y_true_all = np.asarray(y_true_all, dtype=int)
            y_prob_all = np.asarray(y_prob_all, dtype=float)
            return avg_loss, y_true_all, y_prob_all

        def _scan_best_threshold_by_f1(
            y_true: np.ndarray, y_prob: np.ndarray, default: float = 0.5
        ):
            """
            Pick the threshold that maximizes F1 on (y_true, y_prob).
            If all probs are identical or only one class appears, fall back to `default`.
            """
            if (
                y_true.size == 0
                or np.unique(y_true).size < 2
                or np.allclose(y_prob.max(), y_prob.min())
            ):
                return float(default), 0.0, 0.0, 0.0  # thr, f1, prec, rec

            # Use unique probability cut points (sorted). Evaluate F1 at each.
            ths = np.unique(y_prob)
            # To include a low threshold as well
            ths = np.concatenate(([0.0], ths, [1.0]))
            best_thr, best_f1, best_p, best_r = float(default), -1.0, 0.0, 0.0
            for t in ths:
                y_pred = (y_prob >= t).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                if (f1 > best_f1) or (np.isclose(f1, best_f1) and p > best_p):
                    best_thr, best_f1, best_p, best_r = (
                        float(t),
                        float(f1),
                        float(p),
                        float(r),
                    )
            return best_thr, best_f1, best_p, best_r

        # ---------- 6) Training loop ----------
        best_state = None
        best_epoch = -1
        best_ema_prauc = -1.0
        ema_prauc = None
        epochs_no_improve = 0

        best_thr = base_threshold
        best_f1 = -1.0
        best_pr = 0.0
        best_rc = 0.0

        train_loss_hist = []
        val_prauc_hist = []  # track raw PR-AUC (not EMA) for plotting if needed

        eval_every = ND.get("eval_every", 5)

        for epoch in range(1, epochs + 1):
            # --- Train epoch ---
            tr_loss, _, _ = _epoch_pass(train_loader, eval_mode=False, epoch_idx=epoch)
            train_loss_hist.append(tr_loss)

            # --- Validation (for ES + threshold calibration) ---
            if (
                val_loader is not None
            ):  # and (epoch % eval_every == 0 or epoch == epochs):
                with torch.no_grad():
                    va_loss, vy_true, vy_prob = _epoch_pass(
                        val_loader, eval_mode=True, epoch_idx=epoch
                    )

                # Probability-based metric for ES: PR-AUC (Average Precision)
                try:
                    pr_auc = float(average_precision_score(vy_true, vy_prob))
                except Exception:
                    pr_auc = 0.0
                val_prauc_hist.append(pr_auc)

                # EMA update
                ema_prauc = (
                    pr_auc
                    if ema_prauc is None
                    else (ema_alpha * ema_prauc + (1.0 - ema_alpha) * pr_auc)
                )

                # Threshold calibration by F1 on validation
                thr_opt, f1_val, p_val, r_val = _scan_best_threshold_by_f1(
                    vy_true, vy_prob, default=base_threshold
                )

                # Keep the best checkpoint by EMA(PR-AUC)
                improved = (ema_prauc - best_ema_prauc) > min_delta
                if improved:
                    best_ema_prauc = ema_prauc
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.node_model.state_dict().items()
                    }
                    # store the threshold that *F1* preferred on this best-EMA epoch
                    best_thr, best_f1, best_pr, best_rc = (
                        float(thr_opt),
                        float(f1_val),
                        float(p_val),
                        float(r_val),
                    )
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Logging
                if (epoch % 20 == 0) or (epoch == epochs):
                    ema_str = f"{ema_prauc:.4f}" if ema_prauc is not None else "n/a"
                    print(
                        f"Epoch {epoch:03d} | "
                        f"TrainLoss: {tr_loss:.4f} | Val PR-AUC: {pr_auc:.4f} (EMA: {ema_str}) | "
                        f"Val F1*: {f1_val:.4f} (thr={thr_opt:.3f} | P={p_val:.3f}, R={r_val:.3f})"
                    )

                # Early stop?
                if epochs_no_improve >= patience:
                    print(
                        f"[EarlyStop] No EMA(PR-AUC)_val improvement > {min_delta:.4f} for {patience} epochs. Stopping at epoch {epoch}."
                    )
                    break
            else:
                # No validation: light logging
                if (epoch % 20 == 0) or (epoch == epochs):
                    print(f"Epoch {epoch:03d} | TrainLoss: {tr_loss:.4f}")

        # ---------- 7) Restore best & persist calibrated threshold ----------
        if val_loader is not None and best_state is not None:
            self.node_model.load_state_dict(best_state)
            self.prediction_params["node_classifier"]["prediction_threshold"] = float(
                best_thr
            )
            print(
                f"[Best] Restored best GNN2 @ epoch {best_epoch} | EMA(PR-AUC)_val≈{best_ema_prauc:.4f} | thr*={best_thr:.3f} (F1)"
            )
            print(
                f"[Best] Val stats at thr*: F1={best_f1:.4f} | P={best_pr:.4f} | R={best_rc:.4f}"
            )
        else:
            # keep configured base threshold
            self.prediction_params["node_classifier"]["prediction_threshold"] = float(
                base_threshold
            )
            print(
                f"[Info] No validation available → kept threshold={base_threshold:.3f}"
            )

        # ---------- 8) Optional: plot training curve ----------
        out_path = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_training_loss_dashboard_class_{getattr(self, 'target_class', 'bin')}.html",
        )
        plot_training_loss_curve({"train_loss": train_loss_hist}, output_path=out_path)

    def predict_binding_sites_old(self, input_subgraphs):
        """
        Binding-site prediction (proteins, inductive mode) WITHOUT any subgraph filtering.
        - All provided subgraphs are used.
        - If `use_subgraph_classifier=True` AND the node_model requires context
        (fusion_mode in {'concat','film'}), we compute per-subgraph embeddings from GNN1
        and broadcast them to nodes as context for GNN2.
        - Otherwise, standard GNN2 inference is used (no context).

        Expected per-subgraph fields:
        - x                : [num_nodes_sub, feat_dim]
        - edge_index       : [2, num_edges_sub]
        - ego_nodes        : List[str] (stable residue identifiers in local order)
        - ego_center_index : int (optional; local anchor index, needed for 'anchor' mode)
        - ego_center       : str (optional; anchor residue id if index not provided)

        Returns:
        dict[str, float] mapping residue identifier -> averaged probability
        """
        print(f"[!] Starting binding-site prediction... | {self._device}")

        # ---------------------------
        # 1) Read configuration
        # ---------------------------
        params = self.prediction_params
        nd_cfg = params.get("node_classifier", {})

        # Node-labeling strategy (anchor-only or all nodes)
        subg_gen_method = params.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        # Batch size for node-level inference
        node_bs = int(nd_cfg.get("batch_size", 32))

        # Put node classifier in eval mode
        self.node_model.eval()

        # ---------------------------
        # 2) Node-level forward pass on ALL subgraphs
        # ---------------------------
        residue_scores = (
            {}
        )  # Dict: residue_id -> list of predicted probabilities (to be averaged later)

        node_loader = DataLoader(input_subgraphs, batch_size=node_bs, shuffle=False)
        with torch.no_grad():
            for batch in node_loader:
                batch = batch.to(self._device)

                # ----- Forward through node model -----
                logits_nodes = self.node_model(batch.x, batch.edge_index).view(-1)

                # Convert logits to probabilities
                probs_nodes = torch.sigmoid(logits_nodes).cpu().numpy()

                # ----- Split concatenated batch back into individual subgraphs -----
                if hasattr(batch, "ptr") and batch.ptr is not None:
                    # ptr contains prefix sums of node counts; len(ptr) = num_graphs + 1
                    ptr = batch.ptr.long()
                else:
                    # Derive ptr from batch.batch if not provided
                    num_nodes_per_graph = torch.bincount(batch.batch)
                    ptr = torch.cat(
                        [
                            num_nodes_per_graph.new_zeros(1),
                            num_nodes_per_graph.cumsum(0),
                        ]
                    ).long()

                graphs = (
                    batch.to_data_list()
                )  # list of Data objects; len = num_graphs in batch

                # ----- Collect probabilities per residue -----
                for g_idx, sg in enumerate(graphs):
                    start, end = int(ptr[g_idx].item()), int(ptr[g_idx + 1].item())
                    probs_g = probs_nodes[
                        start:end
                    ]  # probabilities for nodes of this subgraph

                    ego_nodes = list(sg.ego_nodes)  # stable residue ids in local order

                    if node_labeling_mode == "anchor":
                        # Only predict the anchor residue
                        if hasattr(sg, "ego_center_index"):
                            c_local = int(sg.ego_center_index)
                        else:
                            # Fallback: resolve anchor index by name
                            if not hasattr(sg, "ego_center"):
                                raise RuntimeError(
                                    "Anchor mode requires 'ego_center_index' or 'ego_center'."
                                )
                            try:
                                c_local = ego_nodes.index(sg.ego_center)
                            except ValueError:
                                raise RuntimeError(
                                    f"ego_center '{sg.ego_center}' not found in ego_nodes (len={len(ego_nodes)})."
                                )

                        res_id = str(ego_nodes[c_local])
                        p = float(probs_g[c_local])
                        residue_scores.setdefault(res_id, []).append(p)

                    else:
                        # Predict all residues in the subgraph
                        for j, res_id in enumerate(ego_nodes):
                            residue_scores.setdefault(str(res_id), []).append(
                                float(probs_g[j])
                            )

        # ---------------------------
        # 3) Aggregate scores per residue (mean across subgraphs)
        # ---------------------------
        averaged_scores = {
            res_id: float(np.mean(scores)) for res_id, scores in residue_scores.items()
        }

        print(
            f"[✓] Inference complete. Predicted scores for {len(averaged_scores)} residues. "
        )

        # Cleanup GPU memory if helper is available
        cleanup_cuda(self.subgraph_model)
        cleanup_cuda(self.node_model)

        return averaged_scores

    def predict_binding_sites(self, input_subgraphs):
        """
        Binding-site prediction (proteins, inductive) with optional:
        - Subgraph FILTERING by GNN1 (keep only subgraphs predicted positive).
        - Node-level CONTEXT from GNN1 (concat/FiLM) for GNN2.

        What changes vs. the old version:
        1) (Optional) Filtering: run GNN1 on all input subgraphs, calibrate probs
            (temperature if available), apply the calibrated threshold to keep only
            "positive" subgraphs.
        2) (Optional) Context: if enabled and supported by GNN2, broadcast the
            subgraph embedding from GNN1 to each node and fuse it (concat/FiLM).
        3) Aggregate per-residue scores by averaging node probabilities across
            all appearances of that residue in the kept subgraphs.

        Expected fields in each subgraph Data:
        - x                  : [N, F]
        - edge_index         : [2, E]
        - ego_nodes          : list[str] (stable residue ids, one per local node)
        - ego_center_index   : int (optional; required if node_labeling_mode='anchor')
        - ego_center         : str (fallback id to resolve anchor if index not provided)

        Returns:
        dict[str, float] : residue_id -> mean probability in [0,1].
                            (No thresholding here; downstream may binarize if needed.)
        """
        import numpy as np
        import torch
        from torch_geometric.loader import DataLoader  # important: PyG DataLoader

        print(f"[!] Starting binding-site prediction... | {self._device}")
        assert (
            isinstance(input_subgraphs, (list, tuple)) and len(input_subgraphs) > 0
        ), "input_subgraphs must be a non-empty list of PyG Data objects."

        # ---------------------------
        # 1) Read configuration
        # ---------------------------
        P = self.prediction_params
        nd_cfg = P.get("node_classifier", {})
        sg_cfg = P.get("subgraph_classifier", {})

        # Node-labeling strategy (anchor-only or all nodes)
        subg_gen_method = P.get("subg_gen_method", "color").lower()
        node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

        # Inference batch sizes
        subg_bs = int(sg_cfg.get("batch_size", 64))
        node_bs = int(nd_cfg.get("batch_size", 32))

        # Filtering switch & threshold
        use_filtering = bool(P.get("use_subgraph_filtering", False))
        thr_gnn1 = float(sg_cfg.get("prediction_threshold", 0.5))

        # Optional temperature for calibrated probabilities (GNN1)
        T = sg_cfg.get("temperature", None)
        if T is not None:
            try:
                T = float(T)
                if T <= 0:
                    print(f"[WARN] Invalid temperature={T}; ignoring calibration.")
                    T = None
            except Exception:
                print(
                    f"[WARN] Non-numeric temperature='{sg_cfg.get('temperature')}' ignored."
                )
                T = None

        # Optional top-K rescue when filtering keeps nothing
        topk_if_empty = int(sg_cfg.get("filter_topk_if_empty", 0))

        # Context (concat/FiLM)
        fusion_mode = getattr(self.node_model, "fusion_mode", "none").lower()
        use_context = bool(P.get("use_subgraph_classifier", False)) and fusion_mode in {
            "concat",
            "film",
        }

        # ---------------------------
        # 2) (Optional) FILTERING with GNN1
        # ---------------------------
        kept_subgraphs = input_subgraphs
        if use_filtering:
            if not hasattr(self, "subgraph_model") or self.subgraph_model is None:
                print(
                    "[WARN] use_subgraph_filtering=True but subgraph_model is missing; skipping filtering."
                )
            else:
                self.subgraph_model.eval()
                sg_loader = DataLoader(
                    input_subgraphs, batch_size=subg_bs, shuffle=False
                )

                all_probs = []
                per_graph_counts = []
                with torch.no_grad():
                    for batch in sg_loader:
                        batch = batch.to(self._device)
                        # logits per subgraph [B,1] from your GNN1 forward
                        # (we support either custom forward_from_data or classic forward)
                        if hasattr(self.subgraph_model, "forward_from_data"):
                            logits = self.subgraph_model.forward_from_data(
                                batch
                            )  # [B,1]
                        else:
                            # fall back if your model expects (x, edge_index, batch)
                            logits = self.subgraph_model(
                                batch.x, batch.edge_index, batch.batch
                            )

                        if T is not None:
                            probs = torch.sigmoid(logits / T).view(-1).cpu().numpy()
                        else:
                            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                        all_probs.extend(probs.tolist())
                        per_graph_counts.append(len(probs))

                # Decide which subgraphs to keep
                all_probs = np.asarray(all_probs, dtype=float)
                assert all_probs.shape[0] == len(
                    input_subgraphs
                ), "mismatch between scored subgraphs and input size."

                keep_mask = all_probs >= thr_gnn1
                num_keep = int(keep_mask.sum())
                if num_keep == 0 and topk_if_empty > 0:
                    # rescue: keep top-K highest prob subgraphs
                    top_idx = np.argsort(-all_probs)[:topk_if_empty]
                    keep_mask = np.zeros_like(keep_mask, dtype=bool)
                    keep_mask[top_idx] = True
                    num_keep = int(keep_mask.sum())
                    print(
                        f"[INFO] Filtering kept 0 by threshold={thr_gnn1:.3f}; rescued top-{topk_if_empty} by prob."
                    )
                else:
                    print(T)
                    print(
                        f"[INFO] Filtering kept {num_keep}/{len(input_subgraphs)} (thr={thr_gnn1:.3f}, T={'none' if T is None else T})."
                    )

                kept_subgraphs = [
                    g for g, keep in zip(input_subgraphs, keep_mask) if keep
                ]
                if len(kept_subgraphs) == 0:
                    print(
                        "[WARN] No subgraphs left after filtering; returning empty result."
                    )
                    return {}

        # ---------------------------
        # 3) Node-level inference (with/without CONTEXT)
        # ---------------------------
        self.node_model.eval()
        residue_scores = {}  # residue_id -> list[prob]

        node_loader = DataLoader(kept_subgraphs, batch_size=node_bs, shuffle=False)
        with torch.no_grad():
            for batch in node_loader:
                batch = batch.to(self._device)

                # ---- Optional context from GNN1, normalized as in training ----
                ctx_nodes = None
                if use_context:
                    if (
                        not hasattr(self, "subgraph_model")
                        or self.subgraph_model is None
                    ):
                        print(
                            "[WARN] use_subgraph_classifier=True but subgraph_model missing; inference without context."
                        )
                    else:
                        self.subgraph_model.eval()
                        # [B, d_ctx] → broadcast to [N, d_ctx] via batch mapping
                        Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                            batch
                        )
                        ctx_nodes = Z_s[batch.batch]
                        ctx_nodes = self._normalize_context_vectors(
                            ctx_nodes, mode=nd_cfg.get("context_norm", "none")
                        )
                        # (No annealing at inference → alpha=1.0)

                        if fusion_mode == "film" and hasattr(
                            self.node_model, "set_context_scale"
                        ):
                            self.node_model.set_context_scale(1.0)

                # ---- Forward GNN2 ----
                if use_context:
                    logits_nodes = self.node_model(
                        batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                    ).view(-1)
                else:
                    logits_nodes = self.node_model(batch.x, batch.edge_index).view(-1)
                probs_nodes = torch.sigmoid(logits_nodes).cpu().numpy()

                # ---- Recover node ranges por subgrafo ----
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

                graphs = batch.to_data_list()

                # ---- Coletar por resíduo ----
                for g_idx, sg in enumerate(graphs):
                    start, end = int(ptr[g_idx].item()), int(ptr[g_idx + 1].item())
                    probs_g = probs_nodes[start:end]
                    ego_nodes = list(sg.ego_nodes)

                    if node_labeling_mode == "anchor":
                        # âncora obrigatório
                        if hasattr(sg, "ego_center_index"):
                            c_local = int(sg.ego_center_index)
                        else:
                            if not hasattr(sg, "ego_center"):
                                raise RuntimeError(
                                    "Anchor mode requires 'ego_center_index' or 'ego_center'."
                                )
                            try:
                                c_local = ego_nodes.index(sg.ego_center)
                            except ValueError:
                                raise RuntimeError(
                                    f"ego_center '{sg.ego_center}' not found in ego_nodes (len={len(ego_nodes)})."
                                )

                        res_id = str(ego_nodes[c_local])
                        residue_scores.setdefault(res_id, []).append(
                            float(probs_g[c_local])
                        )
                    else:
                        for j, res_id in enumerate(ego_nodes):
                            residue_scores.setdefault(str(res_id), []).append(
                                float(probs_g[j])
                            )

        # ---------------------------
        # 4) Aggregate (mean prob per residue)
        # ---------------------------
        averaged_scores = {
            rid: float(np.mean(vs)) for rid, vs in residue_scores.items()
        }
        print(
            f"[✓] Inference complete. Residues scored: {len(averaged_scores)} "
            f"| filtering={'on' if use_filtering else 'off'} | context={'on' if use_context else 'off'}."
        )

        # Optional: free GPU memory
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

    def _compute_pos_weight(self, subgraphs_):
        pos = sum(int(sg.node_labels.sum().item()) for sg in subgraphs_)
        n_nodes = sum(int(sg.node_labels.numel()) for sg in subgraphs_)
        neg = n_nodes - pos
        pos = max(1, pos)
        return torch.tensor([neg / pos], device=self._device, dtype=torch.float)

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

    # -----------
    def _build_template_subgraphs(self, input_protein: str) -> list:
        """
        Build subgraphs template-by-template and attach `template_id` to each subgraph.

        Returns:
            subgraphs_all (list): list of PyG Data objects, each with .template_id set.
        """
        params = self.prediction_params
        subgraph_method = params.get("subg_gen_method", "color")
        neighbor_layers = params.get("subg_neighbor_layers", 3)
        top_n_templates = params.get("top_n_templates", 10)
        exposure_percent = params.get("asa_exposure_percent", 60)

        blast = BLAST(self.dirs["data"]["blast"])
        #selected_templates = blast.run(input_protein, top_n_templates)
        selected_templates = {"1a8t_A"}
        print(f"[!] Selected {len(selected_templates)} templates")

        subgraphs_all = []
        for tpt_id in selected_templates:
            if subgraph_method == "color":
                local_subgraphs = self._build_subgraphs_from_template_color(
                    template_id=tpt_id, num_layers=neighbor_layers
                )
            elif subgraph_method == "anchor":
                local_subgraphs = self._build_subgraphs_from_template_anchor(
                    template_id=tpt_id, num_layers=neighbor_layers
                )
            else:  # "asa"
                local_subgraphs = self._build_subgraphs_from_template_asa(
                    template_id=tpt_id,
                    num_layers=neighbor_layers,
                    exposure_percent=exposure_percent,
                )

            for g in local_subgraphs:
                setattr(g, "template_id", tpt_id)

            subgraphs_all.extend(local_subgraphs)

        if not subgraphs_all:
            raise ValueError("[ERROR] No subgraphs found for selected templates.")

        # Cache dims for later models
        self.node_embd_dim = subgraphs_all[0].x.shape[1]
        self.edge_prop_dim = (
            subgraphs_all[0].edge_attr.shape[1]
            if getattr(subgraphs_all[0], "edge_attr", None) is not None
            else 0
        )

        return subgraphs_all

    def _compute_label_stats(self, subgraphs: list) -> dict:
        """
        Compute and print global label stats for a list of subgraphs.

        Returns:
            dict with keys: num_pos, num_neg, pos_ratio, avg_site_ratio
        """
        import numpy as np

        ys = np.array([int(g.y.item()) for g in subgraphs], dtype=int)
        num_pos = int((ys == 1).sum())
        num_neg = int((ys == 0).sum())
        pos_ratio = num_pos / len(ys) if len(ys) > 0 else float("nan")
        site_ratios = [float(getattr(g, "site_ratio", 0.0)) for g in subgraphs]
        avg_site_ratio = (
            float(np.mean(site_ratios)) if len(site_ratios) > 0 else float("nan")
        )

        print("[INFO] Subgraph label distribution (ALL):")
        print(f"       Positive (binding):   {num_pos}")
        print(f"       Negative (non-bind.): {num_neg}")
        print(f"       Positive ratio:       {pos_ratio:.2%}")
        print(f"       Avg. site ratio:      {avg_site_ratio:.3f}")

        return dict(
            num_pos=num_pos,
            num_neg=num_neg,
            pos_ratio=pos_ratio,
            avg_site_ratio=avg_site_ratio,
        )

    def _report_split(self, name: str, subset: list) -> None:
        """
        Small helper to print label distribution for a subset.
        """
        import numpy as np

        if len(subset) == 0:
            print(f"  - {name}: 0")
            return
        yy = np.array([int(g.y.item()) for g in subset], dtype=int)
        p = int((yy == 1).sum())
        n = int((yy == 0).sum())
        r = p / len(yy)
        print(f"  - {name}: {len(subset)} | +:{p}  -:{n}  (+ratio={r:.2%})")

    def _split_subgraphs(
        self, split_by, val_ratio, subgraphs_all: list
    ) -> tuple[list, list, list]:
        """
        Split subgraphs into train/val(/test) according to config:
        - protein_split_by: 'template' (recommended) or 'subgraph'
        - val_ratio, test_ratio, split_seed

        Returns:
            (train_set, val_set, test_set) — test_set may be [] if test_ratio == 0.
        """
        import numpy as np

        params = self.prediction_params
        test_ratio = float(params.get("test_ratio", 0.00))
        split_seed = int(params.get("split_seed", 42))

        assert split_by in {
            "template",
            "subgraph",
        }, "protein_split_by must be 'template' or 'subgraph'"
        assert 0.0 <= val_ratio < 1.0, "val_ratio must be in [0,1)"

        rng = np.random.RandomState(split_seed)

        if split_by == "template":
            # Group-wise split: keep each template entirely in a single split
            templates = sorted({g.template_id for g in subgraphs_all})
            templates = np.array(templates)
            rng.shuffle(templates)

            n_templates = len(templates)
            n_test = int(round(test_ratio * n_templates))
            n_val = int(round(val_ratio * n_templates))

            test_temps = set(templates[:n_test]) if n_test > 0 else set()
            val_temps = set(templates[n_test : n_test + n_val]) if n_val > 0 else set()
            train_temps = set(templates[n_test + n_val :])

            train_set = [g for g in subgraphs_all if g.template_id in train_temps]
            val_set = [g for g in subgraphs_all if g.template_id in val_temps]
            test_set = [g for g in subgraphs_all if g.template_id in test_temps]

        else:
            # Subgraph-wise split (stratified by label). Beware of potential leakage across templates.
            from sklearn.model_selection import train_test_split

            ys = np.array([int(g.y.item()) for g in subgraphs_all], dtype=int)
            idx = np.arange(len(subgraphs_all))

            # Hold-out test if requested
            if test_ratio > 0.0:
                idx_trainval, idx_test = train_test_split(
                    idx, test_size=test_ratio, random_state=split_seed, stratify=ys
                )
                ys_trainval = ys[idx_trainval]
            else:
                idx_trainval, idx_test = idx, np.array([], dtype=int)
                ys_trainval = ys

            # Validation from the remaining pool
            if val_ratio > 0.0:
                val_ratio_rel = val_ratio / (1.0 - test_ratio)
                idx_train, idx_val = train_test_split(
                    idx_trainval,
                    test_size=val_ratio_rel,
                    random_state=split_seed,
                    stratify=ys_trainval,
                )
            else:
                idx_train, idx_val = idx_trainval, np.array([], dtype=int)

            train_set = [subgraphs_all[i] for i in idx_train]
            val_set = [subgraphs_all[i] for i in idx_val]
            test_set = [subgraphs_all[i] for i in idx_test]

        return train_set, val_set, test_set

    def _make_dataloaders(self, train_set: list, val_set: list, test_set: list):
        """
        Create DataLoaders (train/val/test) using configured batch size.
        """

        batch_size = int(self.prediction_params.get("batch_size", 64))
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = (
            DataLoader(val_set, batch_size=batch_size, shuffle=False)
            if len(val_set) > 0
            else None
        )
        self.test_loader = (
            DataLoader(test_set, batch_size=batch_size, shuffle=False)
            if len(test_set) > 0
            else None
        )

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
