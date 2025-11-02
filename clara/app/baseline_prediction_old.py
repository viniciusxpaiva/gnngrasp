def train_subgraph_classifier_old(self):
    """
    Train the subgraph-level GNN classifier (GNN1) using subgraphs generated
    from Planetoid datasets (e.g., Cora, CiteSeer, PubMed).

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

    # Choose which node-feature key to use (e.g., "x" or "x_view_struct")
    x_key = self.prediction_params.get("gnn1_x_key", "x")  # "x_view_struct" if using E1

    # also set on the model (our classifier supports feature routing via x_key)
    if hasattr(self.subgraph_model, "x_key"):
        self.subgraph_model.x_key = x_key

    self.subgraph_model.train()
    optimizer = torch.optim.AdamW(
        self.subgraph_model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    # Binary classification loss (logits vs. {0,1} labels)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history = []

    # ---- Optional: sanity check input_dim vs chosen feature view (only once) ----
    def _check_input_dim_once(batch):
        # feature tensor from the chosen key (fallback to .x if missing)
        feat = getattr(batch, x_key, None)
        if feat is None:
            feat = getattr(batch, "x", None)
        if feat is None:
            raise AttributeError(
                f"No features found for '{x_key}' or 'x' in Batch/Data."
            )
        in_dim_actual = feat.size(1)
        first_layer_in = self.subgraph_model.gcn_layers[0].in_channels
        if in_dim_actual != first_layer_in:
            raise ValueError(
                f"[GNN1] Input dim mismatch: model expects {first_layer_in} "
                f"but '{x_key}' has {in_dim_actual}. "
                f"Initialize the model with input_dim={in_dim_actual} or change 'gnn1_x_key'."
            )

    did_check = False

    # === Training loop ===
    for epoch in range(1, params["epochs"] + 1):
        epoch_loss = 0.0
        y_true, y_pred = [], []

        for batch in self.train_loader:
            batch = batch.to(self._device)
            if not did_check:
                _check_input_dim_once(batch)
                did_check = True
            optimizer.zero_grad()

            # Forward pass: predict logits for each subgraph in the batch
            # This selects batch.<x_key> if present; otherwise falls back to batch.x.
            logits = self.subgraph_model.forward_from_data(batch)

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
        f"subgraph_classifier_training_loss_dashboard_class_{self.target_class}.html",
    )
    plot_training_loss_curve(
        {1: loss_history}, output_path=training_loss_curve_out_path
    )


def train_subgraph_classifier_f_beta(self):
    """
    Train GNN1 (subgraph-level classifier) with recall-oriented strategy.

    Key features:
    - Optimizer: AdamW with weight decay
    - LR schedule: linear warmup + cosine decay
    - Loss: BCEWithLogits + class imbalance correction (pos_weight, with optional boost for recall)
    - Gradient clipping for stability
    - Validation-driven:
        * Early stopping (by Recall or Fβ with β>1)
        * Threshold calibration (by Recall or Fβ)
        * Optional recall_target: pick the smallest threshold that achieves the target recall
    - Curves saved: training loss, validation metric (Recall or Fβ), and validation PR-AUC
    """

    print(
        f"[!] Starting subgraph-level training (recall-oriented)... | Device: {self._device}"
    )
    params = self.prediction_params["subgraph_classifier"]

    # ----------------------------------------------------------------------
    # Step 1. Select which node feature view the subgraph model should use.
    # Example: "x" (original features), "x_view_struct" (structural view), or a concatenated view.
    # ----------------------------------------------------------------------
    x_key = self.prediction_params.get("gnn1_x_key", "x")
    if hasattr(self.subgraph_model, "x_key"):
        self.subgraph_model.x_key = x_key
    self.subgraph_model.train()

    # ----------------------------------------------------------------------
    # Step 2. Configure optimizer and scheduler.
    # AdamW: robust optimizer with decoupled weight decay.
    # Scheduler: warmup + cosine annealing for smoother training and stable convergence.
    # ----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        self.subgraph_model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = self._make_warmup_cosine_scheduler(
        optimizer,
        total_epochs=int(params["epochs"]),
        warmup_epochs=None,  # default ~5% of total epochs
        min_lr_scale=0.1,  # decay down to 10% of the base LR
    )

    # ----------------------------------------------------------------------
    # Step 3. Configure loss function with class imbalance handling.
    # - pos_weight balances positive vs. negative samples.
    # - recall_posweight_boost (>1.0) further increases the weight of positives,
    #   explicitly biasing training toward reducing false negatives.
    # ----------------------------------------------------------------------
    base_pos_weight = self._compute_pos_weight_from_loader(
        self.train_loader, self._device
    )
    boost = float(params.get("recall_posweight_boost", 1.0))
    if base_pos_weight is not None and boost > 1.0:
        pos_weight = base_pos_weight * boost
    else:
        pos_weight = base_pos_weight
    loss_fn = self._make_bce_with_logits(pos_weight=pos_weight)

    # ----------------------------------------------------------------------
    # Step 4. Setup validation, early stopping, and metric mode.
    # - If a validation loader exists, we use it for:
    #   * Threshold calibration (Recall or Fβ)
    #   * Early stopping (based on Recall or Fβ)
    # - patience: how many epochs without improvement before stopping
    # - min_delta: minimum improvement required to reset patience
    # ----------------------------------------------------------------------
    use_val = getattr(self, "val_loader", None) is not None
    patience = int(params.get("early_stop_patience", 30))
    min_delta = float(
        params.get("early_stop_min_delta", 0.002)
    )  # 0.2 percentage points

    metric_mode = str(
        params.get("metric_mode", "f_beta")
    ).lower()  # "recall" or "f_beta"
    beta = float(params.get("beta", 2.0))  # only relevant for Fβ
    recall_target = params.get(
        "recall_target", None
    )  # desired recall target (e.g., 0.9)

    # Track the best model state and threshold seen during training
    best_val_metric = -1.0
    best_thresh = float(params.get("prediction_threshold", 0.5))
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    # History for plotting curves
    train_loss_hist, val_metric_hist, val_prauc_hist = [], [], []
    total_epochs = int(params["epochs"])

    # ----------------------------------------------------------------------
    # Step 5. Internal helpers for threshold calibration.
    # - _scan_thresholds: sweep thresholds from 0..1 and compute metrics
    # - _choose_threshold_recall: select threshold to maximize recall or satisfy recall_target
    # - _choose_threshold_fbeta: select threshold that maximizes Fβ
    # ----------------------------------------------------------------------
    def _scan_thresholds(y_true, y_prob, beta, num_thresh=200):
        """
        Scan thresholds between 0 and 1 to compute metrics for each cutoff.

        Returns:
            ths     : thresholds
            precs   : precision per threshold
            recs    : recall per threshold
            fbetas  : fbeta per threshold
        """
        y_true = np.asarray(y_true, dtype=int)
        ths = np.linspace(0.0, 1.0, num_thresh)

        precs, recs, fbetas = [], [], []
        for t in ths:
            y_pred = (y_prob >= t).astype(int)

            precs.append(precision_score(y_true, y_pred, zero_division=0))
            recs.append(recall_score(y_true, y_pred, zero_division=0))
            fbetas.append(
                fbeta_score(
                    y_true, y_pred, beta=beta, average="binary", zero_division=0
                )
            )

        return ths, np.array(precs), np.array(recs), np.array(fbetas)

    def _choose_threshold_recall(y_true, y_prob, recall_target=None):
        """
        Select threshold to maximize recall (or hit recall_target if provided).
        - recall_target: find the *lowest* threshold with recall >= target (tie-break by precision).
        - else: select threshold with highest recall (tie-break by precision).
        """
        ths, precs, recs, _ = _scan_thresholds(y_true, y_prob)
        if recall_target is not None:
            ok = np.where(recs >= float(recall_target))[0]
            if ok.size > 0:
                best_idx = ok[0]  # first threshold to hit recall target
                return (
                    float(ths[best_idx]),
                    float(recs[best_idx]),
                    float(precs[best_idx]),
                )
        # fallback: pick max recall
        max_r = np.max(recs)
        idx = np.where(recs == max_r)[0]
        best_idx = idx[np.argmax(precs[idx])] if idx.size > 1 else idx[0]
        return float(ths[best_idx]), float(recs[best_idx]), float(precs[best_idx])

    def _choose_threshold_fbeta(y_true, y_prob, beta):
        """Select threshold that maximizes Fβ (β>1 favors recall)."""
        ths, _, _, fbetas = _scan_thresholds(y_true, y_prob, beta=beta)
        best_idx = int(np.argmax(fbetas))
        return float(ths[best_idx]), float(fbetas[best_idx])

    # ----------------------------------------------------------------------
    # Step 6. Main training loop
    # ----------------------------------------------------------------------
    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0.0
        y_true_ep, y_prob_ep = [], []

        # --- Train for one epoch ---
        for batch in self.train_loader:
            batch = batch.to(self._device)
            optimizer.zero_grad()

            logits = self.subgraph_model.forward_from_data(batch)  # [B, 1]
            labels = batch.y.float().unsqueeze(1)  # [B, 1]

            loss = loss_fn(logits, labels)
            loss.backward()

            # Clip gradients to avoid exploding updates
            nn.utils.clip_grad_norm_(self.subgraph_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())

            # Collect predictions (at fixed threshold 0.5, just for logging)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            y_prob_ep.extend(probs.tolist())
            y_true_ep.extend(labels.detach().cpu().numpy().flatten().tolist())

        # Update learning rate
        scheduler.step()

        # Store average training loss
        avg_train_loss = epoch_loss / max(1, len(self.train_loader))
        train_loss_hist.append(avg_train_loss)

        # Quick training metrics (for logging only)
        y_pred_ep = (np.array(y_prob_ep) >= 0.5).astype(int)
        recall_train = recall_score(
            y_true_ep, y_pred_ep, average="binary", zero_division=0
        )
        f1_train = f1_score(y_true_ep, y_pred_ep, average="binary", zero_division=0)

        # --- Validation phase: calibration + early stopping ---
        this_val_metric, this_val_ap = None, None
        if use_val:
            self.subgraph_model.eval()
            vy_true, vy_prob = [], []
            with torch.no_grad():
                for vbatch in self.val_loader:
                    vbatch = vbatch.to(self._device)
                    vlogits = self.subgraph_model.forward_from_data(vbatch)
                    vprobs = torch.sigmoid(vlogits).cpu().numpy().flatten()
                    vy_prob.extend(vprobs.tolist())
                    vy_true.extend(vbatch.y.cpu().numpy().flatten().tolist())
            self.subgraph_model.train()

            vy_true = np.asarray(vy_true, dtype=int)
            vy_prob = np.asarray(vy_prob, dtype=float)

            # --- Threshold calibration ---
            if metric_mode == "recall":
                t_opt, val_recall, _ = _choose_threshold_recall(
                    vy_true, vy_prob, recall_target
                )
                this_val_metric = float(val_recall)
            else:  # "f_beta"
                t_opt, val_fbeta = _choose_threshold_fbeta(vy_true, vy_prob, beta)
                this_val_metric = float(val_fbeta)
            val_metric_hist.append(this_val_metric)

            # PR-AUC (probability-based)
            try:
                this_val_ap = float(average_precision_score(vy_true, vy_prob))
            except Exception:
                this_val_ap = None
            val_prauc_hist.append(this_val_ap if this_val_ap is not None else np.nan)

            # --- Early stopping ---
            if this_val_metric > best_val_metric + min_delta:
                best_val_metric = this_val_metric
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
                        f"[EarlyStop] No {metric_mode}_val improvement > {min_delta:.4f} "
                        f"for {patience} epochs. Stopping at epoch {epoch}."
                    )
                    break

        # --- Logging ---
        if epoch % 20 == 0 or epoch == total_epochs:
            lr_now = scheduler.get_last_lr()[0]
            if use_val:
                metric_name = (
                    "ValRecall*" if metric_mode == "recall" else f"ValF{beta:.1f}*"
                )
                ap_str = f"{this_val_ap:.4f}" if this_val_ap is not None else "n/a"
                print(
                    f"Epoch {epoch:03d} | "
                    f"TrainLoss: {avg_train_loss:.4f} | TrainRecall@0.5: {recall_train:.4f} | TrainF1@0.5: {f1_train:.4f} | "
                    f"{metric_name}: {this_val_metric:.4f} | ValPR-AUC: {ap_str} | "
                    f"LR: {lr_now:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"TrainLoss: {avg_train_loss:.4f} | TrainRecall@0.5: {recall_train:.4f} | TrainF1@0.5: {f1_train:.4f} | "
                    f"LR: {lr_now:.2e}"
                )

    # ----------------------------------------------------------------------
    # Step 7. Restore best checkpoint and persist best threshold.
    # ----------------------------------------------------------------------
    if use_val and best_state is not None:
        self.subgraph_model.load_state_dict(best_state)
        self.prediction_params["subgraph_classifier"][
            "prediction_threshold"
        ] = best_thresh
        tag = "Recall" if metric_mode == "recall" else f"F{beta:.1f}"
        print(
            f"[Best] Restored best subgraph model @ epoch {best_epoch} "
            f"({tag}_val={best_val_metric:.4f}, thr={best_thresh:.3f})"
        )

    # ----------------------------------------------------------------------
    # Step 8. Save learning curves (train loss, validation metric, PR-AUC).
    # ----------------------------------------------------------------------
    out_dir = self.dirs["output"]["prot_out_dir"]

    # Train loss curve
    plot_training_loss_curve(
        {"train_loss": train_loss_hist},
        output_path=os.path.join(
            out_dir,
            f"subgraph_classifier_train_loss_class_{self.target_class}.html",
        ),
    )

    # Validation metric curve (Recall or Fβ)
    if use_val and len(val_metric_hist) > 0:
        key = "val_recall" if metric_mode == "recall" else f"val_f{beta}"
        plot_training_loss_curve(
            {key: val_metric_hist},
            output_path=os.path.join(
                out_dir, f"subgraph_classifier_{key}_class_{self.target_class}.html"
            ),
        )

        # Validation PR-AUC curve
        if np.isfinite(np.nanmean(val_prauc_hist)):
            plot_training_loss_curve(
                {"val_pr_auc": val_prauc_hist},
                output_path=os.path.join(
                    out_dir,
                    f"subgraph_classifier_val_prauc_class_{self.target_class}.html",
                ),
            )

    if not use_val:
        print(
            "[Info] No validation loader found: training ran with a fixed threshold "
            f"{self.prediction_params['subgraph_classifier'].get('prediction_threshold', 0.5):.3f} and no early stopping."
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
                    (batch.node_labels == self.target_class).float().to(self._device)
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

        if epoch % 20 == 0 or epoch == params["epochs"]:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    # --- Plot training curve ---
    training_loss_curve_out_path = os.path.join(
        self.dirs["output"]["prot_out_dir"],
        f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
    )
    plot_training_loss_curve(
        {1: loss_history}, output_path=training_loss_curve_out_path
    )


def train_node_classifier_old_context(self):
    """
    Train the node-level GNN classifier (GNN2) on nodes within subgraphs.

    - Supports three modes in self.node_model.fusion_mode: 'none' | 'concat'
    * 'none'   : identical to the baseline (no context).
    * 'concat' : concatenates a per-node context vector (broadcasted subgraph embeddings from GNN1).

    Supervision:
    * 'anchor'     → supervise only the anchor node per subgraph (batched).
    * 'all_nodes'  → supervise all nodes in the subgraph (split-aware masking to avoid leakage).

    Requirements:
    - self.train_loader: DataLoader of subgraphs.
    - Each subgraph carries:
        . node_labels (class ids, 0..C-1)
        . (for 'anchor' mode) ego_center_local (local index of anchor)
        . (for split-aware) global_node_ids (indices into the full-graph train_mask)
    - self.subgraph_model: trained GNN1 (used only if fusion_mode != 'none').
    - self.node_model: GCNBiasNodeClassifier (configured with fusion_mode).
    """
    print(f"[!] Starting node-level training... | Device: {self._device}")

    # --- Config ---
    params = self.prediction_params
    node_params = params["node_classifier"]
    subg_gen_method = params.get("subg_gen_method", "color").lower()
    node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

    batch_size = node_params.get("batch_size", 16)
    epochs = node_params.get("epochs", 100)
    context_anneal = node_params.get(
        "context_anneal", "none"
    ).lower()  # 'none'|'linear'|'power'

    # --- Derive 'use_context': only use GNN1 if flag is on AND node_model supports bias ---
    fusion_mode = getattr(self.node_model, "fusion_mode", "concat")  # 'film'| 'concat'
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
            self.subgraph_model.eval()  # frozen provider of context

            ctx_dim = getattr(self.node_model, "context_dim", None)
            gnn1_dim = getattr(self.subgraph_model, "graph_emb_dim", None)
            if ctx_dim is not None and gnn1_dim is not None and ctx_dim != gnn1_dim:
                raise ValueError(
                    f"[ERROR] context_dim ({ctx_dim}) must match subgraph_model.graph_emb_dim ({gnn1_dim}) for 'concat'."
                )
            print(
                f"[INFO] Using subgraph_model context for node_model fusion_mode={fusion_mode} | context_norm={node_params.get('context_norm','none')}"
            )
    else:
        print("[INFO] Training node_model WITHOUT subgraph context/bias.")

    # --- Load train subgraphs from the pre-built train loader ---
    if self.train_loader is None:
        raise ValueError(
            "Train subgraph DataLoader (self.train_loader) not initialized."
        )
    subgraphs = list(self.train_loader.dataset)
    print(f"[INFO] Using all subgraphs for node training: {len(subgraphs)}")

    # --- DataLoader for node training ---
    loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=True)

    # --- Class imbalance handling: compute pos_weight on actual training pool ---
    pos_weight = self._compute_pos_weight_for_node_loss(
        subgraphs, self.target_class, node_labeling_mode
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(self._device) if pos_weight is not None else None,
        reduction="none",  # manual masking/averaging
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
    train_mask_global = self.data.train_mask.to(self._device)

    for epoch in range(1, epochs + 1):
        self.node_model.train()
        total_loss, correct, total = 0.0, 0, 0

        # annealing α in [0,1]
        if context_anneal == "linear":
            alpha = float(epoch) / float(max(1, epochs))
        elif context_anneal == "power":
            alpha = (float(epoch) / float(max(1, epochs))) ** 1.5
        else:
            alpha = 1.0

        # For FiLM, we can set model-side scaling (no grad)
        if (
            use_context
            and fusion_mode == "film"
            and hasattr(self.node_model, "set_context_scale")
        ):
            self.node_model.set_context_scale(alpha)

        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to(self._device)

            # ------------------------------------------------------------
            # Build per-node context from GNN1 (only if fusion_mode != 'none')
            # ------------------------------------------------------------
            ctx_nodes = None  # [N, d_ctx]    (for 'concat')

            # --------- Build context only if use_context ---------
            ctx_nodes = None
            if use_context:
                with torch.no_grad():
                    # 'concat'
                    Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                        batch
                    )  # [B,d]
                ctx_nodes = Z_s[batch.batch]  # [N,d]

                # Normalization:
                ctx_mode = node_params.get("context_norm", "concat")
                ctx_nodes = self._normalize_context_vectors(ctx_nodes, mode=ctx_mode)

                # anneal on the trainer side for concat (FiLM uses set_context_scale)
                if fusion_mode == "concat":
                    ctx_nodes = alpha * ctx_nodes

            # ------------------------------------------------------------
            # Forward GNN2 with/without bias
            # ------------------------------------------------------------
            if use_context:  # and fusion_mode == "concat":
                logits = self.node_model(
                    batch.x, batch.edge_index, ctx_nodes=ctx_nodes
                ).view(-1)
            else:
                logits = self.node_model(batch.x, batch.edge_index).view(-1)

            # ------------------------------------------------------------
            # Supervision: 'anchor' vs 'all_nodes'
            # ------------------------------------------------------------
            if node_labeling_mode == "anchor":
                centers_global = self._gather_anchor_indices_in_batch(
                    batch
                )  # [B_graphs]
                logit_center = logits[centers_global]  # [B_graphs]

                # Make labels binary according to target class
                y_center = (
                    batch.node_labels[centers_global] == self.target_class
                ).float()
                loss_vec = loss_fn(logit_center, y_center)
                loss = loss_vec.mean()

                with torch.no_grad():
                    preds = (torch.sigmoid(logit_center) >= 0.5).long()
                    correct += (preds == y_center.long()).sum().item()
                    total += y_center.numel()

            else:
                # all_nodes supervision with split-aware mask
                # Make labels binary according to target class
                y = (
                    (batch.node_labels == self.target_class).float().to(self._device)
                )  # [N]
                if not hasattr(batch, "global_node_ids"):
                    raise RuntimeError(
                        "Subgraphs must carry 'global_node_ids' for split-aware masking."
                    )
                # map global train_mask to current batch nodes [N] bool
                train_mask_sub = train_mask_global[batch.global_node_ids]  # [N] bool

                # compute loss per node [N]
                loss_per_node = loss_fn(logits, y)  # [N]

                # number of train nodes (avoid /0)
                denom = max(1, int(train_mask_sub.sum().item()))

                # average loss over train nodes only
                loss = (loss_per_node * train_mask_sub.float()).sum() / denom

                with torch.no_grad():
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    correct += ((preds == y.long()) & train_mask_sub).sum().item()
                    total += train_mask_sub.sum().item()

            # ------------------------------------------------------------
            # Optimize
            # ------------------------------------------------------------

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        acc = correct / max(1, total)
        loss_history.append(avg_loss)

        if epoch % 20 == 0 or epoch == node_params["epochs"]:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc(train-supervised): {acc:.4f} α={alpha:.2f}"
            )

    # --- Plot training curve ---
    training_loss_curve_out_path = os.path.join(
        self.dirs["output"]["prot_out_dir"],
        f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
    )
    plot_training_loss_curve(
        {1: loss_history}, output_path=training_loss_curve_out_path
    )


def train_node_classifier_f_beta(self):
    """
    Train the node-level GNN classifier (GNN2) with validation, early stopping,
    and *metric-driven* threshold calibration (default: precision).

    What this does:
    - Supervision modes:
        * 'anchor'     → only the anchor node per subgraph is supervised/evaluated
        * 'all_nodes'  → all nodes supervised, but restricted to the split mask
                        (train/test/val) to avoid leakage
    - Optional subgraph context (from GNN1) if fusion_mode in {'concat','film'}
    with trainer-side annealing (alpha in [0,1]).
    - Class imbalance handled by pos_weight in BCEWithLogitsLoss.
    - LR warmup + cosine decay (same helper used in GNN1).
    - Validation every epoch on self.val_loader:
        * Calibrate the *decision threshold* on validation **maximizing a chosen metric**.
        * Early stop when this metric does not improve for `patience` epochs.
    - Stores the calibrated threshold back into params["node_classifier"]["prediction_threshold"].

    Metric you optimize/calibrate:
    - Set via params["node_classifier"]["metric_mode"], one of:
        * "precision" (default) → favors high precision (few false positives)
        * "recall"              → favors high recall (few false negatives)
        * "f_beta"              → F_beta score; beta via params["node_classifier"]["f_beta"]
                                (beta<1 favors precision; beta>1 favors recall)
    Notes:
    - Loss remains BCEWithLogits (on TRAIN). The metric only drives early stopping
    and threshold calibration (on VAL).
    - When metric ties occur at multiple thresholds, we break ties by:
        1) higher recall, then 2) lower threshold (more conservative).
    """

    print(
        f"[!] Starting node-level training (metric-driven) ... | Device: {self._device}"
    )

    # -------------------------- Config & bookkeeping --------------------------
    params = self.prediction_params
    node_params = params["node_classifier"]

    # Supervision mode follows how subgraphs were built
    subg_gen_method = params.get("subg_gen_method", "color").lower()
    node_labeling_mode = "anchor" if subg_gen_method == "anchor" else "all_nodes"

    batch_size = int(node_params.get("batch_size", 16))
    epochs = int(node_params.get("epochs", 100))
    base_thresh = float(node_params.get("prediction_threshold", 0.5))
    fusion_mode = getattr(
        self.node_model, "fusion_mode", "concat"
    )  # 'none'|'concat'|'film'
    context_anneal = node_params.get(
        "context_anneal", "none"
    ).lower()  # 'none'|'linear'|'power'

    # ---- Metric selection (drives early stopping & threshold calibration) ----
    metric_mode = str(node_params.get("metric_mode", "precision")).lower()
    assert metric_mode in {
        "precision",
        "recall",
        "f_beta",
    }, "metric_mode must be 'precision', 'recall', or 'f_beta'."
    beta = float(node_params.get("f_beta", 0.5))  # only used for f_beta

    # Early stopping
    patience = int(node_params.get("early_stop_patience", 20))
    min_delta = float(
        node_params.get("early_stop_min_delta", 1e-4)
    )  # absolute delta on metric
    best_metric, best_epoch = -1.0, 0
    best_threshold = base_thresh
    epochs_no_improve = 0

    # -------------------------- Context usage (GNN1) --------------------------
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

    # -------------------------- Check loaders & masks --------------------------
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

    train_subgraphs = list(self.train_loader.dataset)
    print(f"[INFO] Using {len(train_subgraphs)} subgraphs for node training.")
    train_loader = DataLoader(train_subgraphs, batch_size=batch_size, shuffle=True)
    val_loader = self.val_loader  # could be None

    # Split masks on the *global* graph
    train_mask_global = self.data.train_mask.to(self._device)
    val_mask_global = getattr(self.data, "val_mask", None)
    if val_loader is not None and val_mask_global is None:
        raise RuntimeError("val_loader provided but self.data.val_mask is missing.")
    if val_mask_global is not None:
        val_mask_global = val_mask_global.to(self._device)

    # -------------------------- Loss & optimizer/scheduler --------------------------
    # Class imbalance: pos_weight computed on *train* pool respecting node_labeling_mode
    pos_weight = self._compute_pos_weight_for_node_loss(
        train_subgraphs, self.target_class, node_labeling_mode
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(self._device) if pos_weight is not None else None,
        reduction="none",  # manual masking/averaging
    )
    if pos_weight is not None:
        print(
            f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight={pos_weight.item():.3f}"
        )
    else:
        print(f"[INFO] node-level mode='{node_labeling_mode}' | pos_weight=None")

    optimizer = torch.optim.Adam(
        self.node_model.parameters(),
        lr=node_params["lr"],
        weight_decay=node_params["weight_decay"],
    )
    scheduler = self._make_warmup_cosine_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=None,  # default ~5% of epochs
        min_lr_scale=0.1,
    )

    # -------------------------- Histories for plots --------------------------
    train_loss_hist, val_loss_hist, val_metric_hist = [], [], []

    # -------------------------- Helpers --------------------------
    def _normalize_ctx(ctx):
        """Normalize/bias the context vector according to node_params['context_norm']."""
        mode = node_params.get("context_norm", "none")
        return self._normalize_context_vectors(ctx, mode=mode)

    def _build_context_nodes(batch, alpha: float):
        """
        Build per-node context vectors from GNN1, broadcast to nodes.
        Applies normalization and (for 'concat') trainer-side annealing by alpha.
        """
        if not use_context:
            return None
        with torch.no_grad():
            Z_s = self.subgraph_model.get_subgraph_embeddings_from_data(
                batch
            )  # [B_graphs, d]
        ctx_nodes = Z_s[batch.batch]  # [N, d]
        ctx_nodes = _normalize_ctx(ctx_nodes)
        return alpha * ctx_nodes if fusion_mode == "concat" else ctx_nodes

    def _calc_metric(y_true: np.ndarray, y_prob: np.ndarray):
        """
        Calibrate threshold on validation scores to maximize the chosen metric.
        Returns (best_threshold, best_metric_value).
        Tie-breaking: prefer higher recall, then lower threshold.
        """
        if y_true.size == 0:
            return base_thresh, 0.0

        # Candidate thresholds using precision_recall_curve (monotonic recall).
        # sklearn returns thresholds for points except first; we augment endpoints.
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        # Build candidate threshold list (include edges)
        cand_thr = np.unique(
            np.concatenate([thr, [y_prob.min() - 1e-8, y_prob.max() + 1e-8]])
        )

        best_t, best_val = base_thresh, -1.0
        best_recall, best_tie_thr = 0.0, None

        for t in cand_thr:
            y_pred = (y_prob >= t).astype(int)

            if metric_mode == "precision":
                m = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )[0]
            elif metric_mode == "recall":
                m = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )[1]
            else:  # 'f_beta'
                m = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

            # tie-breakers: prefer higher recall, then lower threshold
            r = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[1]
            if (m > best_val + 1e-12) or (
                abs(m - best_val) <= 1e-12
                and (
                    r > best_recall + 1e-12
                    or (
                        abs(r - best_recall) <= 1e-12
                        and (best_tie_thr is None or t < best_tie_thr)
                    )
                )
            ):
                best_val = m
                best_t = float(t)
                best_recall = float(r)
                best_tie_thr = float(t)

        return best_t, best_val

    def _epoch_pass(loader, split: str, eval_mode: bool, epoch_idx: int):
        """
        Single pass over a loader to compute:
        - average BCE loss over supervised nodes in this split
        - vectors (y_true, y_prob) for threshold calibration
        """
        if eval_mode:
            self.node_model.eval()
        else:
            self.node_model.train()

        # Pick proper split mask
        split_mask_global = train_mask_global if split == "train" else val_mask_global

        total_loss, denom_total = 0.0, 0
        y_true_all, y_prob_all = [], []

        with torch.set_grad_enabled(not eval_mode):
            for batch in loader:
                if not eval_mode:
                    optimizer.zero_grad()
                batch = batch.to(self._device)

                # Context annealing alpha (only matters in training)
                if context_anneal == "linear":
                    alpha = (
                        float(epoch_idx) / float(max(1, epochs))
                        if not eval_mode
                        else 1.0
                    )
                elif context_anneal == "power":
                    alpha = (
                        (float(epoch_idx) / float(max(1, epochs))) ** 1.5
                        if not eval_mode
                        else 1.0
                    )
                else:
                    alpha = 1.0

                # For FiLM: set scale directly on the model
                if (
                    use_context
                    and fusion_mode == "film"
                    and hasattr(self.node_model, "set_context_scale")
                ):
                    self.node_model.set_context_scale(alpha)

                ctx_nodes = _build_context_nodes(batch, alpha)

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
                    denom = y_center.numel()

                    probs = torch.sigmoid(logit_center).detach().cpu().numpy().flatten()
                    ybin = y_center.detach().cpu().numpy().astype(int).flatten()

                else:
                    # all_nodes with split-aware mask
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
                        loss = (loss_per_node * split_mask_sub.float()).sum() / denom
                        probs = (
                            torch.sigmoid(logits[split_mask_sub])
                            .detach()
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                        ybin = (
                            y[split_mask_sub]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                            .flatten()
                        )
                    else:
                        loss = logits.new_tensor(0.0)
                        probs = np.array([])
                        ybin = np.array([])

                # Optimize
                if not eval_mode:
                    loss.backward()
                    optimizer.step()

                # Accumulate
                total_loss += float(loss.item()) if denom > 0 else 0.0
                denom_total += max(1, denom)
                if probs.size > 0:
                    y_prob_all.extend(probs.tolist())
                    y_true_all.extend(ybin.tolist())

        avg_loss = total_loss / max(1, denom_total)
        return avg_loss, np.asarray(y_true_all, int), np.asarray(y_prob_all, float)

    # -------------------------- Training loop --------------------------
    for epoch in range(1, epochs + 1):
        # 1) Train epoch
        train_loss, _, _ = _epoch_pass(
            train_loader, split="train", eval_mode=False, epoch_idx=epoch
        )
        train_loss_hist.append(train_loss)

        # 2) Step LR
        scheduler.step()

        # 3) Validation: calibrate threshold & check early stop
        if val_loader is not None:
            with torch.no_grad():
                val_loss, vy_true, vy_prob = _epoch_pass(
                    val_loader, split="val", eval_mode=True, epoch_idx=epoch
                )
            val_loss_hist.append(val_loss)

            # Calibrate threshold by chosen metric
            t_opt, m_val = _calc_metric(vy_true, vy_prob)
            val_metric_hist.append(m_val)

            improved = (m_val - best_metric) > min_delta
            if improved:
                best_metric = m_val
                best_threshold = float(t_opt)
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Logging (every 20 epochs or last)
            if epoch % 20 == 0 or epoch == epochs:
                curr_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch:03d} | "
                    f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                    f"Val{metric_mode.upper()}*(best-thr): {m_val:.4f} (thr={t_opt:.3f}) | "
                    f"LR: {curr_lr:.2e}"
                )

            # Early stop
            if epochs_no_improve >= patience:
                print(
                    f"[EarlyStop] No val {metric_mode} improvement > {min_delta:.4f} for {patience} epochs. Stopping at epoch {epoch}."
                )
                break
        else:
            # No validation: log train only
            if epoch % 20 == 0 or epoch == epochs:
                curr_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch:03d} | TrainLoss: {train_loss:.4f} | LR: {curr_lr:.2e}"
                )

    # -------------------------- End of training: plots & threshold --------------------------
    # Plot train/val loss curves
    out_path = os.path.join(
        self.dirs["output"]["prot_out_dir"],
        f"node_classifier_training_loss_dashboard_class_{self.target_class}.html",
    )
    series = {"train_loss": train_loss_hist}
    if val_loader is not None and len(val_loss_hist) > 0:
        series["val_loss"] = val_loss_hist
    plot_training_loss_curve(series, output_path=out_path)

    # Plot validation metric curve (optional)
    if val_loader is not None and len(val_metric_hist) > 0:
        out_metric = os.path.join(
            self.dirs["output"]["prot_out_dir"],
            f"node_classifier_val_{metric_mode}_class_{self.target_class}.html",
        )
        plot_training_loss_curve(
            {f"val_{metric_mode}": val_metric_hist}, output_path=out_metric
        )

    # Persist calibrated threshold (if we had validation)
    if val_loader is not None and best_metric >= 0:
        self.prediction_params["node_classifier"][
            "prediction_threshold"
        ] = best_threshold
        print(
            f"[Calib] Best Val {metric_mode}={best_metric:.4f} at threshold={best_threshold:.3f} (epoch={best_epoch})"
        )
    else:
        print(
            f"[Calib] No validation available. Keeping configured threshold={base_thresh:.3f}."
        )
