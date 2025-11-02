import copy
import math
import time
import numpy as np
from scipy import stats


def mean_confidence_interval(data, confidence=0.95):
    """Return (mean, lower, upper) for the given list of runs."""
    a = np.array(data, dtype=float)
    n = len(a)
    if n < 2:
        return float(np.mean(a)), float(np.mean(a)), float(np.mean(a))
    mean = np.mean(a)
    sem = stats.sem(a, nan_policy="omit")  # erro padrão
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)  # margem
    return mean, mean - h, mean + h


def pct(x: float) -> float:
    """Convert 0..1 to %, keep >1 as-is; handle NaN."""
    try:
        return x * 100.0 if (not math.isnan(x) and x <= 1.0) else x
    except TypeError:
        return x


def run_planetoid_grid_experiments(
    pipeline,
    hier_params,
    n_base_runs,
    dataset_name,
    GNN1_TYPES,
    GNN1_LAYERS,
    SG_NEIGHBORS,
    GNN2_TYPES,
    GNN2_LAYERS,
    POOL_TYPE,
):
    """
    Grid-search over:
      - node_classifier.gnn_type   ∈ {GCN, GAT, SAGE, ...}
      - node_classifier.num_layers ∈ {1, 2, 3}
      - subg_neighbor_layers       ∈ {2, 3, 4, ...}

    For each configuration, run N times and aggregate metrics for:
      • GNN1 (subgraph classifier): accuracy / balanced_accuracy / precision / recall / f1 / mcc / roc_auc / pr_auc
      • GNN2 (node classifier):     accuracy / balanced_accuracy / precision / recall / f1 / macro_f1 / mcc / roc_auc / pr_auc

    Notes:
      - We keep backward compatibility: if the pipeline returns only old node metrics
        (balanced_accuracy, macro_f1), other node metrics are filled with NaN so the
        averaging (nanmean) still works.
      - AUCs may be undefined in degenerate splits; we treat them as NaN and ignore
        them in means (nanmean).
    """

    center_policy = hier_params["subg_gen_method"]
    use_sub_classifier_flag = hier_params["use_subgraph_classifier"]
    fusion_mode = hier_params["node_classifier"].get("fusion_mode", "none")
    context_norm = hier_params["node_classifier"].get("context_norm", "none")
    use_sub_classifier = (
        "subT" if use_sub_classifier_flag and fusion_mode != "none" else "subF"
    )
    bias_name = (
        f"_bT{context_norm[:3]}"
        if (use_sub_classifier_flag and fusion_mode != "none")
        else ""
    )

    # Helper: canonical metric keys for each head
    subg_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    node_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    global_summary = []
    test_num = 1

    for gnn1_type in GNN1_TYPES:
        for gnn1_layer in GNN1_LAYERS:
            for pool_type in POOL_TYPE:
                for gnn2_type in GNN2_TYPES:
                    for gnn2_layer in GNN2_LAYERS:
                        for k_neighbors in SG_NEIGHBORS:
                            print("\n" + "=" * 70)
                            print(
                                f"[CFG] gnn1_type={gnn1_type} | gnn1_layers={gnn1_layer} | "
                                f"pool={pool_type} | k={k_neighbors} | "
                                f"gnn2_type={gnn2_type} | gnn2_layers={gnn2_layer}"
                            )
                            print("=" * 70)

                            # Clone base params and inject the grid config
                            H = copy.deepcopy(hier_params)
                            H["subgraph_classifier"]["gnn_type"] = gnn1_type
                            H["subgraph_classifier"]["num_layers"] = gnn1_layer
                            H["subgraph_classifier"]["pool_type"] = pool_type
                            H["subg_neighbor_layers"] = k_neighbors

                            H["node_classifier"]["gnn_type"] = gnn2_type
                            H["node_classifier"]["num_layers"] = gnn2_layer

                            # --- Accumulators across runs for this config ---
                            runs_node_metrics = {k: [] for k in node_keys}
                            runs_subg_metrics = {k: [] for k in subg_keys}
                            run_times = []

                            # --- Repeat N times for stability ---
                            for i in range(n_base_runs):
                                total_start = time.time()
                                print(
                                    f"\n*** BASELINE PREDICTION RUN {i+1}/{n_base_runs} | {dataset_name} Dataset ***"
                                )
                                result = pipeline.baseline_prediction(H, dataset_name)

                                # --- Parse node summary robustly (old/new formats) ---
                                if "node_summary" in result:
                                    node_summary = result["node_summary"]
                                else:
                                    # very old format: {'balanced_accuracy': float, 'macro_f1': float}
                                    node_summary = {
                                        "accuracy": np.nan,
                                        "balanced_accuracy": float(
                                            result.get("balanced_accuracy", np.nan)
                                        ),
                                        "precision": np.nan,
                                        "recall": np.nan,
                                        "f1": np.nan,
                                        "macro_f1": float(
                                            result.get("macro_f1", np.nan)
                                        ),
                                        "mcc": np.nan,
                                        "roc_auc": np.nan,
                                        "pr_auc": np.nan,
                                    }

                                # --- Parse subgraph summary robustly (may be missing) ---
                                if "subgraph_summary" in result:
                                    subg_summary = result["subgraph_summary"]
                                else:
                                    subg_summary = {k: np.nan for k in subg_keys}

                                # Accumulate node metrics
                                for k in node_keys:
                                    v = node_summary.get(k, np.nan)
                                    runs_node_metrics[k].append(
                                        np.nan if v is None else float(v)
                                    )

                                # Accumulate subgraph metrics
                                for k in subg_keys:
                                    v = subg_summary.get(k, np.nan)
                                    runs_subg_metrics[k].append(
                                        np.nan if v is None else float(v)
                                    )

                                run_times.append(time.time() - total_start)

                            # --- Averages/stdevs over runs (per config) ---
                            print(runs_node_metrics)
                            node_means = {
                                k: float(np.nanmean(vs)) if len(vs) > 0 else 0.0
                                for k, vs in runs_node_metrics.items()
                            }
                            node_stds = {
                                k: float(np.nanstd(vs)) if len(vs) > 0 else 0.0
                                for k, vs in runs_node_metrics.items()
                            }
                            subg_means = {
                                k: float(np.nanmean(vs)) if len(vs) > 0 else 0.0
                                for k, vs in runs_subg_metrics.items()
                            }
                            subg_stds = {
                                k: float(np.nanstd(vs)) if len(vs) > 0 else 0.0
                                for k, vs in runs_subg_metrics.items()
                            }

                            metrics_ci = {}
                            for m in ["balanced_accuracy", "macro_f1"]:
                                if (
                                    m in runs_node_metrics
                                    and len(runs_node_metrics[m]) > 0
                                ):
                                    mean, low, high = mean_confidence_interval(
                                        runs_node_metrics[m]
                                    )
                                    metrics_ci[m] = (mean, low, high)

                            avg_time = float(np.mean(run_times)) if run_times else 0.0
                            total_time = float(np.sum(run_times))

                            # --- Per-config console summary (BOTH models) ---
                            print(
                                f"\n[✓] {dataset_name.lower()}_{center_policy}_{use_sub_classifier}_{gnn1_type.lower()} | {n_base_runs} runs"
                            )
                            print(
                                f"GNN2 type: {gnn2_type.lower()} | Layers: {gnn2_layer} | SG neighbors: {k_neighbors}"
                            )

                            print("   --- Node (GNN2) metrics (means ± 95% CI) ---")
                            print(
                                f"   bacc    : {pct(metrics_ci['balanced_accuracy'][0]):5.2f} "
                                f"[{pct(metrics_ci['balanced_accuracy'][1]):5.2f}, {pct(metrics_ci['balanced_accuracy'][2]):5.2f}]"
                            )
                            print(
                                f"   macro_f1: {pct(metrics_ci['macro_f1'][0]):5.2f} "
                                f"[{pct(metrics_ci['macro_f1'][1]):5.2f}, {pct(metrics_ci['macro_f1'][2]):5.2f}]"
                            )

                            print(
                                "   --- Node (GNN2) metrics (means ± std over runs) ---"
                            )

                            print(
                                f"   acc     : {pct(node_means['accuracy']):5.2f} ± {pct(node_stds['accuracy']):5.2f}"
                            )
                            print(
                                f"   bacc    : {pct(node_means['balanced_accuracy']):5.2f} ± {pct(node_stds['balanced_accuracy']):5.2f}"
                            )
                            print(
                                f"   prec    : {pct(node_means['precision']):5.2f} ± {pct(node_stds['precision']):5.2f}"
                            )
                            print(
                                f"   rec     : {pct(node_means['recall']):5.2f} ± {pct(node_stds['recall']):5.2f}"
                            )
                            print(
                                f"   f1      : {pct(node_means['f1']):5.2f} ± {pct(node_stds['f1']):5.2f}"
                            )
                            print(
                                f"   macro_f1: {pct(node_means['macro_f1']):5.2f} ± {pct(node_stds['macro_f1']):5.2f}"
                            )
                            print(
                                f"   mcc     : {node_means['mcc']:7.4f} ± {node_stds['mcc']:7.4f}"
                            )
                            print(
                                f"   roc_auc : {node_means['roc_auc']:7.4f} ± {node_stds['roc_auc']:7.4f}"
                            )
                            print(
                                f"   pr_auc  : {node_means['pr_auc']:7.4f} ± {node_stds['pr_auc']:7.4f}"
                            )

                            print(
                                "   --- Subgraph (GNN1) metrics (OvR means ± std) ---"
                            )
                            print(
                                f"   acc     : {subg_means['accuracy']*100:5.2f} ± {subg_stds['accuracy']*100:5.2f}"
                            )
                            print(
                                f"   bacc    : {subg_means['balanced_accuracy']*100:5.2f} ± {subg_stds['balanced_accuracy']*100:5.2f}"
                            )
                            print(
                                f"   prec    : {subg_means['precision']*100:5.2f} ± {subg_stds['precision']*100:5.2f}"
                            )
                            print(
                                f"   rec     : {subg_means['recall']*100:5.2f} ± {subg_stds['recall']*100:5.2f}"
                            )
                            print(
                                f"   f1      : {subg_means['f1']*100:5.2f} ± {subg_stds['f1']*100:5.2f}"
                            )
                            print(
                                f"   mcc     : {subg_means['mcc']:7.4f} ± {subg_stds['mcc']:7.4f}"
                            )
                            print(
                                f"   roc_auc : {subg_means['roc_auc']:7.4f} ± {subg_stds['roc_auc']:7.4f}"
                            )
                            print(
                                f"   pr_auc  : {subg_means['pr_auc']:7.4f} ± {subg_stds['pr_auc']:7.4f}"
                            )

                            print(
                                f"[⏱️] Average time per run: {avg_time:.2f} sec | Total: {total_time:.2f} sec"
                            )

                            print("Macro f1: ", end="")
                            for res in runs_node_metrics["macro_f1"]:
                                print(f"{res*100:5.2f}", end=" | ")

                            # --- Persist config results (append mode) ---
                            test_name = f"{dataset_name.lower()}_{center_policy}_{gnn1_type.lower()}{gnn1_layer}_{use_sub_classifier}_{gnn2_type.lower()}{gnn2_layer}"
                            test_name = (
                                f"{dataset_name}_teste"  # keeping your current override
                            )
                            results_filename = f"results_{test_name}.txt"
                            with open(results_filename, "a") as f:
                                f.write(f"Test {test_num}\n")
                                test_num += 1
                                gnn1_params = (
                                    f"GNN1 type: {H['subgraph_classifier']['gnn_type']} | "
                                    f"Layers: {H['subgraph_classifier']['num_layers']} | "
                                    f"{H['subgraph_classifier']['pool_type']} pooling | "
                                    f"Dropout: {H['subgraph_classifier']['dropout']} | "
                                    f"Epochs: {H['subgraph_classifier']['epochs']} | "
                                    f"LR: {H['subgraph_classifier']['lr']} | "
                                    f"{H['subgraph_classifier']['metric_mode']} | "
                                    f"{H['subgraph_classifier']['beta']} | "
                                )
                                gnn2_params = (
                                    f"GNN2 type: {H['node_classifier']['gnn_type']} | "
                                    f"Layers: {H['node_classifier']['num_layers']} | "
                                    f"SG neighbors: {k_neighbors}"
                                )
                                bias_params = (
                                    f" | fusion_mode={fusion_mode} | context_norm={context_norm}"
                                    if use_sub_classifier_flag and fusion_mode != "none"
                                    else ""
                                )
                                f.write(f"[✓] {test_name} | {n_base_runs} runs\n")
                                f.write(f"{gnn1_params}\n")
                                f.write(f"{gnn2_params}{bias_params}\n")

                                f.write(
                                    "   --- Node (GNN2) metrics (means ± 95% CI) ---\n"
                                )
                                f.write("   Macro f1: ")
                                for res in runs_node_metrics["macro_f1"]:
                                    f.write(f"{res*100:5.2f} | ")
                                f.write(
                                    f"\n   bacc    : {pct(metrics_ci['balanced_accuracy'][0]):5.2f} "
                                    f"[{pct(metrics_ci['balanced_accuracy'][1]):5.2f}, {pct(metrics_ci['balanced_accuracy'][2]):5.2f}]\n"
                                )
                                f.write(
                                    f"   macro_f1: {pct(metrics_ci['macro_f1'][0]):5.2f} "
                                    f"[{pct(metrics_ci['macro_f1'][1]):5.2f}, {pct(metrics_ci['macro_f1'][2]):5.2f}]\n"
                                )

                                """
                                # Node (GNN2)
                                f.write(
                                    "   --- Node (GNN2) metrics (means ± std) ---\n"
                                )
                                f.write(
                                    f"   acc     : {pct(node_means['accuracy']):5.2f} ± {pct(node_stds['accuracy']):5.2f}\n"
                                )
                                f.write(
                                    f"   bacc    : {pct(node_means['balanced_accuracy']):5.2f} ± {pct(node_stds['balanced_accuracy']):5.2f}\n"
                                )
                                f.write(
                                    f"   prec    : {pct(node_means['precision']):5.2f} ± {pct(node_stds['precision']):5.2f}\n"
                                )
                                f.write(
                                    f"   rec     : {pct(node_means['recall']):5.2f} ± {pct(node_stds['recall']):5.2f}\n"
                                )
                                f.write(
                                    f"   f1      : {pct(node_means['f1']):5.2f} ± {pct(node_stds['f1']):5.2f}\n"
                                )
                                f.write(
                                    f"   macro_f1: {pct(node_means['macro_f1']):5.2f} ± {pct(node_stds['macro_f1']):5.2f}\n"
                                )
                                f.write(
                                    f"   mcc     : {node_means['mcc']:7.4f} ± {node_stds['mcc']:7.4f}\n"
                                )
                                f.write(
                                    f"   roc_auc : {node_means['roc_auc']:7.4f} ± {node_stds['roc_auc']:7.4f}\n"
                                )
                                f.write(
                                    f"   pr_auc  : {node_means['pr_auc']:7.4f} ± {node_stds['pr_auc']:7.4f}\n"
                                )

                                # Subgraph (GNN1)
                                f.write(
                                    "   --- Subgraph (GNN1) metrics (OvR means ± std) ---\n"
                                )
                                f.write(
                                    f"   acc     : {subg_means['accuracy']*100:5.2f} ± {subg_stds['accuracy']*100:5.2f}\n"
                                )
                                f.write(
                                    f"   bacc    : {subg_means['balanced_accuracy']*100:5.2f} ± {subg_stds['balanced_accuracy']*100:5.2f}\n"
                                )
                                f.write(
                                    f"   prec    : {subg_means['precision']*100:5.2f} ± {subg_stds['precision']*100:5.2f}\n"
                                )
                                f.write(
                                    f"   rec     : {subg_means['recall']*100:5.2f} ± {subg_stds['recall']*100:5.2f}\n"
                                )
                                f.write(
                                    f"   f1      : {subg_means['f1']*100:5.2f} ± {subg_stds['f1']*100:5.2f}\n"
                                )
                                f.write(
                                    f"   mcc     : {subg_means['mcc']:7.4f} ± {subg_stds['mcc']:7.4f}\n"
                                )
                                f.write(
                                    f"   roc_auc : {subg_means['roc_auc']:7.4f} ± {subg_stds['roc_auc']:7.4f}\n"
                                )
                                f.write(
                                    f"   pr_auc  : {subg_means['pr_auc']:7.4f} ± {subg_stds['pr_auc']:7.4f}\n"
                                )
                                f.write(
                                    f"[⏱️] Average time per run: {avg_time:.2f} sec | Total: {total_time:.2f} sec\n\n"
                                )
                                """

                            # --- Row for global ranking (prefer macro_f1 if available; else f1) ---
                            rank_f1 = node_means["macro_f1"]
                            if np.isnan(rank_f1):
                                rank_f1 = node_means["f1"]

                            global_summary.append(
                                {
                                    "gnn1_type": gnn1_type,
                                    "gnn1_layer": gnn1_layer,
                                    "k": k_neighbors,
                                    "gnn2_type": gnn2_type,
                                    "gnn2_layer": gnn2_layer,
                                    "bAcc_mean": pct(node_means["balanced_accuracy"]),
                                    "bAcc_std": pct(node_stds["balanced_accuracy"]),
                                    "f1_mean": pct(rank_f1),
                                    "f1_std": pct(
                                        node_stds["macro_f1"]
                                        if not np.isnan(node_stds["macro_f1"])
                                        else node_stds["f1"]
                                    ),
                                    "avg_time": avg_time,
                                    # extras for quick reference
                                    "subg_f1_mean": subg_means["f1"],
                                    "subg_bacc_mean": subg_means["balanced_accuracy"],
                                }
                            )

    # ---- Global summary table (ranked by NODE F1 / Macro-F1) ----
    print("\n" + "=" * 70)
    print("[✓] GRID SUMMARY (top-10 by NODE Macro F1):")
    print("=" * 70)
    top = sorted(global_summary, key=lambda d: d["f1_mean"], reverse=True)[:10]
    for row in top:
        print(
            f"{row['gnn1_type']:<4} | L={row['gnn1_layer']} | K={row['k']} | "
            f"{row['gnn2_type']:<4} | L={row['gnn2_layer']} | "
            f"bAcc={row['bAcc_mean']:.2f}±{row['bAcc_std']:.2f} | "
            f"F1={row['f1_mean']:.2f}±{row['f1_std']:.2f} | "
            f"{row['avg_time']:.2f}s/run | "
            f"SubG F1={row['subg_f1_mean']*100:5.1f} | SubG bAcc={row['subg_bacc_mean']*100:5.1f}"
        )
