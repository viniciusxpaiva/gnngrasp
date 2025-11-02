# -*- coding: utf-8 -*-
"""
Three side-by-side forest plots (mean ± CI) with per-axes inset zoom.
Datasets: Cora, CiteSeer, PubMed (keys in the input dict).
Matplotlib-only, publication-friendly (no fixed colors).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence, List, Tuple, Optional, Union
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

Number = Union[int, float, np.number]

# ---------- helpers ----------


def maybe_to_percent(values: Sequence[Number]) -> np.ndarray:
    """
    Convert to percentage if the data looks like probabilities in [0, 1.2].
    Otherwise, return as-is.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and (finite.min() >= 0.0) and (finite.max() <= 1.2):
        return arr * 100.0
    return arr


def mean_and_ci(
    data: Sequence[Number],
    confidence: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute mean and (lower, upper) confidence interval using Student's t.
    If n < 2, CI collapses to the mean.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    mean = float(np.mean(x))
    if n < 2:
        return mean, (mean, mean)
    sem = stats.sem(x)
    t_val = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    h = float(sem * t_val)
    return mean, (mean - h, mean + h)


def robust_zoom_limits(
    means: np.ndarray, lows: np.ndarray, highs: np.ndarray, pad: float = 0.25
) -> Tuple[float, float]:
    """
    Compute a zoom window that focuses on the central cluster of means:
    - Use interquartile range (Q1..Q3) of means,
    - expand by a robust CI half-width and a small pad.
    Falls back to overall min/max if needed.
    """
    valid = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
    if not np.any(valid):
        return 0.0, 1.0
    m = means[valid]
    l = lows[valid]
    h = highs[valid]

    q1, q3 = np.percentile(m, [25, 75])
    halfwidths = np.maximum(m - l, h - m)
    hw = np.percentile(halfwidths, 75)  # generous half-width
    if q3 - q1 < 1e-9:
        lo = q1 - max(hw, 0.2) - pad
        hi = q3 + max(hw, 0.2) + pad
    else:
        lo = q1 - hw - pad
        hi = q3 + hw + pad

    lo = float(min(lo, np.min(l) - pad))
    hi = float(max(hi, np.max(h) + pad))
    if hi - lo < 1e-6:  # degenerate
        lo, hi = float(np.min(l) - pad), float(np.max(h) + pad)
    return lo, hi


def nice_main_limits(
    lows: np.ndarray, highs: np.ndarray, margin: float = 0.8
) -> Tuple[float, float]:
    """
    Compute padded full-range x-limits for the main axis.
    """
    valid = np.isfinite(lows) & np.isfinite(highs)
    if not np.any(valid):
        return 0.0, 1.0
    lo = float(np.min(lows[valid]))
    hi = float(np.max(highs[valid]))
    return lo - margin, hi + margin


# ---------- core plotting (one axes + inset) ----------


def draw_forest_with_inset_on_ax(
    ax: plt.Axes,
    results_by_model: Dict[str, Sequence[Number]],
    metric_name: str = "Macro-F1",
    confidence: float = 0.95,
    include_models: Optional[List[str]] = None,
    main_xlim: Optional[Tuple[float, float]] = None,
    inset_frac: Tuple[str, str] = ("52%", "70%"),  # (width, height)
    inset_loc: str = "upper right",
) -> None:
    """
    Draw a horizontal forest plot (mean ± CI) with an inset zoom on a given axes.
    """
    # Filter/select models in a defined order if provided
    model_names = (
        include_models if include_models is not None else list(results_by_model.keys())
    )

    labels: List[str] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []

    for name in model_names:
        runs = maybe_to_percent(results_by_model.get(name, []))
        if len(runs) == 0 or not np.isfinite(runs).any():
            continue
        mean, (lo, hi) = mean_and_ci(runs, confidence=confidence)
        labels.append(name)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)

    if len(labels) == 0:
        ax.text(
            0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")
        return

    means = np.asarray(means, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)

    y = np.arange(len(labels))[::-1]  # top-to-bottom order
    xerr = np.vstack([means - lows, highs - means])

    # Main error bars (horizontal)
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=4, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{metric_name} (%)")
    if main_xlim is None:
        main_xlim = nice_main_limits(lows, highs, margin=0.8)
    ax.set_xlim(*main_xlim)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inset (zoomed on central cluster)
    ax_ins = inset_axes(
        ax, width=inset_frac[0], height=inset_frac[1], loc=inset_loc, borderpad=1.0
    )
    ax_ins.errorbar(means, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.2)
    zlo, zhi = robust_zoom_limits(means, lows, highs, pad=0.3)
    # clamp zoom to main limits to avoid going out of bounds
    zlo = max(zlo, main_xlim[0])
    zhi = min(zhi, main_xlim[1])
    if zhi - zlo < 1e-6:
        zlo, zhi = main_xlim
    ax_ins.set_xlim(zlo, zhi)
    ax_ins.set_yticks(y)
    ax_ins.set_yticklabels([])
    ax_ins.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.4)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)


# ---------- figure with 3 datasets side-by-side ----------


def forest_three_datasets_with_insets(
    results_by_dataset: Dict[str, Dict[str, Sequence[Number]]],
    metric_name: str = "Macro-F1",
    confidence: float = 0.95,
    include_models: Optional[
        List[str]
    ] = None,  # e.g. ["Hierarchical","Subgraphs only","NodeImport","BAT"]
    figsize: Tuple[float, float] = (15, 4.8),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Build a single figure with 3 horizontal forest plots (Cora, CiteSeer, PubMed),
    each with its own inset zoom.
    """
    # Enforce dataset order if keys exist; otherwise keep dict order
    ordered_keys = [
        k for k in ["Cora", "CiteSeer", "PubMed"] if k in results_by_dataset
    ]
    if len(ordered_keys) < len(results_by_dataset):
        # append any extra keys (if present) to the end
        for k in results_by_dataset:
            if k not in ordered_keys:
                ordered_keys.append(k)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    # If fewer than 3 datasets provided, hide unused axes
    for i in range(3):
        if i >= len(ordered_keys):
            axes[i].axis("off")

    for ax, ds in zip(axes, ordered_keys[:3]):
        draw_forest_with_inset_on_ax(
            ax=ax,
            results_by_model=results_by_dataset[ds],
            metric_name=metric_name,
            confidence=confidence,
            include_models=include_models,
        )
        ax.set_title(ds, fontsize=13)

    fig.suptitle(f"{metric_name} — mean ± CI (with zoomed inset)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


# ---------------------- example usage ----------------------

if __name__ == "__main__":
    # Use your real dict here:
    results_macro_f1 = {
        "Cora": {
            "Hierarchical": [
                87.24,
                88.11,
                86.15,
                87.12,
                87.09,
                88.81,
                87.87,
                87.62,
                87.39,
                87.87,
            ],
            "Subgraphs only": [
                84.23,
                85.64,
                84.57,
                85.64,
                85.45,
                84.46,
                85.53,
                85.45,
                85.94,
                85.24,
            ],
            "NodeImport": [
                0.8351425414641892,
                0.8744059171045997,
                0.8619940846571602,
                0.8584345676153511,
                0.821783731512691,
                0.8386322643147068,
                0.8382411530135349,
                0.8592800041859779,
                0.8406444065339757,
                0.838122464348244,
            ],
            "BAT": [
                64.81826053,
                69.18665867,
                70.53711426,
                68.85954109,
                73.58171857,
                69.02468334,
                70.40725392,
                65.78924477,
                71.84020957,
                66.07684155,
            ],
        },
        "CiteSeer": {
            "Hierarchical": [
                78.45,
                78.06,
                78.63,
                78.53,
                77.71,
                76.84,
                79.05,
                78.42,
                77.82,
                77.06,
            ],
            "Subgraphs only": [
                71.27,
                72.39,
                71.31,
                71.87,
                72.49,
                73.92,
                72.57,
                72.50,
                72.29,
                72.46,
            ],
            "NodeImport": [
                0.6922092513894373,
                0.705947898203651,
                0.6566785622082861,
                0.6733961678993466,
                0.7105117901574558,
                0.6687200474345248,
                0.6846156872305232,
                0.689823846701421,
                0.6925689213335562,
                0.6756914289962893,
            ],
            "BAT": [
                59.45900128,
                49.08970274,
                58.58315751,
                35.58404173,
                60.47078445,
                59.08028269,
                49.20770367,
                59.09176226,
                59.36177833,
                57.0003025,
            ],
        },
        "PubMed": {
            "Hierarchical": [
                88.47,
                87.94,
                88.61,
                88.59,
                88.15,
                88.47,
                88.10,
                88.21,
                87.54,
                88.31,
            ],
            "Subgraphs only": [
                85.71,
                85.30,
                84.97,
                85.71,
                84.03,
                85.85,
                85.29,
                85.30,
                85.39,
                85.24,
            ],
            "NodeImport": [
                0.8465725107150884,
                0.8461481096973711,
                0.8329895609052688,
                0.8240130324249174,
                0.8413346117522357,
                0.843407848698447,
                0.8409544529145325,
                0.836274046918318,
                0.8353134719275674,
                0.8331439266302473,
            ],
            "BAT": [
                54.34398997,
                63.31925991,
                65.19744237,
                74.97331617,
                49.04786532,
                66.79352762,
                68.60498801,
                67.99978363,
                60.46520323,
                50.68104247,
            ],
        },
    }

    # Choose a consistent ordering (optional)
    order = ["Hierarchical", "Subgraphs only", "NodeImport", "BAT"]

    # Single figure with 3 panels (Cora, CiteSeer, PubMed), each with its own inset zoom
    forest_three_datasets_with_insets(
        results_by_dataset=results_macro_f1,
        metric_name="Macro-F1",
        confidence=0.95,
        include_models=order,  # or None to use dict order
        figsize=(15, 4.8),
        save_path="macro_f1_forest_3panels.png",
    )
