# -*- coding: utf-8 -*-
"""
Three side-by-side bar charts (mean) with confidence-interval error bars.
Datasets: Cora, CiteSeer, PubMed (keys in the input dict).

- Matplotlib-only (no seaborn), no explicit colors set.
- Independent Y axes per dataset (better readability).
- Converts probabilities in [0, 1.2] to percentages automatically.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence, List, Tuple, Optional, Union
from scipy import stats

Number = Union[int, float, np.number]

# ---------- helpers ----------


def maybe_to_percent(values: Sequence[Number]) -> np.ndarray:
    """
    Convert to percentage if the data looks like probabilities in [0, 1.2].
    Otherwise, return values as-is.
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
) -> Tuple[float, float, float]:
    """
    Compute mean and (lower, upper) confidence interval using Student's t.
    If n < 2, the CI collapses to the mean.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if n < 2:
        return mean, mean, mean
    sem = stats.sem(x)
    t_val = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    h = float(sem * t_val)
    return mean, mean - h, mean + h


# ---------- bars with CI for three datasets ----------


def plot_datasets_bars_with_ci(
    results_by_dataset: Dict[str, Dict[str, Sequence[Number]]],
    metric_name: str,
    unit: str = "%",
    confidence: float = 0.95,  # e.g., 0.90, 0.95, 0.99
    bar_width: float = 0.6,  # width of each bar
    group_spacing: float = 0.6,  # padding on left/right of the axis
    cap_size: float = 6.0,
    figsize: Tuple[float, float] = (14, 6),
    save_basename: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Build one figure with 3 subplots (Cora, CiteSeer, PubMed).
    Each subplot shows a grouped bar chart where:
      - bar height = mean
      - vertical error bar = confidence interval around the mean
    The CI is drawn centered on the bar (above/below the mean).
    """
    # Keep a stable dataset order if the typical keys exist
    ordered_keys = [
        k for k in ["Cora", "CiteSeer", "PubMed"] if k in results_by_dataset
    ]
    # Append any extra keys if present
    for k in results_by_dataset:
        if k not in ordered_keys:
            ordered_keys.append(k)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    # Hide unused axes if fewer datasets were provided
    for i in range(3):
        if i >= len(ordered_keys):
            axes[i].axis("off")

    for ax, dataset in zip(axes, ordered_keys[:3]):
        models = results_by_dataset[dataset]

        # Prepare labels and statistics (skip empty arrays)
        labels: List[str] = []
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []

        for model_name, runs in models.items():
            vals = maybe_to_percent(runs)
            if len(vals) == 0 or not np.isfinite(vals).any():
                continue
            m, lo, hi = mean_and_ci(vals, confidence=confidence)
            labels.append(model_name)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        if len(labels) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        means = np.asarray(means, dtype=float)
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)

        okabe_ito = {
            "black": "#000000",
            "orange": "#E69F00",
            "skyblue": "#56B4E9",
            "bluishgreen": "#009E73",
            "yellow": "#F0E442",
            "blue": "#0072B2",
            "vermillion": "#D55E00",
            "reddishpurple": "#CC79A7",
        }

        color_map = {
            "CLARA": okabe_ito["blue"],  # azul forte
            "CLARA-S": okabe_ito["skyblue"],  # laranja-avermelhado
            "NodeImport": okabe_ito["bluishgreen"],  # verde-azulado
            "GraphSHA": okabe_ito["skyblue"],  # azul claro
            "BAT": okabe_ito["orange"],  # roxo
            "GraphSHA": okabe_ito["vermillion"],  # laranja
        }

        ax.axvline(
            x=1.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7, zorder=0
        )

        # X positions for groups
        x = np.arange(len(labels), dtype=float)

        # Bars (height = mean)
        bars = ax.bar(
            x,
            means,
            width=bar_width,
            zorder=2,
            color=[color_map.get(lbl, "#7f7f7f") for lbl in labels],
            edgecolor="black",  # borda preta para mais contraste
            linewidth=0.8,
        )

        # Error bars (CI centered on the bar)
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=cap_size,
            linewidth=1.5,
            zorder=3,
        )

        # Cosmetics
        ax.set_title(dataset, fontsize=20, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=45, rotation_mode="anchor", ha="right", fontsize=18
        )
        ax.set_ylabel(f"{metric_name} ({unit})" if ax is axes[0] else "", fontsize=18)
        ax.tick_params(axis="y", labelsize=14)

        if dataset == "Cora":
            ax.set_ylim(65, 90)

        elif dataset == "CiteSeer":
            ax.set_ylim(50, 80)
        else:
            ax.set_ylim(55, 90)

        if len(labels) >= 3:
            ax.axvline(
                x=1.5,
                color="black",
                linestyle="dashed",
                linewidth=1.2,
                alpha=0.7,
                zorder=0,
            )

        # Expand x-limits to avoid clipping bar edges and caps
        ax.set_xlim(-group_spacing, (len(labels) - 1) + group_spacing)

        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    conf_txt = f"{int(confidence*100)}%"
    # fig.suptitle(f"{metric_name} — mean (bars) ± CI ({conf_txt})", fontsize=14, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_basename:
        plt.savefig(f"{save_basename}.png", dpi=dpi, bbox_inches="tight")
        plt.savefig(f"{save_basename}.pdf", dpi=dpi, bbox_inches="tight")

    plt.show()

    """
    "GraphSHA-Cora": [
        0.479482,
        0.396425,
        0.407255,
        0.455903,
        0.485836,
        0.509743,
        0.491936,
        0.489863,
        0.526652,
        0.453320,
    ],"""


# ---------- example usage ----------

if __name__ == "__main__":
    # Replace below with your real runs (lists of N results per model)
    results_macro_f1 = {
        "Cora": {
            "CLARA": [
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
            "CLARA-S": [
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
            "GraphSHA": [
                83.61,
                84.26,
                83.92,
                84.49,
                84.37,
                83.70,
                84.82,
                84.17,
                84.17,
                84.26,
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
            "CLARA": [
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
            "CLARA-S": [
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
            "GraphSHA": [
                70.07,
                70.49,
                70.07,
                71.03,
                69.79,
                70.68,
                69.72,
                70.88,
                69.84,
                70.69,
            ],
            "BAT": [
                59.45900128,
                49.08970274,
                58.58315751,
                55.58404173,
                60.47078445,
                59.08028269,
                49.20770367,
                59.09176226,
                59.36177833,
                57.0003025,
            ],
        },
        "PubMed": {
            "CLARA": [
                88.09,
                88.56,
                88.58,
                88.39,
                88.36,
                88.34,
                88.20,
                88.37,
                87.86,
                88.48,
            ],
            "CLARA-S": [
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
            "GraphSHA": [
                82.51,
                83.70,
                83.45,
                83.03,
                83.35,
                83.27,
                83.22,
                83.26,
                83.82,
                83.65,
            ],
            "BAT": [
                54.34398997,
                63.31925991,
                65.19744237,
                74.97331617,
                59.04786532,
                66.79352762,
                68.60498801,
                67.99978363,
                60.46520323,
                50.68104247,
            ],
        },
    }

    # 1) Boxplots with distribution (no p-values)

    # 2) Mean with CI error bars (pick one or several confidence levels)
    plot_datasets_bars_with_ci(
        results_by_dataset=results_macro_f1,
        metric_name="Macro F1",
        unit="%",
        confidence=0.99,  # or (0.90, 0.95, 0.99)
        figsize=(14, 6),
        save_basename="graphics/planetoid",
    )
