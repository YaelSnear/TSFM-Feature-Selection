"""Publication-quality plots for EXP_metr_la_lag_2h_20260525_170421.

Figures:
  1. rmse_highlight.png      — RMSE vs K, highlighting Lagged_CKA_raw_L10
  2. sensor_matrix.png       — K=5 dot matrix: which sensors each method selected
  3. layer_ablation.png      — K=5 bar chart: Lagged_CKA_raw across layers 6/8/10
  4. sota_best_tsfm.png      — Grouped bar chart identical to plot_sota_comparison_bar,
                               but each "Best TSFM" bar is annotated with its config name

Usage:
    python scripts/plot_publication.py
    python scripts/plot_publication.py --out_dir <path>
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)

OUT_DIR = "outputs/EXP_metr_la_lag_2h_20260525_170421"

# ── Hardcoded results (parsed from results.csv) ──────────────────────────────

KS = [5, 10, 20, 30]

RMSE = {
    "Univariate":            [9.8409, 9.8409, 9.8409, 9.8409],
    "Geographic":            [9.3628, 9.1821, 9.2344, 9.6667],
    "Lasso":                 [9.2740, 9.3056, 9.3098, 9.1673],
    "Pearson":               [9.2884, 9.2736, 9.3866, 9.6949],
    "RF":                    [9.9393, 9.9701, 9.5906, 9.5519],
    "Lagged_CKA_raw_L10":   [8.8863, 9.3949, 9.3813, 9.3981],
    "Mean_Pooling_raw_L6":   [9.2626, 9.2497, 9.2521, 9.1711],
    "Mean_Pooling_raw_L8":   [9.2340, 9.4737, 9.0222, 9.3359],
    "Mean_Pooling_raw_L10":  [9.2883, 9.4037, 9.0939, 9.2136],
    "Lagged_CKA_raw_L6":     [9.3079, 9.2361, 9.4323, 9.5392],
    "Lagged_CKA_raw_L8":     [9.4468, 9.4637, 9.5604, 9.7646],
    "Soft_DTW_raw_L6":       [9.2884, 9.4131, 9.4164, 9.6665],
    "Soft_DTW_raw_L8":       [9.2884, 9.2211, 9.7051, 9.5806],
    "Soft_DTW_raw_L10":      [9.2369, 9.2495, 9.5516, 9.6219],
    "Mean_Pooling_wh_L6":    [9.4862, 9.4774, 9.5044, 9.3634],
    "Mean_Pooling_wh_L8":    [9.3839, 9.3293, 9.2703, 9.3730],
    "Mean_Pooling_wh_L10":   [9.6658, 9.6112, 9.4987, 9.5139],
    "Lagged_CKA_wh_L6":      [9.8450, 9.7482, 9.7067, 9.7179],
    "Lagged_CKA_wh_L8":      [9.8298, 9.8442, 9.5450, 9.2544],
    "Lagged_CKA_wh_L10":     [10.0928,10.2111, 9.6862, 9.6552],
    "Soft_DTW_wh_L6":        [9.1308, 9.5270, 9.4050, 9.6796],
    "Soft_DTW_wh_L8":        [9.1308, 9.5270, 9.5520, 9.6566],
    "Soft_DTW_wh_L10":       [9.3979, 9.4102, 9.3712, 9.7543],
}

# Best TSFM per K: (short_label, rmse) — determined by min across all TSFM configs
BEST_TSFM = {
    5:  ("LC·raw·L10",  8.8863),
    10: ("DTW·raw·L8",  9.2211),
    20: ("MP·raw·L8",   9.0222),
    30: ("MP·raw·L6",   9.1711),
}

# ── Sensor selections by K ───────────────────────────────────────────────────

# IPG is computed at n_relevant=5 only; used as a fixed reference row for all K.
_IPG = [138, 124, 37, 90, 102]

SELECTIONS = {
    5: {
        "IPG (top-5)":        _IPG,
        "Geographic":         [145, 142, 37, 54, 116],
        "Lagged_CKA_raw_L10":[166, 145, 142, 37, 12],
        "Lasso":              [37, 145, 34, 157, 89],
    },
    10: {
        "IPG (top-5)":        _IPG,
        "Geographic":         [145, 142, 37, 54, 116, 127, 206, 166, 15, 169],
        "Lagged_CKA_raw_L10":[166, 145, 142, 37, 12, 181, 138, 81, 73, 15],
        "Lasso":              [37, 145, 34, 157, 89, 181, 107, 126, 72, 84],
    },
    20: {
        "IPG (top-5)":        _IPG,
        "Geographic":         [145, 142, 37, 54, 116, 127, 206, 166, 15, 169,
                                12, 18, 202, 107, 181, 126, 167, 90, 165, 146],
        "Lagged_CKA_raw_L10":[166, 145, 142, 37, 12, 181, 138, 81, 73, 15,
                                102, 127, 57, 68, 84, 143, 128, 76, 33, 172],
        "Lasso":              [37, 145, 34, 157, 89, 181, 107, 126, 72, 84,
                                76, 128, 68, 156, 165, 172, 12, 127, 73, 23],
    },
}

SELECTIONS_K5 = SELECTIONS[5]   # kept for backward compatibility

# ── Layer ablation at K=5 ────────────────────────────────────────────────────

LAYER_RMSE = {6: 9.3079, 8: 9.4468, 10: 8.8863}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: RMSE vs K  (no faded range)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_vs_k(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(KS, RMSE["Univariate"], color="black",   linestyle="--", linewidth=1.4,
            label="Univariate")
    ax.plot(KS, RMSE["Geographic"], color="dimgray", linestyle="--", linewidth=1.4,
            marker="^", markersize=7, label="Geographic")
    ax.plot(KS, RMSE["Lasso"],      color="dimgray", linestyle=":",  linewidth=1.4,
            marker="s", markersize=7, label="Lasso")
    ax.plot(KS, RMSE["Lagged_CKA_raw_L10"], color="#1a6fc4", linestyle="-",
            linewidth=2.5, marker="*", markersize=12,
            label="Lagged-CKA (raw, L10)", zorder=5)

    ax.annotate(
        f"RMSE = {RMSE['Lagged_CKA_raw_L10'][0]:.3f}",
        xy=(5, RMSE["Lagged_CKA_raw_L10"][0]),
        xytext=(7.5, RMSE["Lagged_CKA_raw_L10"][0] - 0.22),
        fontsize=11, color="#1a6fc4",
        arrowprops=dict(arrowstyle="->", color="#1a6fc4", lw=1.2),
    )

    ax.set_xlabel("Number of selected sensors (K)", fontsize=14)
    ax.set_ylabel("Test RMSE (mph)", fontsize=14)
    ax.set_title("Downstream RMSE vs. Sensors Selected", fontsize=14)
    ax.set_xticks(KS)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(8.4, 10.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Sensor selection dot matrix at K=5  (legend outside, no overlap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensor_matrix(out_path: str) -> None:
    row_labels = list(SELECTIONS_K5.keys())
    all_sensors = sorted({s for sensors in SELECTIONS_K5.values() for s in sensors})
    n_rows = len(row_labels)
    n_cols = len(all_sensors)
    sensor_idx = {s: i for i, s in enumerate(all_sensors)}

    matrix = np.zeros((n_rows, n_cols))
    for r, method in enumerate(row_labels):
        for s in SELECTIONS_K5[method]:
            matrix[r, sensor_idx[s]] = 1

    row_colors = {
        "IPG Ground Truth":   "#2ca02c",
        "IPG (top-5)":        "#2ca02c",
        "Geographic":         "#7f7f7f",
        "Lagged_CKA_raw_L10": "#1a6fc4",
        "Lasso":              "#d62728",
    }

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 0.6 + 1.5), 3.8))

    for r, method in enumerate(row_labels):
        color = row_colors[method]
        for c, sensor in enumerate(all_sensors):
            if matrix[r, c] == 1:
                ax.scatter(c, r, s=220, color=color, zorder=3,
                           edgecolors="white", linewidths=0.6)
            else:
                ax.scatter(c, r, s=40, color="#dddddd", zorder=2, marker="o")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(s) for s in all_sensors], fontsize=10,
                        rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Sensor (df column index)", fontsize=13)
    ax.set_title("Sensor Selections at K = 5", fontsize=14)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=row_colors[m], label=m) for m in row_labels]
    ax.legend(handles=patches, fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, framealpha=0.9)

    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2b: Sensor selection dot matrix (generic, any K)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensor_matrix_k(k: int, out_path: str) -> None:
    """Dot matrix for a given K.  IPG row always shows only the top-5 sensors
    as filled dots regardless of K, so it acts as a fixed reference."""
    selections = SELECTIONS[k]
    row_labels = list(selections.keys())
    all_sensors = sorted({s for sensors in selections.values() for s in sensors})
    n_rows = len(row_labels)
    n_cols = len(all_sensors)
    sensor_idx = {s: i for i, s in enumerate(all_sensors)}

    row_colors = {
        "IPG (top-5)":        "#2ca02c",
        "Geographic":         "#7f7f7f",
        "Lagged_CKA_raw_L10": "#1a6fc4",
        "Lasso":              "#d62728",
    }

    dot_w = max(0.38, 9.0 / n_cols)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * dot_w + 2.5), 3.8))

    for r, method in enumerate(row_labels):
        color = row_colors[method]
        selected = set(selections[method])
        for c, sensor in enumerate(all_sensors):
            if sensor in selected:
                ax.scatter(c, r, s=200, color=color, zorder=3,
                           edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(c, r, s=30, color="#e0e0e0", zorder=2, marker="o")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(s) for s in all_sensors], fontsize=8,
                        rotation=60, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Sensor (df column index)", fontsize=13)
    ax.set_title(f"Sensor Selections at K = {k}", fontsize=14)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=row_colors[m], label=m) for m in row_labels]
    ax.legend(handles=patches, fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, framealpha=0.9)

    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Layer ablation bar chart at K=5
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_ablation(out_path: str) -> None:
    layers = [6, 8, 10]
    rmse_vals = [LAYER_RMSE[l] for l in layers]
    colors = ["#a8c8e8", "#6aaed6", "#1a6fc4"]
    labels = [f"Layer {l}" for l in layers]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, rmse_vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, rmse_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.025,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    best_idx = np.argmin(rmse_vals)
    bars[best_idx].set_edgecolor("#e8a000")
    bars[best_idx].set_linewidth(2.5)
    ax.annotate(
        "Best",
        xy=(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            rmse_vals[best_idx] - 0.05),
        xytext=(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
                rmse_vals[best_idx] - 0.25),
        ha="center", fontsize=11, color="#e8a000", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#e8a000", lw=1.5),
    )

    ax.set_ylim(8.5, 9.8)
    ax.set_ylabel("Test RMSE (mph)", fontsize=14)
    ax.set_title("Lagged-CKA (raw): RMSE by Layer  (K = 5)", fontsize=14)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: SOTA comparison bar chart with best-TSFM config annotations
# ─────────────────────────────────────────────────────────────────────────────

def plot_sota_best_tsfm(out_path: str) -> None:
    """Grouped bar chart identical in layout to plot_sota_comparison_bar,
    with the 'Best TSFM' bar at each K annotated with the winning config name.
    """
    bar_methods = ["Best TSFM", "Geographic", "Lasso", "Pearson", "RF"]
    colors = {
        "Best TSFM":  "#1a6fc4",
        "Geographic": "#7f7f7f",
        "Lasso":      "#e8a000",
        "Pearson":    "#55a868",
        "RF":         "#d62728",
    }

    rmse_by_method = {
        "Best TSFM": [BEST_TSFM[k][1] for k in KS],
        "Geographic": RMSE["Geographic"],
        "Lasso":      RMSE["Lasso"],
        "Pearson":    RMSE["Pearson"],
        "RF":         RMSE["RF"],
    }

    n_groups = len(KS)
    n_bars   = len(bar_methods)
    width    = 0.14
    offsets  = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5))

    bar_objs = {}
    for i, method in enumerate(bar_methods):
        vals = rmse_by_method[method]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=method, color=colors[method], edgecolor="white")
        bar_objs[method] = bars

    # Annotate best-TSFM bars with config name
    for j, k in enumerate(KS):
        label, rmse = BEST_TSFM[k]
        bar = bar_objs["Best TSFM"][j]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rmse + 0.02,
            label,
            ha="center", va="bottom",
            fontsize=7.5, color="#1a6fc4", fontweight="bold",
            rotation=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in KS], fontsize=11)
    ax.set_ylabel("Test RMSE (mph)", fontsize=13)
    ax.set_title("Downstream RMSE by Method and K", fontsize=14)
    ax.set_ylim(8.5, 10.4)
    ax.legend(fontsize=11, loc="upper right")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Pairwise Jaccard heatmap at K=5
# ─────────────────────────────────────────────────────────────────────────────

_JACCARD_METHODS = {
    "IPG":      [138, 124, 37, 90, 102],
    "Geo":      [145, 142, 37, 54, 116],
    "LC·L10":   [166, 145, 142, 37, 12],
    "MP·L10":   [142, 145, 37, 57, 175],
    "DTW·L10":  [145, 37, 142, 116, 54],
    "Lasso":    [37, 145, 34, 157, 89],
    "Pearson":  [145, 37, 142, 54, 116],
}


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def plot_jaccard_heatmap(out_path: str) -> None:
    labels = list(_JACCARD_METHODS.keys())
    n = len(labels)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            j_val = _jaccard(
                list(_JACCARD_METHODS.values())[i],
                list(_JACCARD_METHODS.values())[j],
            )
            mat[i, j] = mat[j, i] = j_val

    # Mask lower triangle (keep diagonal + upper)
    mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        mat,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Jaccard similarity", "shrink": 0.75},
        annot_kws={"size": 11, "weight": "bold"},
        square=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Pairwise Selection Agreement at K = 5  (Jaccard)", fontsize=14)
    ax.tick_params(axis="x", labelsize=11, rotation=30)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4b: SOTA bar chart v2 — y from 0, minimal axis labels (PENDING APPROVAL)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sota_best_tsfm_v2(out_path: str) -> None:
    """Same as plot_sota_best_tsfm but y-axis starts at 0 and label is just 'RMSE'."""
    bar_methods = ["Best TSFM", "Geographic", "Lasso", "Pearson", "RF"]
    colors = {
        "Best TSFM":  "#1a6fc4",
        "Geographic": "#7f7f7f",
        "Lasso":      "#e8a000",
        "Pearson":    "#55a868",
        "RF":         "#d62728",
    }

    rmse_by_method = {
        "Best TSFM": [BEST_TSFM[k][1] for k in KS],
        "Geographic": RMSE["Geographic"],
        "Lasso":      RMSE["Lasso"],
        "Pearson":    RMSE["Pearson"],
        "RF":         RMSE["RF"],
    }

    n_groups = len(KS)
    n_bars   = len(bar_methods)
    width    = 0.14
    offsets  = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5))

    bar_objs = {}
    for i, method in enumerate(bar_methods):
        vals = rmse_by_method[method]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=method, color=colors[method], edgecolor="white")
        bar_objs[method] = bars

    for j, k in enumerate(KS):
        label, rmse = BEST_TSFM[k]
        bar = bar_objs["Best TSFM"][j]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rmse + 0.15,
            label,
            ha="center", va="bottom",
            fontsize=7.5, color="#1a6fc4", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in KS], fontsize=11)
    ax.set_ylabel("RMSE", fontsize=13)
    ax.set_title("Downstream RMSE by Method and K", fontsize=14)
    ax.set_ylim(0, 11.5)
    ax.legend(fontsize=11, loc="upper right")
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main(out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    plot_rmse_vs_k(f"{out_dir}/rmse_highlight.png")
    plot_sensor_matrix(f"{out_dir}/sensor_matrix.png")
    plot_sensor_matrix_k(10, f"{out_dir}/sensor_matrix_k10.png")
    plot_sensor_matrix_k(20, f"{out_dir}/sensor_matrix_k20.png")
    plot_layer_ablation(f"{out_dir}/layer_ablation.png")
    plot_sota_best_tsfm(f"{out_dir}/sota_best_tsfm.png")
    plot_sota_best_tsfm_v2(f"{out_dir}/sota_best_tsfm_v2.png")
    plot_jaccard_heatmap(f"{out_dir}/jaccard_k5.png")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    main(args.out_dir)
