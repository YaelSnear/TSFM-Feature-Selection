"""Publication-quality plots for the TSFM feature selection ablation.

Four plots:
    1. plot_rmse_vs_k            — Scalability: RMSE vs K for fixed best configs + baselines.
    2. plot_sota_comparison_bar  — Grouped bar chart: best TSFM vs baselines at each K value.
    3. plot_layer_ablation       — Layer Collapse Ablation: RMSE vs Layer at K=5 (line chart).
    4. plot_overlap_with_geo     — Topological Gap: Jaccard(best TSFM, Geographic) across K.

Plot 3 strictly filters the DataFrame to k == 5.

Aesthetic rules (publication style):
    - Data labels on bars only for Plot 2 (as specified).
    - Legends placed outside the plot area (bbox_to_anchor).
    - Clean titles; no timestamps.
    - Raw=Blue, Whitened=Orange; Lasso=Purple, RF=Brown.
    - Each baseline has a unique colour and linestyle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_C_RAW      = "#4C72B0"   # blue  — Raw embeddings
_C_WHITENED = "#DD8452"   # orange — Whitened embeddings
_C_LASSO    = "#9467BD"   # purple — Lasso baseline
_C_RF       = "#8C564B"   # brown  — RF baseline
_C_PEARSON  = "#17BECF"   # teal  — Pearson baseline

_BASELINE_STYLES: dict[str, dict] = {
    "Univariate": {
        "color":     "#55A868",
        "linestyle": "--",
        "label":     "Univariate (Y only)",
    },
    "Geographic": {
        "color":     "#C44E52",
        "linestyle": "-.",
        "label":     "Geographic (adj. sensors)",
    },
    "Lasso": {
        "color":     _C_LASSO,
        "linestyle": ":",
        "label":     "Lasso",
    },
    "RF": {
        "color":     _C_RF,
        "linestyle": (0, (3, 1, 1, 1)),
        "label":     "Random Forest",
    },
    "Pearson": {
        "color":     _C_PEARSON,
        "linestyle": (0, (5, 2)),
        "label":     "Pearson",
    },
}

_LAYER_PALETTE = {"6": "#4878D0", "8": "#EE854A", "10": "#6ACC65"}

_METHOD_DISPLAY = {
    "Mean_Pooling": "Mean\nPooling",
    "Lagged_CKA":   "Lagged\nCKA",
    "Soft_DTW":     "Soft-DTW\n(γ=1.0)",
    "Univariate":   "Univariate",
    "Geographic":   "Geographic",
    "Lasso":        "Lasso",
    "RF":           "Random\nForest",
    "Pearson":      "Pearson",
}

_METHOD_READABLE = {
    "Mean_Pooling": "Mean Pooling",
    "Lagged_CKA":   "Lagged CKA",
    "Soft_DTW":     "Soft-DTW",
    "Univariate":   "Univariate",
    "Geographic":   "Geographic",
    "Lasso":        "Lasso",
    "RF":           "Random Forest",
    "Pearson":      "Pearson",
}

_METHOD_COLORS = {
    "Mean_Pooling": "#4C72B0",
    "Lagged_CKA":   "#DD8452",
    "Soft_DTW":     "#55A868",
    "Univariate":   "#55A868",
    "Geographic":   "#C44E52",
    "Lasso":        _C_LASSO,
    "RF":           _C_RF,
    "Pearson":      _C_PEARSON,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalise_softdtw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Soft_DTW_g* → Soft_DTW in the method column."""
    out = df.copy()
    out["method"] = out["method"].str.replace(r"^Soft_DTW_g.*", "Soft_DTW", regex=True)
    return out


def _apply_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _outside_legend(ax: plt.Axes) -> None:
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
    )


def _best_latent_config_at_k5(df: pd.DataFrame) -> dict[str, tuple]:
    """Return {method_family: (layer, condition)} for minimum RMSE at K=5.

    Only considers the three latent method families (Mean_Pooling, Lagged_CKA,
    Soft_DTW). Soft_DTW_g* variants are normalised to Soft_DTW first.
    """
    df5 = df[df["k"] == 5].copy()
    df5 = _normalise_softdtw(df5)
    latent_methods = {"Mean_Pooling", "Lagged_CKA", "Soft_DTW"}
    scored = df5[df5["method"].isin(latent_methods)]
    result: dict[str, tuple] = {}
    for method, grp in scored.groupby("method"):
        idx = grp["RMSE"].idxmin()
        result[method] = (grp.loc[idx, "layer"], grp.loc[idx, "condition"])
    return result


# ---------------------------------------------------------------------------
# Plot 1: RMSE vs K — Scalability
# ---------------------------------------------------------------------------

def plot_rmse_vs_k(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: X=K, Y=RMSE.

    For each of Mean_Pooling, Lagged_CKA, and Soft_DTW, a single (layer,
    condition) configuration is chosen — the one with the lowest RMSE at K=5
    (no cherry-picking: the champion is fixed at K=5, then traced across all K).
    Separate lines are drawn for Geographic, Lasso, and RF.
    """
    df = _normalise_softdtw(df)
    best_configs = _best_latent_config_at_k5(df)

    k_values = sorted(df["k"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    latent_line_styles = ["-o", "-s", "-^"]
    for i, (method, (layer, condition)) in enumerate(sorted(best_configs.items())):
        # Select the rows for this exact (method, layer, condition) at every K.
        mask = (
            (df["method"] == method) &
            (df["layer"] == layer) &
            (df["condition"] == condition)
        )
        sub = df[mask].sort_values("k")
        if sub.empty:
            continue
        color = _METHOD_COLORS.get(method, "#888888")
        cond_abbr = "raw" if condition == "raw" else "wht"
        label = f"{_METHOD_READABLE.get(method, method)} (L{layer}, {cond_abbr})"
        ax.plot(sub["k"], sub["RMSE"], latent_line_styles[i % 3],
                color=color, label=label, linewidth=1.8, markersize=6)

    # Baseline lines: Geographic, Lasso, RF (one value per K, no layer/condition).
    for bl in ("Geographic", "Lasso", "RF"):
        style = _BASELINE_STYLES[bl]
        sub = df[df["method"] == bl].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["RMSE"],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.8, label=style["label"], marker="D", markersize=5)

    ax.set_xticks(k_values)
    ax.set_xlabel("K (number of sensors selected)", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title(
        "Scalability: RMSE vs K (fixed best config at K=5)",
        fontsize=12, fontweight="bold", pad=10,
    )
    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_rmse_vs_k.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: SOTA comparison bar chart (K=5 only)
# ---------------------------------------------------------------------------

def plot_sota_comparison_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: one group per K value, bars for Best TSFM / Geographic /
    RF / Lasso / Pearson.

    The 'Best TSFM' bar at each K is the minimum RMSE across all latent method
    configurations (method family × layer × condition) at that K — determined
    independently per K, not locked to the K=5 winner.
    """
    df = _normalise_softdtw(df)
    k_values = sorted(df["k"].unique())
    latent_methods = {"Mean_Pooling", "Lagged_CKA", "Soft_DTW"}

    # Methods shown as bars within each K group (order determines bar position).
    bar_methods = ["Best TSFM", "Geographic", "RF", "Lasso", "Pearson"]
    bar_colors  = {
        "Best TSFM": _C_RAW,
        "Geographic": _BASELINE_STYLES["Geographic"]["color"],
        "RF":         _C_RF,
        "Lasso":      _C_LASSO,
        "Pearson":    _C_PEARSON,
    }

    # Collect RMSE values: {method_label: [rmse_at_k5, rmse_at_k10, ...]}
    rmse_by_method: dict[str, list[float]] = {m: [] for m in bar_methods}
    for k in k_values:
        dft = df[df["k"] == k]
        # Best TSFM: minimum RMSE across all latent rows at this K.
        latent_rows = dft[dft["method"].isin(latent_methods)]
        rmse_by_method["Best TSFM"].append(
            float(latent_rows["RMSE"].min()) if not latent_rows.empty else np.nan
        )
        for bl in ("Geographic", "RF", "Lasso", "Pearson"):
            rows = dft[dft["method"] == bl]
            rmse_by_method[bl].append(
                float(rows["RMSE"].iloc[0]) if not rows.empty else np.nan
            )

    n_groups  = len(k_values)
    n_bars    = len(bar_methods)
    width     = 0.14
    offsets   = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    x         = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(bar_methods):
        vals = rmse_by_method[method]
        ax.bar(x + offsets[i], vals, width,
               label=method, color=bar_colors[method], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in k_values], fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_xlabel("K (sensors selected)", fontsize=10)
    ax.set_title(
        f"Downstream RMSE at K = {', '.join(str(k) for k in k_values)}",
        fontsize=12, fontweight="bold", pad=10,
    )
    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_sota_comparison_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Layer ablation — RMSE vs Layer at K=5
# ---------------------------------------------------------------------------

def plot_layer_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart at K=5: X=Layer [6, 8, 10], Y=RMSE.

    Three lines:
        - Mean_Pooling / raw
        - Lagged_CKA   / raw
        - Lagged_CKA   / whitened
    """
    df5 = df[df["k"] == 5].copy()
    df5 = _normalise_softdtw(df5)

    lines = [
        ("Mean_Pooling", "raw",      _C_RAW,      "-o", "Mean Pooling (raw)"),
        ("Lagged_CKA",   "raw",      _C_WHITENED, "-s", "Lagged CKA (raw)"),
        ("Lagged_CKA",   "whitened", "#55A868",   "-^", "Lagged CKA (whitened)"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for method, condition, color, marker, label in lines:
        mask = (df5["method"] == method) & (df5["condition"] == condition)
        sub  = df5[mask].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"].astype(int), sub["RMSE"], marker,
                color=color, label=label, linewidth=1.8, markersize=7)

    layers_present = sorted(
        df5[df5["method"].isin({"Mean_Pooling", "Lagged_CKA"})]["layer"]
        .dropna()
        .astype(int)
        .unique()
    )
    ax.set_xticks(layers_present)
    ax.set_xlabel("Transformer Layer", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title(
        "Layer Collapse Ablation (K=5)",
        fontsize=12, fontweight="bold", pad=10,
    )
    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_layer_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Sensor Overlap Heatmap (Jaccard at K=5)
# ---------------------------------------------------------------------------

def plot_overlap_with_geo(
    selected_sensors_dict: dict[str, dict[str, list[int]]],
    results_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Line chart: Jaccard similarity between the best TSFM method and the
    Geographic baseline across all K values (The Topological Gap).

    At each K, the 'best TSFM' is determined dynamically as the latent method
    configuration (method × condition × layer) with the lowest RMSE at that K.
    The method_key format used to look up sensor sets matches the one written
    by run_experiment.py: "{method}_{condition}_L{layer}".

    A low, flat curve here is evidence that the latent model is discovering a
    topology distinct from the physical road-network adjacency.
    """
    df = _normalise_softdtw(results_df.copy())
    latent_methods = {"Mean_Pooling", "Lagged_CKA", "Soft_DTW"}

    k_values = sorted(int(k) for k in selected_sensors_dict.keys())
    jaccard_vals: list[float] = []
    point_labels: list[str]   = []

    for k in k_values:
        k_map   = selected_sensors_dict.get(str(k), {})
        geo_set = set(k_map.get("Geographic", []))

        dft         = df[df["k"] == k]
        latent_rows = dft[dft["method"].isin(latent_methods)]

        if latent_rows.empty or not geo_set:
            jaccard_vals.append(np.nan)
            point_labels.append("")
            continue

        best_row  = latent_rows.loc[latent_rows["RMSE"].idxmin()]
        method    = best_row["method"]
        layer     = best_row["layer"]
        condition = best_row["condition"]

        # Build the exact key used when saving to selected_sensors.json.
        method_key = f"{method}_{condition}_L{layer}"
        best_set   = set(k_map.get(method_key, []))

        union = len(best_set | geo_set)
        jaccard_vals.append(len(best_set & geo_set) / union if union > 0 else 1.0)

        # Short label for the winning config (annotated on the plot).
        cond_abbr = "raw" if condition == "raw" else "wht"
        point_labels.append(
            f"{_METHOD_READABLE.get(method, method)}\n(L{layer}, {cond_abbr})"
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(k_values, jaccard_vals, "-o",
            color=_C_RAW, linewidth=2.0, markersize=8, zorder=3)

    # Annotate each point with the winning TSFM config.
    for k, j, label in zip(k_values, jaccard_vals, point_labels):
        if np.isnan(j) or not label:
            continue
        ax.annotate(
            label,
            xy=(k, j),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=_C_RAW,
        )

    ax.set_xticks(k_values)
    ax.set_xlim(k_values[0] - 2, k_values[-1] + 2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("K (sensors selected)", fontsize=10)
    ax.set_ylabel("Jaccard Similarity with Geographic", fontsize=10)
    ax.set_title(
        "Topological Gap: Best TSFM vs. Geographic Baseline",
        fontsize=12, fontweight="bold", pad=10,
    )
    _apply_style(ax)

    path = out_dir / "plot_overlap_with_geo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
