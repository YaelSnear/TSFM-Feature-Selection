"""Publication-quality bar charts for the TSFM feature selection ablation.

Four plots:
    1. Downstream RMSE: best layer per (method × condition), with baselines.
       Each bar is annotated with its winning layer and condition (e.g. "L8 (raw)").
    2. Proxy MRR: same method grouping, same annotations.
    3. Layer comparison: RMSE across layers [6, 8, 10] per method (best condition).
    4. Precision@K vs RMSE: minimalist scatter — one best point per method (min RMSE)
       plus Geographic star marker, with text labels. No trendline.

Aesthetic rules (publication style):
    - No numeric data labels above bars (layer annotations only).
    - Legends placed outside the plot area (bbox_to_anchor).
    - Clean titles with no timestamps or experiment IDs.
    - Consistent palette: Raw=Blue, Whitened=Orange.
    - Each baseline has a unique colour and linestyle so they are distinguishable.
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

_BASELINE_STYLES: dict[str, dict] = {
    "Univariate": {
        "color":     "#55A868",   # green
        "linestyle": "--",
        "label":     "Univariate (Y only)",
    },
    "Geographic": {
        "color":     "#C44E52",   # red
        "linestyle": "-.",
        "label":     "Geographic (adj. sensors)",
    },
}

_LAYER_PALETTE = {"6": "#4878D0", "8": "#EE854A", "10": "#6ACC65"}

_METHOD_DISPLAY = {
    "Mean_Pooling": "Mean\nPooling",
    "Lagged_CKA":   "Lagged\nCKA",
    "Soft_DTW":     "Soft-DTW\n(γ=1.0)",
}

# Readable single-line names for scatter labels.
_METHOD_READABLE = {
    "Mean_Pooling": "Mean Pooling",
    "Lagged_CKA":   "Lagged CKA",
    "Soft_DTW":     "Soft-DTW",
}

_METHOD_COLORS = {
    "Mean_Pooling": "#4C72B0",
    "Lagged_CKA":   "#DD8452",
    "Soft_DTW":     "#55A868",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_softdtw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Soft_DTW_g* → Soft_DTW in the method column."""
    out = df.copy()
    out["method"] = out["method"].str.replace(r"^Soft_DTW_g.*", "Soft_DTW", regex=True)
    return out


def _best_layer_per_method_condition(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each (method, condition), keep only the row with the best metric value."""
    scored = df[~df["method"].isin(["Univariate", "Geographic"])].copy()
    scored = _normalise_softdtw(scored)
    agg_fn = "min" if metric == "RMSE" else "max"
    idx = scored.groupby(["method", "condition"])[metric].transform(agg_fn) == scored[metric]
    return scored[idx].drop_duplicates(subset=["method", "condition"])


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


def _layer_annotation(scored: pd.DataFrame, method: str, condition: str) -> str:
    """Return 'L{layer} ({cond_abbr})' for the winning bar, or '' if missing."""
    row = scored.loc[(scored["method"] == method) & (scored["condition"] == condition)]
    if row.empty:
        return ""
    layer = row["layer"].values[0]
    cond_abbr = "raw" if condition == "raw" else "wht"
    try:
        return f"L{int(layer)} ({cond_abbr})"
    except (ValueError, TypeError):
        return ""


def _annotate_bars(
    ax: plt.Axes,
    scored: pd.DataFrame,
    methods: list[str],
    x: np.ndarray,
    width: float,
    raw_vals: list[float],
    whi_vals: list[float],
    y_top: float,
) -> None:
    """Add winning-layer annotations inside the top of each bar."""
    inset = y_top * 0.025   # distance from bar top, going downward

    for i, m in enumerate(methods):
        for vals, condition, sign in [
            (raw_vals, "raw", -1),
            (whi_vals, "whitened", +1),
        ]:
            val = vals[i]
            if np.isnan(val) or val <= 0:
                continue
            label = _layer_annotation(scored, m, condition)
            if label:
                ax.text(
                    x[i] + sign * width / 2,
                    val - inset,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )


# ---------------------------------------------------------------------------
# Plot 1: RMSE comparison (best layer per method)
# ---------------------------------------------------------------------------

def plot_rmse_bar(
    df: pd.DataFrame,
    out_dir: Path,
    timestamp: str,
) -> None:
    """Grouped bar chart: best-layer Downstream RMSE per method × condition + baselines."""
    scored = _best_layer_per_method_condition(
        df[~df["method"].isin(["Univariate", "Geographic"])], "RMSE"
    )
    methods = ["Mean_Pooling", "Lagged_CKA", "Soft_DTW"]

    raw_vals = [
        scored.loc[(scored["method"] == m) & (scored["condition"] == "raw"), "RMSE"].mean()
        for m in methods
    ]
    whi_vals = [
        scored.loc[(scored["method"] == m) & (scored["condition"] == "whitened"), "RMSE"].mean()
        for m in methods
    ]

    x     = np.arange(len(methods))
    width = 0.30

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - width / 2, raw_vals, width, label="Raw",      color=_C_RAW,      edgecolor="white")
    ax.bar(x + width / 2, whi_vals, width, label="Whitened", color=_C_WHITENED, edgecolor="white")

    finite_vals = [v for v in raw_vals + whi_vals if not np.isnan(v)]
    y_top = max(finite_vals) if finite_vals else 1.0

    for bl_name, style in _BASELINE_STYLES.items():
        rows = df[df["method"] == bl_name]
        if not rows.empty:
            val = float(rows["RMSE"].iloc[0])
            if not np.isnan(val):
                y_top = max(y_top, val)
                ax.axhline(
                    val,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                    alpha=0.9,
                    label=style["label"],
                )

    _annotate_bars(ax, scored, methods, x, width, raw_vals, whi_vals, y_top)

    ax.set_xticks(x)
    ax.set_xticklabels([_METHOD_DISPLAY.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_xlabel("Scoring Method", fontsize=10)
    ax.set_title("Optimal Downstream RMSE (Best Layer per Method)", fontsize=12, fontweight="bold", pad=10)

    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: MRR comparison (best layer per method)
# ---------------------------------------------------------------------------

def plot_mrr_bar(
    df: pd.DataFrame,
    out_dir: Path,
    timestamp: str,
) -> None:
    """Grouped bar chart: best-layer Proxy MRR per method × condition."""
    scored = _best_layer_per_method_condition(
        df[~df["method"].isin(["Univariate", "Geographic"])], "MRR"
    )
    methods = ["Mean_Pooling", "Lagged_CKA", "Soft_DTW"]

    raw_vals = [
        scored.loc[(scored["method"] == m) & (scored["condition"] == "raw"), "MRR"].mean()
        for m in methods
    ]
    whi_vals = [
        scored.loc[(scored["method"] == m) & (scored["condition"] == "whitened"), "MRR"].mean()
        for m in methods
    ]

    x     = np.arange(len(methods))
    width = 0.30

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - width / 2, raw_vals, width, label="Raw",      color=_C_RAW,      edgecolor="white")
    ax.bar(x + width / 2, whi_vals, width, label="Whitened", color=_C_WHITENED, edgecolor="white")

    finite_vals = [v for v in raw_vals + whi_vals if not np.isnan(v)]
    y_top = max(finite_vals) if finite_vals else 1.0
    _annotate_bars(ax, scored, methods, x, width, raw_vals, whi_vals, y_top)

    ax.set_xticks(x)
    ax.set_xticklabels([_METHOD_DISPLAY.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("MRR (higher = better)", fontsize=10)
    ax.set_xlabel("Scoring Method", fontsize=10)
    ax.set_title("Optimal Proxy MRR (Best Layer per Method)", fontsize=12, fontweight="bold", pad=10)

    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_mrr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Layer comparison — RMSE across layers per method
# ---------------------------------------------------------------------------

def plot_layer_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: RMSE per method × layer (best condition per cell)."""
    scored = df[~df["method"].isin(["Univariate", "Geographic"])].copy()
    scored = _normalise_softdtw(scored)

    methods = ["Mean_Pooling", "Lagged_CKA", "Soft_DTW"]
    layers  = sorted(scored["layer"].unique())

    best = (
        scored.groupby(["method", "layer"])["RMSE"]
              .min()
              .reset_index()
    )

    x        = np.arange(len(methods))
    n_layers = len(layers)
    width    = 0.22
    offsets  = np.linspace(-(n_layers - 1) / 2, (n_layers - 1) / 2, n_layers) * width

    fig, ax = plt.subplots(figsize=(8, 5))

    for offset, layer in zip(offsets, layers):
        vals = [
            best.loc[(best["method"] == m) & (best["layer"] == layer), "RMSE"].values
            for m in methods
        ]
        vals = [v[0] if len(v) > 0 else float("nan") for v in vals]
        color = _LAYER_PALETTE.get(str(layer), "#888888")
        ax.bar(x + offset, vals, width, label=f"Layer {layer}", color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([_METHOD_DISPLAY.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_xlabel("Scoring Method", fontsize=10)
    ax.set_title("RMSE by Layer (best condition per method)", fontsize=12, fontweight="bold", pad=10)

    _apply_style(ax)
    _outside_legend(ax)

    path = out_dir / "plot_layer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Precision@K vs RMSE — minimalist scatter (best point per method)
# ---------------------------------------------------------------------------

def plot_precision_vs_rmse(df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    """Scatter: one best point per method + Geographic star. No trendline."""
    prec_col = f"Precision_at_{top_k}"

    scored = df[~df["method"].isin(["Univariate", "Geographic"])].copy()
    scored = _normalise_softdtw(scored)
    scored = scored[scored[prec_col].notna() & scored["RMSE"].notna()]

    # Single absolute best point per method (minimum RMSE across all layer/condition combos).
    best_per_method = scored.loc[scored.groupby("method")["RMSE"].idxmin()].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in best_per_method.iterrows():
        method   = row["method"]
        color    = _METHOD_COLORS.get(method, "#888888")
        prec_val = row[prec_col]
        rmse_val = row["RMSE"]

        ax.scatter(prec_val, rmse_val, color=color, s=80, zorder=3)

        # Build label: "Lagged CKA (L10, raw)"
        readable = _METHOD_READABLE.get(method, method)
        cond_abbr = "raw" if row["condition"] == "raw" else "wht"
        try:
            layer_str = f"L{int(row['layer'])}"
        except (ValueError, TypeError):
            layer_str = str(row["layer"])
        label_text = f"{readable} ({layer_str}, {cond_abbr})"

        ax.annotate(
            label_text,
            xy=(prec_val, rmse_val),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )

    # Geographic baseline as a star marker (Precision=1.0 by construction).
    geo_rows = df[df["method"] == "Geographic"]
    if not geo_rows.empty:
        geo_rmse = float(geo_rows["RMSE"].iloc[0])
        ax.scatter(
            [1.0],
            [geo_rmse],
            marker="*",
            s=200,
            color=_BASELINE_STYLES["Geographic"]["color"],
            label="Geographic baseline",
            zorder=4,
        )
        ax.annotate(
            "Geographic",
            xy=(1.0, geo_rmse),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=_BASELINE_STYLES["Geographic"]["color"],
        )

    ax.set_xlabel(f"Precision@{top_k}", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title(
        f"Feature Selection Quality vs Downstream RMSE (Precision@{top_k})",
        fontsize=12, fontweight="bold", pad=10,
    )

    _apply_style(ax)

    path = out_dir / "plot_precision_vs_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
