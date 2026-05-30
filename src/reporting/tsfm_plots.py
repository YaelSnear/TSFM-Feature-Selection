"""Decision-oriented plots for the TSFM downstream experiment.

Six plots:
    1. plot_bar_rmse_by_method               — Mean RMSE per method at a fixed K
    2. plot_bar_mae_by_method                — Mean MAE per method at a fixed K
    3. plot_pct_improvement_vs_target_only   — % RMSE improvement over target_only
    4. plot_pct_improvement_vs_all_features  — % RMSE improvement over all_features_N
    5. plot_win_count_by_method              — Win count vs target_only / Pearson / all_features_N
    6. plot_rolewise_rmse                    — Grouped bar chart by role

All plots:
    - Use SEM error bars across n targets (paired units).
    - Do not pool K values — each plot uses a single fixed K.
    - Exclude is_replicated_baseline=True rows.
    - Aggregate random_k repeats per (target, K) before comparisons.
    - Exploratory: title prefix [Exploratory] on all plots.

All functions silently skip if required data is missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_C = {
    "target_only":    "#55A868",
    "all_features":   "#C44E52",   # matches all_features_N by prefix lookup
    "random_k":       "#8C564B",
    "Pearson":        "#17BECF",
    "Lagged_CKA":     "#DD8452",
    "Mean_CKA":       "#4C72B0",
    "Mean_Pooling":   "#4C72B0",
    "RandomForest":   "#6ACC65",
    "SparseLinear_L1":"#9467BD",
    "Soft_DTW":       "#F7B6D2",
}

_ROLE_SHORT = {
    "role_A_central":       "A-Central",
    "role_B_peripheral":    "B-Periph",
    "role_C_bridge":        "C-Bridge",
    "role_D_dense":         "D-Dense",
    "role_E_high_variance": "E-HighVar",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _outside_legend(ax: plt.Axes, ncol: int = 1) -> None:
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
        ncol=ncol,
    )


def _filter_scored(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_replicated_baseline" in out.columns:
        out = out[out["is_replicated_baseline"] != True]
    return out


def _method_base(method_str: str) -> str:
    return re.sub(r"_L\d+$", "", str(method_str))


def _is_all_features(method_str: str) -> bool:
    return str(method_str).startswith("all_features_")


def _get_color(method: str) -> str:
    if _is_all_features(method):
        return _C["all_features"]
    base = _method_base(method)
    return _C.get(method, _C.get(base, "#888888"))


def _find_all_features_method(df: pd.DataFrame) -> str | None:
    """Return the first method name starting with 'all_features_', or None."""
    for m in df["method"].dropna().unique():
        if _is_all_features(m):
            return str(m)
    return None


def _get_fixed_k(df: pd.DataFrame, prefer: int = 10) -> int:
    """Single K value from scored non-baseline methods."""
    baseline_methods = {"target_only"} | {m for m in df["method"].dropna().unique() if _is_all_features(m)}
    scored_ks = sorted(
        df[~df["method"].isin(baseline_methods)]["top_k"].dropna().unique()
    )
    if not scored_ks:
        return prefer
    return int(prefer) if prefer in scored_ks else int(scored_ks[0])


def _aggregate_random_k(df: pd.DataFrame) -> pd.DataFrame:
    """Replace per-repeat random_k rows with one row per (target, top_k) showing mean RMSE/MAE."""
    rk = df[df["method"] == "random_k"]
    if rk.empty:
        return df
    rk_agg = (
        rk.groupby(["target_sensor_id", "top_k"])[["RMSE", "MAE"]]
        .mean()
        .reset_index()
    )
    rk_agg["method"] = "random_k"
    for col in df.columns:
        if col not in rk_agg.columns:
            rk_agg[col] = np.nan
    return pd.concat([df[df["method"] != "random_k"], rk_agg], ignore_index=True, sort=False)


def _build_stats(df_k: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and SEM of metric across target sensors, per method."""
    stats = (
        df_k.groupby("method")[metric]
        .agg(mean_val="mean", sem_val=lambda x: x.sem(ddof=1) if len(x) > 1 else 0.0)
        .reset_index()
    )
    stats.columns = ["method", "mean", "sem"]
    return stats


# ---------------------------------------------------------------------------
# Plot 1: Bar RMSE by method
# ---------------------------------------------------------------------------

def plot_bar_rmse_by_method(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Mean RMSE ± SEM across targets per method at a fixed K."""
    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        k = _get_fixed_k(df)
        df_k = df[df["top_k"] == k].copy()
        to_rows = df[df["top_k"] == 0].copy()
        df_k = pd.concat([to_rows, df_k[df_k["top_k"] != 0]], ignore_index=True, sort=False)

        stats = _build_stats(df_k, "RMSE").sort_values("mean")
        if stats.empty:
            return

        n_targets = df_k["target_sensor_id"].nunique()
        fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.8 + 2), 5))
        colors = [_get_color(m) for m in stats["method"]]
        ax.bar(stats["method"], stats["mean"], yerr=stats["sem"],
               capsize=4, color=colors, alpha=0.85, error_kw={"linewidth": 1.2})
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Mean RMSE", fontsize=11)
        ax.set_title(f"[Exploratory] RMSE by Method (K={k}, n={n_targets} targets)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        _apply_style(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "bar_rmse_by_method.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] bar_rmse_by_method failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Plot 2: Bar MAE by method
# ---------------------------------------------------------------------------

def plot_bar_mae_by_method(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Mean MAE ± SEM across targets per method at a fixed K."""
    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        k = _get_fixed_k(df)
        df_k = df[df["top_k"] == k].copy()
        to_rows = df[df["top_k"] == 0].copy()
        df_k = pd.concat([to_rows, df_k[df_k["top_k"] != 0]], ignore_index=True, sort=False)

        stats = _build_stats(df_k, "MAE").sort_values("mean")
        if stats.empty:
            return

        n_targets = df_k["target_sensor_id"].nunique()
        fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.8 + 2), 5))
        colors = [_get_color(m) for m in stats["method"]]
        ax.bar(stats["method"], stats["mean"], yerr=stats["sem"],
               capsize=4, color=colors, alpha=0.85, error_kw={"linewidth": 1.2})
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Mean MAE", fontsize=11)
        ax.set_title(f"[Exploratory] MAE by Method (K={k}, n={n_targets} targets)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        _apply_style(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "bar_mae_by_method.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] bar_mae_by_method failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Plot 3: % improvement vs target_only
# ---------------------------------------------------------------------------

def plot_pct_improvement_vs_target_only(results_df: pd.DataFrame, out_dir: Path) -> None:
    """% RMSE improvement over target_only per method at a fixed K."""
    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        to_rmse = (
            df[df["method"] == "target_only"][["target_sensor_id", "RMSE"]]
            .rename(columns={"RMSE": "RMSE_to"})
            .drop_duplicates("target_sensor_id")
        )
        if to_rmse.empty:
            return

        k = _get_fixed_k(df)
        df_k = df[(df["top_k"] == k) & (df["method"] != "target_only")].copy()
        if df_k.empty:
            return

        df_k = df_k.merge(to_rmse, on="target_sensor_id", how="inner")
        df_k["pct_impr"] = (df_k["RMSE_to"] - df_k["RMSE"]) / df_k["RMSE_to"] * 100

        stats = (
            df_k.groupby("method")["pct_impr"]
            .agg(mean_val="mean", sem_val=lambda x: x.sem(ddof=1) if len(x) > 1 else 0.0)
            .reset_index()
        )
        stats.columns = ["method", "mean", "sem"]
        stats = stats.sort_values("mean", ascending=False)
        if stats.empty:
            return

        n_targets = df_k["target_sensor_id"].nunique()
        fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.8 + 2), 5))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in stats["mean"]]
        ax.bar(stats["method"], stats["mean"], yerr=stats["sem"],
               capsize=4, color=colors, alpha=0.75, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("% RMSE Improvement over target_only", fontsize=11)
        ax.set_title(f"[Exploratory] % Improvement vs target_only (K={k}, n={n_targets})", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        _apply_style(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "pct_improvement_vs_target_only.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] pct_improvement_vs_target_only failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Plot 4: % improvement vs all_features_N
# ---------------------------------------------------------------------------

def plot_pct_improvement_vs_all_features(results_df: pd.DataFrame, out_dir: Path) -> None:
    """% RMSE improvement over all_features_N per method at a fixed K.

    Detects the all_features_N method by prefix 'all_features_'.
    """
    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        af_method = _find_all_features_method(df)
        if af_method is None:
            return

        af_rmse = (
            df[df["method"] == af_method][["target_sensor_id", "RMSE"]]
            .rename(columns={"RMSE": "RMSE_af"})
            .drop_duplicates("target_sensor_id")
        )
        if af_rmse.empty:
            return

        k = _get_fixed_k(df)
        exclude = {"target_only", af_method}
        df_k = df[(df["top_k"] == k) & (~df["method"].isin(exclude))].copy()
        if df_k.empty:
            return

        df_k = df_k.merge(af_rmse, on="target_sensor_id", how="inner")
        df_k["pct_impr"] = (df_k["RMSE_af"] - df_k["RMSE"]) / df_k["RMSE_af"] * 100

        stats = (
            df_k.groupby("method")["pct_impr"]
            .agg(mean_val="mean", sem_val=lambda x: x.sem(ddof=1) if len(x) > 1 else 0.0)
            .reset_index()
        )
        stats.columns = ["method", "mean", "sem"]
        stats = stats.sort_values("mean", ascending=False)
        if stats.empty:
            return

        n_targets = df_k["target_sensor_id"].nunique()
        fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.8 + 2), 5))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in stats["mean"]]
        ax.bar(stats["method"], stats["mean"], yerr=stats["sem"],
               capsize=4, color=colors, alpha=0.75, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel(f"% RMSE Improvement over {af_method}", fontsize=11)
        ax.set_title(f"[Exploratory] % Improvement vs {af_method} (K={k}, n={n_targets})", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        _apply_style(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "pct_improvement_vs_all_features.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] pct_improvement_vs_all_features failed: {e}", flush=True)


# backward-compat alias (old name referenced from old orchestrator)
plot_pct_improvement_vs_all_candidates = plot_pct_improvement_vs_all_features


# ---------------------------------------------------------------------------
# Plot 5: Win count by method
# ---------------------------------------------------------------------------

def plot_win_count_by_method(results_df: pd.DataFrame, out_dir: Path) -> None:
    """For each scored method: count targets where it beats target_only / Pearson / all_features_N."""
    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        k = _get_fixed_k(df)
        af_method = _find_all_features_method(df)

        def _baseline_rmse(method_name: str) -> dict:
            rows = df[df["method"] == method_name][["target_sensor_id", "RMSE"]]
            rows = rows.drop_duplicates("target_sensor_id")
            return dict(zip(rows["target_sensor_id"], rows["RMSE"]))

        to_map   = _baseline_rmse("target_only")
        pear_map = _baseline_rmse("Pearson")
        af_map   = _baseline_rmse(af_method) if af_method else {}

        exclude = {"target_only"} | ({af_method} if af_method else set())
        df_k = df[(df["top_k"] == k) & (~df["method"].isin(exclude))].copy()
        if df_k.empty:
            return

        methods = sorted(df_k["method"].unique())
        win_to   = []
        win_pear = []
        win_af   = []
        for m in methods:
            grp = df_k[df_k["method"] == m]
            wt = wp = wa = 0
            for _, row in grp.iterrows():
                sid = row["target_sensor_id"]
                r = row["RMSE"]
                if r < to_map.get(sid, np.inf):   wt += 1
                if r < pear_map.get(sid, np.inf): wp += 1
                if r < af_map.get(sid, np.inf):   wa += 1
            win_to.append(wt)
            win_pear.append(wp)
            win_af.append(wa)

        x = np.arange(len(methods))
        width = 0.25
        n_targets = df_k["target_sensor_id"].nunique()
        af_label = af_method if af_method else "all_features_N"

        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.9 + 2), 5))
        ax.bar(x - width, win_to,   width, label="vs target_only", color="#55A868", alpha=0.85)
        ax.bar(x,         win_pear, width, label="vs Pearson",     color="#17BECF", alpha=0.85)
        ax.bar(x + width, win_af,   width, label=f"vs {af_label}", color="#C44E52", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"Win count (out of {n_targets} targets)", fontsize=11)
        ax.set_title(f"[Exploratory] Win Count by Method (K={k})", fontsize=12)
        ax.legend(fontsize=9)
        _apply_style(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "win_count_by_method.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] win_count_by_method failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Plot 6: Role-wise RMSE
# ---------------------------------------------------------------------------

def plot_rolewise_rmse(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: mean RMSE per role for key methods.

    Focus method bases: target_only, all_features_N (detected), Pearson, Lagged_CKA, Mean_CKA.
    Groups by role (A-E), error bars = SEM across targets per role.
    """
    FOCUS_BASE_PRIORITY = [
        "target_only",
        "all_features",   # matched by prefix
        "Pearson",
        "Lagged_CKA",
        "Mean_CKA",
    ]

    try:
        df = _filter_scored(results_df)
        df = _aggregate_random_k(df)
        df = df[df["repeat_id"].isna() | (df["method"] == "random_k")]

        if "target_role" not in df.columns:
            return

        af_method = _find_all_features_method(df)
        df = df.copy()
        df["method_base"] = df["method"].apply(
            lambda m: "all_features" if _is_all_features(m) else _method_base(m)
        )

        k = _get_fixed_k(df)
        to_rows    = df[(df["method"] == "target_only") & (df["top_k"] == 0)].copy()
        af_rows    = df[(df["method"] == af_method) & (df["top_k"] == (len(df[df["method"] == af_method]["top_k"].dropna().unique()) and df["top_k"] == df["top_k"]))].copy() if af_method else pd.DataFrame()
        # simpler: all rows for all_features method
        af_rows    = df[df["method_base"] == "all_features"].copy()
        other_k    = df[(df["top_k"] == k) & (~df["method"].isin({"target_only"} | ({af_method} if af_method else set())))].copy()
        df_k = pd.concat([to_rows, af_rows, other_k], ignore_index=True, sort=False)
        df_k["method_base"] = df_k["method"].apply(
            lambda m: "all_features" if _is_all_features(m) else _method_base(m)
        )

        df_k = df_k[df_k["method_base"].isin(FOCUS_BASE_PRIORITY)]
        if df_k.empty:
            return

        df_k["role_short"] = df_k["target_role"].map(_ROLE_SHORT).fillna(df_k["target_role"])
        roles   = [_ROLE_SHORT[r] for r in _ROLE_SHORT if _ROLE_SHORT[r] in df_k["role_short"].unique()]
        methods = [m for m in FOCUS_BASE_PRIORITY if m in df_k["method_base"].unique()]
        if not roles or not methods:
            return

        x     = np.arange(len(roles))
        width = 0.8 / max(len(methods), 1)

        fig, ax = plt.subplots(figsize=(max(8, len(roles) * 2), 5))
        for i, m in enumerate(methods):
            grp   = df_k[df_k["method_base"] == m]
            means = []
            sems  = []
            for role_short in roles:
                vals = grp[grp["role_short"] == role_short]["RMSE"].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem(ddof=1) if len(vals) > 1 else 0.0)
            color  = _get_color(m)
            offset = (i - len(methods) / 2 + 0.5) * width
            label  = af_method if (m == "all_features" and af_method) else m
            ax.bar(x + offset, means, width * 0.9,
                   yerr=sems, capsize=3, label=label,
                   color=color, alpha=0.85, error_kw={"linewidth": 1.0})

        ax.set_xticks(x)
        ax.set_xticklabels(roles, fontsize=9)
        ax.set_ylabel("Mean RMSE", fontsize=11)
        ax.set_title(f"[Exploratory] Role-wise RMSE (K={k})", fontsize=12)
        _apply_style(ax)
        _outside_legend(ax)
        plt.tight_layout()
        fig.savefig(out_dir / "rolewise_rmse.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[plot] rolewise_rmse failed: {e}", flush=True)
