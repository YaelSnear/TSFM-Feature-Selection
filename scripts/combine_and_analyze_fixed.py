"""Combine original FULL RUN results with Lagged_CKA_L8_fixed rerun.

Reads:
  - outputs/.../results/results_incremental.jsonl           (original FULL RUN)
  - outputs/.../lagged_cka_fixed_rerun/results_incremental_fixed.jsonl

Renames old Lagged_CKA_L8 → Lagged_CKA_L8_INVALID.

Writes combined outputs to:
  outputs/.../lagged_cka_fixed_rerun/combined_*/

Usage:
    conda run --no-capture-output -n yael_env \
        python scripts/combine_and_analyze_fixed.py \
            --exp_dir outputs/EXP_tsfm_full_run_all206_20260530_172932
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

VALID_METHODS_ORDER = [
    "target_only",
    "all_features_206",
    "random_k",
    "Pearson",
    "SparseLinear_L1",
    "RandomForest",
    "Mean_CKA_L8",
    "Lagged_CKA_L8_fixed",
    "Lagged_CKA_L8_INVALID",   # kept only for diagnostic reference
]

METHOD_COLORS = {
    "target_only":            "#aaaaaa",
    "all_features_206":       "#bbbbbb",
    "random_k":               "#ccbb44",
    "Pearson":                "#4477aa",
    "SparseLinear_L1":        "#66ccee",
    "RandomForest":           "#228833",
    "Mean_CKA_L8":            "#ee6677",
    "Lagged_CKA_L8_fixed":    "#cc0099",
    "Lagged_CKA_L8_INVALID":  "#dddddd",
}


# ---------------------------------------------------------------------------
# Load + merge JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_combined(exp_dir: Path) -> pd.DataFrame:
    orig_path  = exp_dir / "results" / "results_incremental.jsonl"
    fixed_path = exp_dir / "lagged_cka_fixed_rerun" / "results_incremental_fixed.jsonl"

    if not orig_path.exists():
        raise FileNotFoundError(f"Original JSONL not found: {orig_path}")
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed JSONL not found: {fixed_path}")

    orig_rows  = load_jsonl(orig_path)
    fixed_rows = load_jsonl(fixed_path)

    # Rename old Lagged_CKA_L8 → Lagged_CKA_L8_INVALID
    for r in orig_rows:
        if r.get("method") == "Lagged_CKA_L8":
            r["method"] = "Lagged_CKA_L8_INVALID"

    # Drop replicated baseline rows (they existed in original for K-axis plots)
    orig_clean = [r for r in orig_rows if not r.get("is_replicated_baseline", False)]

    all_rows = orig_clean + fixed_rows
    df = pd.DataFrame(all_rows)
    print(f"Combined: {len(orig_clean)} original (non-replicated) + {len(fixed_rows)} fixed = {len(all_rows)} rows")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_fixed(df: pd.DataFrame) -> None:
    fixed = df[df["method"] == "Lagged_CKA_L8_fixed"]
    n_rows    = len(fixed)
    n_targets = fixed["target_sensor_id"].nunique()
    k_vals    = sorted(fixed["top_k"].unique())
    any_nan   = fixed["RMSE"].isna().any() or fixed["MAE"].isna().any()

    print(f"\n[validation] Lagged_CKA_L8_fixed:")
    print(f"  rows={n_rows} (expected 30)")
    print(f"  targets={n_targets} (expected 10)")
    print(f"  K values={k_vals} (expected [5,10,20])")
    print(f"  any NaN RMSE/MAE: {any_nan}")
    print(f"  feature_universe_size unique: {fixed['feature_universe_size'].unique()}")

    errors = []
    if n_rows != 30:
        errors.append(f"row count {n_rows} != 30")
    if n_targets != 10:
        errors.append(f"target count {n_targets} != 10")
    if k_vals != [5, 10, 20]:
        errors.append(f"K values {k_vals} != [5,10,20]")
    if any_nan:
        errors.append("NaN in RMSE/MAE")
    if not all(v == 206 for v in fixed["feature_universe_size"].dropna()):
        errors.append("feature_universe_size != 206")

    # Check selected_sensors are not trivial
    for _, row in fixed[fixed["top_k"] == 5].iterrows():
        sel = json.loads(row["selected_sensors"])
        if sel == list(range(5)):
            errors.append(f"trivial selected_sensors for target {row['target_sensor_id']}")

    if errors:
        print(f"  ERRORS: {errors}")
        raise ValueError(f"Validation failed: {errors}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Aggregate random_k repeats
    rk_mask = df["method"] == "random_k"
    rk_agg = (
        df[rk_mask]
        .groupby(["target_sensor_id", "top_k"])[["RMSE", "MAE"]]
        .mean().reset_index()
    )
    rk_agg["method"]      = "random_k"
    rk_agg["target_role"] = (
        df[rk_mask].groupby(["target_sensor_id", "top_k"])["target_role"]
        .first().reset_index()["target_role"]
    )
    df = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    to_rmse = df[df["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = df[df["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()
    pe_rmse_k = {}  # {(target, k): rmse}
    for _, r in df[df["method"] == "Pearson"].iterrows():
        pe_rmse_k[(str(r["target_sensor_id"]), int(r["top_k"]))] = r["RMSE"]
    mc_rmse_k = {}
    for _, r in df[df["method"] == "Mean_CKA_L8"].iterrows():
        mc_rmse_k[(str(r["target_sensor_id"]), int(r["top_k"]))] = r["RMSE"]
    sl_rmse_k = {}
    for _, r in df[df["method"] == "SparseLinear_L1"].iterrows():
        sl_rmse_k[(str(r["target_sensor_id"]), int(r["top_k"]))] = r["RMSE"]

    # Replicate target_only and all_features_206 to all K values
    to_df = df[df["method"] == "target_only"].copy()
    af_df = df[df["method"] == "all_features_206"].copy()
    fs_df = df[~df["method"].isin(["target_only", "all_features_206"])].copy()

    rows_all = []
    for top_k in [5, 10, 20]:
        for _, r in to_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        for _, r in af_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        rows_all.extend(fs_df[fs_df["top_k"] == top_k].to_dict("records"))

    df_long = pd.DataFrame(rows_all)

    def _agg(sub):
        rmse  = sub["RMSE"].values
        mae   = sub["MAE"].values
        tgts  = sub["target_sensor_id"].values.astype(str)
        to_r  = np.array([to_rmse.get(t, np.nan) for t in tgts])
        af_r  = np.array([af_rmse.get(t, np.nan) for t in tgts])
        top_k_val = sub["top_k"].iloc[0] if "top_k" in sub.columns else np.nan
        try:
            top_k_int = int(top_k_val)
        except (ValueError, TypeError):
            top_k_int = None
        pe_r  = np.array([pe_rmse_k.get((t, top_k_int), np.nan) if top_k_int else np.nan for t in tgts])
        mc_r  = np.array([mc_rmse_k.get((t, top_k_int), np.nan) if top_k_int else np.nan for t in tgts])
        sl_r  = np.array([sl_rmse_k.get((t, top_k_int), np.nan) if top_k_int else np.nan for t in tgts])
        return pd.Series({
            "mean_RMSE":          np.mean(rmse),
            "median_RMSE":        np.median(rmse),
            "std_RMSE":           np.std(rmse),
            "mean_MAE":           np.mean(mae),
            "median_MAE":         np.median(mae),
            "pct_impr_vs_to":     np.nanmean((to_r - rmse) / to_r * 100),
            "pct_impr_vs_af":     np.nanmean((af_r - rmse) / af_r * 100),
            "win_vs_to":          int(np.sum(rmse < to_r)),
            "win_vs_af":          int(np.sum(rmse < af_r)),
            "win_vs_Pearson":     int(np.nansum(rmse < pe_r)),
            "win_vs_Mean_CKA":    int(np.nansum(rmse < mc_r)),
            "win_vs_SparseLinear":int(np.nansum(rmse < sl_r)),
            "n_targets":          len(sub),
        })

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", FutureWarning)
        summary = df_long.groupby(["method", "top_k"]).apply(_agg).reset_index()

    # Best-method count (excluding baselines and INVALID)
    valid_fs = df_long[~df_long["method"].isin([
        "target_only", "all_features_206", "Lagged_CKA_L8_INVALID"
    ])].copy()
    best_rows = valid_fs.loc[valid_fs.groupby(["target_sensor_id", "top_k"])["RMSE"].idxmin()]
    best_cnt  = best_rows.groupby(["method", "top_k"]).size().reset_index(name="best_method_count")
    summary   = summary.merge(best_cnt, on=["method", "top_k"], how="left")
    summary["best_method_count"] = summary["best_method_count"].fillna(0).astype(int)

    return summary


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------

def compute_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    df = df.copy()
    rk_mask = df["method"] == "random_k"
    rk_agg = (
        df[rk_mask].groupby(["target_sensor_id", "top_k"])["RMSE"]
        .mean().reset_index()
    )
    rk_agg["method"] = "random_k"
    df = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    to_rmse = df[df["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = df[df["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()

    COMPARISONS = [
        ("Lagged_CKA_L8_fixed", "target_only"),
        ("Lagged_CKA_L8_fixed", "all_features_206"),
        ("Lagged_CKA_L8_fixed", "random_k"),
        ("Lagged_CKA_L8_fixed", "Pearson"),
        ("Lagged_CKA_L8_fixed", "Mean_CKA_L8"),
        ("Lagged_CKA_L8_fixed", "SparseLinear_L1"),
        ("Mean_CKA_L8",         "Pearson"),
        ("Mean_CKA_L8",         "random_k"),
        ("Mean_CKA_L8",         "all_features_206"),
    ]

    rows = []
    for top_k in [5, 10, 20]:
        df_k = df[df["top_k"] == top_k]

        for method_a, method_b in COMPARISONS:
            rows_a = df_k[df_k["method"] == method_a][["target_sensor_id", "RMSE"]]
            if method_b == "target_only":
                rows_b = pd.DataFrame([{"target_sensor_id": t, "RMSE": v}
                                        for t, v in to_rmse.items()])
            elif method_b == "all_features_206":
                rows_b = pd.DataFrame([{"target_sensor_id": t, "RMSE": v}
                                        for t, v in af_rmse.items()])
            else:
                rows_b = df_k[df_k["method"] == method_b][["target_sensor_id", "RMSE"]]

            merged = rows_a.merge(rows_b, on="target_sensor_id", suffixes=("_a", "_b"))
            if len(merged) < 5:
                continue

            a = merged["RMSE_a"].values
            b = merged["RMSE_b"].values
            diffs = a - b

            if np.all(diffs == 0):
                p_val, stat = 1.0, np.nan
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        stat, p_val = wilcoxon(diffs, alternative="two-sided")
                except Exception:
                    p_val, stat = np.nan, np.nan

            rows.append({
                "comparison":              f"{method_a}_vs_{method_b}",
                "top_k":                   top_k,
                "n_pairs":                 len(merged),
                "mean_RMSE_A":             float(np.mean(a)),
                "mean_RMSE_B":             float(np.mean(b)),
                "mean_diff_A_minus_B":     float(np.mean(diffs)),
                "median_diff_A_minus_B":   float(np.median(diffs)),
                "win_count_A_better":      int(np.sum(a < b)),
                "win_count_B_better":      int(np.sum(b < a)),
                "W_statistic":             float(stat) if np.isfinite(stat) else np.nan,
                "p_value":                 float(p_val),
                "significant_at_05":       bool(p_val < 0.05) if np.isfinite(p_val) else False,
                "note":                    "EXPLORATORY. n=10. Low power.",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Role summary
# ---------------------------------------------------------------------------

def compute_role_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rk_mask = df["method"] == "random_k"
    rk_agg = (
        df[rk_mask].groupby(["target_sensor_id", "top_k", "target_role"])[["RMSE", "MAE"]]
        .mean().reset_index()
    )
    rk_agg["method"] = "random_k"
    df = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    methods_show = [m for m in VALID_METHODS_ORDER if m != "Lagged_CKA_L8_INVALID"]
    to_df = df[df["method"] == "target_only"].copy()
    af_df = df[df["method"] == "all_features_206"].copy()
    fs_df = df[~df["method"].isin(["target_only", "all_features_206"])].copy()

    rows_all = []
    for top_k in [5, 10, 20]:
        for _, r in to_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        for _, r in af_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        rows_all.extend(fs_df[fs_df["top_k"] == top_k].to_dict("records"))

    df_long = pd.DataFrame(rows_all)
    df_long = df_long[df_long["method"].isin(methods_show)]
    return (
        df_long.groupby(["target_role", "method", "top_k"])[["RMSE", "MAE"]]
        .agg(mean_RMSE=("RMSE", "mean"), median_RMSE=("RMSE", "median"), n=("RMSE", "count"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plots_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Exclude INVALID from primary plots
    valid_methods = [m for m in VALID_METHODS_ORDER if m != "Lagged_CKA_L8_INVALID"]

    for metric, label, fname in [
        ("mean_RMSE", "Mean RMSE", "combined_bar_rmse.png"),
        ("mean_MAE",  "Mean MAE",  "combined_bar_mae.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[(summary["top_k"] == k) & summary["method"].isin(valid_methods)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=valid_methods, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
            ax.bar(range(len(sub)), sub[metric], color=colors)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"K={k}")
            ax.set_ylabel(label if k == 5 else "")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{label} by Method (exploratory; n=10 targets; Lagged_CKA_L8_INVALID excluded)")
        plt.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    for vs_col, vs_label, fname in [
        ("pct_impr_vs_to", "% improvement vs target_only",     "combined_pct_vs_target_only.png"),
        ("pct_impr_vs_af", "% improvement vs all_features_206","combined_pct_vs_all_features.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[(summary["top_k"] == k) & summary["method"].isin(valid_methods)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=valid_methods, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
            bars = ax.bar(range(len(sub)), sub[vs_col], color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"K={k}")
            ax.set_ylabel(vs_label if k == 5 else "")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{vs_label} (exploratory)")
        plt.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Win counts
    for win_col, vs_label, fname in [
        ("win_vs_to", "Win count vs target_only",    "combined_win_vs_target_only.png"),
        ("win_vs_af", "Win count vs all_features_206","combined_win_vs_all_features.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[(summary["top_k"] == k) & summary["method"].isin(valid_methods)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=valid_methods, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
            ax.bar(range(len(sub)), sub[win_col], color=colors)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"K={k}")
            ax.set_ylabel(vs_label if k == 5 else "")
            ax.set_ylim(0, 11)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{vs_label} (exploratory; n=10 targets)")
        plt.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Score behavior summary (from per-target score CSVs)
# ---------------------------------------------------------------------------

def summarize_score_behavior(rerun_dir: Path, sensor_id_map: dict) -> None:
    from collections import Counter
    score_files = sorted(rerun_dir.glob("lagged_cka_fixed_scores_*.csv"))
    if not score_files:
        print("  No per-target score files found.")
        return

    print(f"\n{'='*70}")
    print("SCORE BEHAVIOR PER TARGET")
    print(f"{'='*70}")
    print(f"{'target':>10}  {'role':>22}  {'min':>7}  {'max':>7}  {'std':>7}  "
          f"{'lag=0':>7}  {'lag=±9':>8}  {'lag_other':>9}")
    seen_targets = set()
    for f in score_files:
        sdf = pd.read_csv(f)
        target = sdf["target_sensor_id"].iloc[0]
        if target in seen_targets:
            continue
        seen_targets.add(target)
        role   = sdf["target_role"].iloc[0]
        scores = sdf["score"].values
        lags   = sdf["best_lag"].values
        lag_counts  = Counter(lags.tolist())
        n_lag0      = lag_counts.get(0, 0)
        n_lag_bnd   = lag_counts.get(9, 0) + lag_counts.get(-9, 0)
        n_lag_other = len(lags) - n_lag0 - n_lag_bnd
        print(f"  {target:>10}  {role:>22}  {scores.min():>7.4f}  {scores.max():>7.4f}"
              f"  {scores.std():>7.4f}  {n_lag0:>7d}  {n_lag_bnd:>8d}  {n_lag_other:>9d}")


# ---------------------------------------------------------------------------
# Lagged_Pearson comparison (from score comparison CSV if available)
# ---------------------------------------------------------------------------

def summarize_pearson_comparison(rerun_dir: Path) -> None:
    cmp_path = rerun_dir / "lagged_sanity_score_comparison.csv"
    if not cmp_path.exists():
        print("  lagged_sanity_score_comparison.csv not found — skipping comparison")
        return

    cdf = pd.read_csv(cmp_path)
    from scipy.stats import spearmanr

    def overlap_k(a: list, b: list, k: int) -> int:
        return len(set(a[:k]) & set(b[:k]))

    lc_order = cdf.sort_values("Lagged_CKA_fixed_rank")["candidate_df_col"].tolist()
    lp_order = cdf.sort_values("Lagged_Pearson_rank")["candidate_df_col"].tolist()

    lc_scores = cdf.sort_values("candidate_df_col")["Lagged_CKA_fixed_score"].values
    lp_scores = cdf.sort_values("candidate_df_col")["Lagged_Pearson_score"].values

    rho, _ = spearmanr(lc_scores, lp_scores)

    print(f"\n  Lagged_CKA_fixed vs Lagged_Pearson (target 717469):")
    print(f"    Spearman rho = {rho:.4f}")
    print(f"    Overlap@5:  {overlap_k(lc_order, lp_order, 5)}/5")
    print(f"    Overlap@10: {overlap_k(lc_order, lp_order, 10)}/10")
    print(f"    Overlap@20: {overlap_k(lc_order, lp_order, 20)}/20")


# ---------------------------------------------------------------------------
# Print critical summary
# ---------------------------------------------------------------------------

def print_critical_summary(df: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame,
                            rerun_dir: Path) -> None:
    print(f"\n{'='*78}")
    print("COMBINED CRITICAL RESULT SUMMARY")
    print("(exploratory; n=10 targets; 91.7% window overlap; Lagged_CKA_L8_INVALID excluded)")
    print(f"{'='*78}")

    valid_methods_show = [
        "target_only", "all_features_206", "random_k",
        "Pearson", "SparseLinear_L1", "RandomForest", "Mean_CKA_L8",
        "Lagged_CKA_L8_fixed",
    ]

    for k in [5, 10, 20]:
        print(f"\n--- K={k} ---")
        hdr = (f"{'Method':<24} {'mean_RMSE':>10} {'med_RMSE':>10} {'std':>7} "
               f"{'mean_MAE':>10} {'%vs_TO':>8} {'%vs_AF':>8} "
               f"{'win_TO':>7} {'win_AF':>7} {'win_PE':>7} {'win_MC':>7} {'best':>5}")
        print(hdr)
        for m in valid_methods_show:
            row = summary[(summary["method"] == m) & (summary["top_k"] == k)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"  {m:<24} {r['mean_RMSE']:>10.4f} {r['median_RMSE']:>10.4f} "
                  f"{r['std_RMSE']:>7.4f} {r['mean_MAE']:>10.4f} "
                  f"{r.get('pct_impr_vs_to', float('nan')):>8.2f} "
                  f"{r.get('pct_impr_vs_af', float('nan')):>8.2f} "
                  f"{int(r.get('win_vs_to', 0)):>7} "
                  f"{int(r.get('win_vs_af', 0)):>7} "
                  f"{int(r.get('win_vs_Pearson', 0)):>7} "
                  f"{int(r.get('win_vs_Mean_CKA', 0)):>7} "
                  f"{int(r.get('best_method_count', 0)):>5}")

    if not tests.empty:
        print(f"\n--- PAIRWISE WILCOXON (Lagged_CKA_L8_fixed; exploratory; n=10) ---")
        print(f"{'Comparison':<45} {'K':>4} {'diff':>10} {'win_A':>6} {'win_B':>6} {'p':>8}")
        relevant = tests[tests["comparison"].str.startswith("Lagged_CKA_L8_fixed")]
        for _, r in relevant.sort_values(["comparison", "top_k"]).iterrows():
            print(f"  {r['comparison']:<43} K={int(r['top_k']):<4} "
                  f"{r['mean_diff_A_minus_B']:>+10.4f} "
                  f"{int(r['win_count_A_better']):>6} "
                  f"{int(r['win_count_B_better']):>6} "
                  f"{r['p_value']:>8.4f}")

    summarize_score_behavior(rerun_dir, {})
    summarize_pearson_comparison(rerun_dir)

    print(f"\n--- NOTE: Lagged_CKA_L8_INVALID ---")
    invalid = df[df["method"] == "Lagged_CKA_L8_INVALID"]
    if not invalid.empty:
        rmse_i = invalid.groupby("top_k")["RMSE"].mean()
        print(f"  Retained for reference only. INVALID (all scores=1.0, selected [0..K-1]).")
        for k, v in rmse_i.items():
            print(f"    K={k}: mean_RMSE={v:.4f} (meaningless — degenerate selection)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    args = parser.parse_args()

    exp_dir   = Path(args.exp_dir)
    rerun_dir = exp_dir / "lagged_cka_fixed_rerun"
    out_dir   = rerun_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    # ---- Load and merge ----
    df = load_combined(exp_dir)

    # ---- Validate fixed rows ----
    validate_fixed(df)

    # ---- Save combined CSV ----
    csv_path = out_dir / "combined_tsfm_downstream_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ---- Rebuild selected_sensors_by_method.json ----
    sel_map = {}
    fs_methods = ["Pearson", "SparseLinear_L1", "RandomForest",
                  "Mean_CKA_L8", "Lagged_CKA_L8_fixed"]
    for m in fs_methods:
        sub = df[df["method"] == m]
        for _, row in sub.iterrows():
            tgt = str(row["target_sensor_id"])
            k   = str(int(row["top_k"]))
            sel = json.loads(row["selected_sensors"]) if isinstance(row["selected_sensors"], str) else []
            sel_map.setdefault(tgt, {}).setdefault(k, {})[m] = sel
    with open(out_dir / "combined_selected_sensors_by_method.json", "w") as f:
        json.dump(sel_map, f, indent=2)

    # ---- Statistical summary ----
    summary = compute_summary(df)
    sum_path = out_dir / "combined_statistical_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"  Saved: {sum_path}")

    # ---- Wilcoxon tests ----
    try:
        tests = compute_wilcoxon(df)
        test_path = out_dir / "combined_statistical_tests.csv"
        with open(test_path, "w") as f:
            f.write("# EXPLORATORY: n=10 targets, low power\n")
            tests.to_csv(f, index=False)
        print(f"  Saved: {test_path}")
    except ImportError:
        print("  [SKIP] scipy not available — statistical_tests.csv not generated")
        tests = pd.DataFrame()

    # ---- Plots ----
    try:
        make_plots(df, plots_dir, summary)
    except Exception as e:
        print(f"  [warn] plot error: {e}")

    # ---- Role summary ----
    role_sum = compute_role_summary(df)
    role_path = out_dir / "combined_rolewise_summary.csv"
    role_sum.to_csv(role_path, index=False)
    print(f"  Saved: {role_path}")

    # ---- Print critical summary ----
    print_critical_summary(df, summary, tests, rerun_dir)


if __name__ == "__main__":
    main()
