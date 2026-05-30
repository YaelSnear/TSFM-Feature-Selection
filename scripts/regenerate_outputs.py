"""Regenerate all final outputs from results_incremental.jsonl.

Source of truth: results_incremental.jsonl (FULL RUN only).
Does NOT run any experiments. Only reads the JSONL and regenerates:
  - results/tsfm_downstream_results.csv
  - results/statistical_summary.csv
  - results/statistical_tests.csv
  - results/selected_sensors_by_method.json
  - plots/

Usage:
    python scripts/regenerate_outputs.py \
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

EXPECTED_METHODS = {
    "target_only",
    "all_features_206",
    "random_k",
    "Pearson",
    "SparseLinear_L1",
    "RandomForest",
    "Mean_CKA_L8",
    "Lagged_CKA_L8",
}
STALE_NAMES = {"all_candidates", "raw", "whitened", "Lagged_CKA_raw_L10", "Lagged_CKA_whitened_L10"}


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Keep only non-replicated rows
    if "is_replicated_baseline" in df.columns:
        df = df[df["is_replicated_baseline"] != True].copy()
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    print("\n=== VALIDATION ===")
    methods = set(df["method"].unique())
    print(f"Methods found: {sorted(methods)}")
    stale = methods & STALE_NAMES
    assert not stale, f"STALE method names found: {stale}"
    unexpected = methods - EXPECTED_METHODS
    if unexpected:
        print(f"  [WARN] Unexpected methods (not in expected set): {unexpected}")

    targets = df["target_sensor_id"].unique()
    n_targets = len(targets)
    print(f"Targets: N={n_targets} -> {sorted(targets)}")
    assert n_targets == 10, f"Expected 10 targets, got {n_targets}"

    univ_sizes = df["feature_universe_size"].unique()
    print(f"feature_universe_size: {univ_sizes}")
    assert all(v == 206 for v in univ_sizes), "feature_universe_size != 206"

    k_vals = sorted(df[df["method"] == "Pearson"]["top_k"].unique())
    print(f"K values (Pearson): {k_vals}")
    assert k_vals == [5, 10, 20], f"Expected K=[5,10,20], got {k_vals}"

    layers = df[df["method"].str.startswith("Mean_CKA", na=False)]["layer"].unique()
    print(f"CKA layer: {layers}")
    assert all(str(l) == "8" for l in layers), f"Expected layer 8, got {layers}"

    # target_only: 1 per target
    to = df[df["method"] == "target_only"]
    assert len(to) == n_targets, f"target_only count={len(to)}, expected {n_targets}"

    # all_features_206: 1 per target
    af = df[df["method"] == "all_features_206"]
    assert len(af) == n_targets, f"all_features_206 count={len(af)}, expected {n_targets}"

    # random_k: 3 repeats per target per K
    rk = df[df["method"] == "random_k"]
    for k in [5, 10, 20]:
        for tgt in targets:
            cnt = len(rk[(rk["top_k"] == k) & (rk["target_sensor_id"] == tgt)])
            assert cnt == 3, f"random_k target={tgt} K={k}: expected 3 repeats, got {cnt}"

    # Non-random FS: 1 per target per K
    non_rnd = ["Pearson", "SparseLinear_L1", "RandomForest", "Mean_CKA_L8", "Lagged_CKA_L8"]
    for m in non_rnd:
        for k in [5, 10, 20]:
            cnt = len(df[(df["method"] == m) & (df["top_k"] == k)])
            assert cnt == n_targets, f"{m} K={k}: expected {n_targets}, got {cnt}"

    nan_rmse = df["RMSE"].isna().sum()
    nan_mae = df["MAE"].isna().sum()
    assert nan_rmse == 0, f"NaN RMSE count: {nan_rmse}"
    assert nan_mae == 0, f"NaN MAE count: {nan_mae}"

    print("VALIDATION PASSED\n")


# ---------------------------------------------------------------------------
# Rebuild selected_sensors_by_method.json
# ---------------------------------------------------------------------------

def rebuild_selected_sensors(df: pd.DataFrame, out_path: Path) -> None:
    fs_methods = [m for m in EXPECTED_METHODS
                  if m not in ("target_only", "all_features_206", "random_k")]
    result: dict = {}
    for m in fs_methods:
        sub = df[df["method"] == m]
        for _, row in sub.iterrows():
            tgt = str(row["target_sensor_id"])
            k   = str(int(row["top_k"]))
            sel = json.loads(row["selected_sensors"]) if isinstance(row["selected_sensors"], str) else row["selected_sensors"]
            result.setdefault(tgt, {}).setdefault(k, {})[m] = sel
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (method, top_k) — random_k averaged over repeats first."""
    df = df.copy()

    # Get target_only and all_features_206 RMSE per target
    to_rmse = df[df["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = df[df["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()

    # Aggregate random_k repeats per (target, top_k)
    rk_mask = df["method"] == "random_k"
    rk_agg = (
        df[rk_mask]
        .groupby(["target_sensor_id", "top_k"])[["RMSE", "MAE"]]
        .mean()
        .reset_index()
    )
    rk_agg["method"] = "random_k"
    rk_agg["target_role"] = df[rk_mask].groupby(["target_sensor_id", "top_k"])["target_role"].first().reset_index()["target_role"]
    df_clean = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    # For target_only and all_features_206, replicate to each K for summary
    to_df = df_clean[df_clean["method"] == "target_only"].copy()
    af_df = df_clean[df_clean["method"] == "all_features_206"].copy()
    fs_df = df_clean[~df_clean["method"].isin(["target_only", "all_features_206"])].copy()

    rows = []
    for top_k in [5, 10, 20]:
        fs_k = fs_df[fs_df["top_k"] == top_k]

        # Add target_only and all_features_206 as reference rows
        for _, r in to_df.iterrows():
            rows.append({**r, "top_k": top_k})
        for _, r in af_df.iterrows():
            rows.append({**r, "top_k": top_k})
        for _, r in fs_k.iterrows():
            rows.append(dict(r))

    df_long = pd.DataFrame(rows)

    def _agg(sub):
        rmse = sub["RMSE"].values
        mae  = sub["MAE"].values
        tgts = sub["target_sensor_id"].values
        to_r = np.array([to_rmse.get(t, np.nan) for t in tgts])
        af_r = np.array([af_rmse.get(t, np.nan) for t in tgts])
        pct_vs_to = np.nanmean((to_r - rmse) / to_r * 100)
        pct_vs_af = np.nanmean((af_r - rmse) / af_r * 100)
        n_win_to  = int(np.sum(rmse < to_r))
        n_win_af  = int(np.sum(rmse < af_r))
        return pd.Series({
            "mean_RMSE":        np.mean(rmse),
            "median_RMSE":      np.median(rmse),
            "std_RMSE":         np.std(rmse),
            "mean_MAE":         np.mean(mae),
            "median_MAE":       np.median(mae),
            "pct_impr_vs_to":   pct_vs_to,
            "pct_impr_vs_af":   pct_vs_af,
            "win_vs_to":        n_win_to,
            "win_vs_af":        n_win_af,
            "n_targets":        len(sub),
        })

    summary = df_long.groupby(["method", "top_k"]).apply(_agg).reset_index()

    # Best-method count: for each (target, top_k), which method (excluding baselines) wins?
    fs_only = df_long[~df_long["method"].isin(["target_only", "all_features_206"])].copy()
    best_rows = fs_only.loc[fs_only.groupby(["target_sensor_id", "top_k"])["RMSE"].idxmin()]
    best_cnt = best_rows.groupby(["method", "top_k"]).size().reset_index(name="best_method_count")
    summary = summary.merge(best_cnt, on=["method", "top_k"], how="left")
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
    df_clean = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    to_rmse = df[df["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = df[df["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()

    COMPARISONS = [
        ("Mean_CKA_L8",   "Pearson"),
        ("Mean_CKA_L8",   "random_k"),
        ("Mean_CKA_L8",   "RandomForest"),
        ("Mean_CKA_L8",   "SparseLinear_L1"),
        ("Mean_CKA_L8",   "all_features_206"),
        ("Lagged_CKA_L8", "Pearson"),
        ("Lagged_CKA_L8", "random_k"),
        ("Lagged_CKA_L8", "all_features_206"),
    ]

    rows = []
    for top_k in [5, 10, 20]:
        df_k = df_clean[df_clean["top_k"] == top_k]

        for method_a, method_b in COMPARISONS:
            # Get method_a rows
            rows_a = df_k[df_k["method"] == method_a][["target_sensor_id", "RMSE"]]
            # Get method_b rows (target_only / all_features at their natural K)
            if method_b == "all_features_206":
                rows_b = pd.DataFrame([
                    {"target_sensor_id": t, "RMSE": v}
                    for t, v in af_rmse.items()
                ])
            elif method_b == "target_only":
                rows_b = pd.DataFrame([
                    {"target_sensor_id": t, "RMSE": v}
                    for t, v in to_rmse.items()
                ])
            else:
                rows_b = df_k[df_k["method"] == method_b][["target_sensor_id", "RMSE"]]

            merged = rows_a.merge(rows_b, on="target_sensor_id", suffixes=("_a", "_b"))
            if len(merged) < 5:
                continue

            a = merged["RMSE_a"].values
            b = merged["RMSE_b"].values
            diffs = a - b

            if np.all(diffs == 0):
                p_val = 1.0
                stat  = np.nan
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        stat, p_val = wilcoxon(diffs, alternative="two-sided")
                except Exception:
                    p_val = np.nan
                    stat  = np.nan

            n = len(a)
            rows.append({
                "comparison":         f"{method_a}_vs_{method_b}",
                "top_k":              top_k,
                "n_pairs":            n,
                "mean_RMSE_A":        float(np.mean(a)),
                "mean_RMSE_B":        float(np.mean(b)),
                "mean_diff_A_minus_B": float(np.mean(diffs)),
                "median_diff_A_minus_B": float(np.median(diffs)),
                "win_count_A_better": int(np.sum(a < b)),
                "win_count_B_better": int(np.sum(b < a)),
                "W_statistic":        float(stat) if stat == stat else np.nan,
                "p_value":            float(p_val),
                "significant_at_05":  bool(p_val < 0.05) if p_val == p_val else False,
                "note":               "EXPLORATORY. n=10. Low power.",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Role-based summary
# ---------------------------------------------------------------------------

def compute_role_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rk_mask = df["method"] == "random_k"
    rk_agg = (
        df[rk_mask].groupby(["target_sensor_id", "top_k", "target_role"])[["RMSE", "MAE"]]
        .mean().reset_index()
    )
    rk_agg["method"] = "random_k"
    df_clean = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    methods_of_interest = [
        "target_only", "all_features_206", "Pearson", "RandomForest",
        "SparseLinear_L1", "Mean_CKA_L8", "Lagged_CKA_L8", "random_k",
    ]

    # Replicate target_only and all_features_206 to all K for role analysis
    to_df = df_clean[df_clean["method"] == "target_only"].copy()
    af_df = df_clean[df_clean["method"] == "all_features_206"].copy()
    fs_df = df_clean[~df_clean["method"].isin(["target_only", "all_features_206"])].copy()

    rows_all = []
    for top_k in [5, 10, 20]:
        for _, r in to_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        for _, r in af_df.iterrows():
            rows_all.append({**r.to_dict(), "top_k": top_k})
        rows_all.extend(fs_df[fs_df["top_k"] == top_k].to_dict("records"))

    df_long = pd.DataFrame(rows_all)
    df_long = df_long[df_long["method"].isin(methods_of_interest)]

    summary = (
        df_long.groupby(["target_role", "method", "top_k"])[["RMSE", "MAE"]]
        .agg(mean_RMSE=("RMSE", "mean"), median_RMSE=("RMSE", "median"), n=("RMSE", "count"))
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plots_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METHOD_ORDER = [
        "target_only", "random_k", "Pearson", "SparseLinear_L1",
        "RandomForest", "Mean_CKA_L8", "Lagged_CKA_L8", "all_features_206",
    ]
    METHOD_COLORS = {
        "target_only":       "#aaaaaa",
        "random_k":          "#ccbb44",
        "Pearson":           "#4477aa",
        "SparseLinear_L1":   "#66ccee",
        "RandomForest":      "#228833",
        "Mean_CKA_L8":       "#ee6677",
        "Lagged_CKA_L8":     "#aa3377",
        "all_features_206":  "#bbbbbb",
    }

    for metric, label in [("mean_RMSE", "Mean RMSE"), ("mean_MAE", "Mean MAE")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[summary["top_k"] == k]
            sub = sub[sub["method"].isin(METHOD_ORDER)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888888") for m in sub["method"]]
            ax.bar(range(len(sub)), sub[metric], color=colors)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"K={k}")
            ax.set_ylabel(label if k == 5 else "")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{label} by Method (exploratory)")
        plt.tight_layout()
        fname = "bar_rmse_by_method.png" if "RMSE" in metric else "bar_mae_by_method.png"
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {plots_dir / fname}")

    # Improvement vs target_only
    for vs_col, vs_label, fname in [
        ("pct_impr_vs_to",  "% improvement vs target_only",    "pct_improvement_vs_target_only.png"),
        ("pct_impr_vs_af",  "% improvement vs all_features_206", "pct_improvement_vs_all_features.png"),
    ]:
        if vs_col not in summary.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[(summary["top_k"] == k) & summary["method"].isin(METHOD_ORDER)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888888") for m in sub["method"]]
            bars = ax.bar(range(len(sub)), sub[vs_col], color=colors)
            for bar, val in zip(bars, sub[vs_col]):
                if val == val:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f"{val:.1f}%", ha="center", va="bottom", fontsize=6)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"K={k}")
            ax.set_ylabel(vs_label if k == 5 else "")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{vs_label} (exploratory)")
        plt.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {plots_dir / fname}")

    # Win count
    win_col = "win_vs_to"
    if win_col in summary.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary[(summary["top_k"] == k) & summary["method"].isin(METHOD_ORDER)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888888") for m in sub["method"]]
            ax.bar(range(len(sub)), sub[win_col], color=colors)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"K={k}")
            ax.set_ylabel("Win count vs target_only" if k == 5 else "")
            ax.set_ylim(0, 11)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Win count vs target_only (exploratory, n=10 targets)")
        plt.tight_layout()
        fig.savefig(plots_dir / "win_count_by_method.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {plots_dir / 'win_count_by_method.png'}")

    # Role-wise RMSE
    role_summary = compute_role_summary(df)
    roles = sorted(role_summary["target_role"].unique())
    k_plot = 10  # show K=10 only for role plot
    sub_r = role_summary[(role_summary["top_k"] == k_plot) & role_summary["method"].isin(METHOD_ORDER)]
    if not sub_r.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(roles))
        width = 0.9 / len(METHOD_ORDER)
        for i, m in enumerate(METHOD_ORDER):
            vals = [sub_r[(sub_r["target_role"] == r) & (sub_r["method"] == m)]["mean_RMSE"].values
                    for r in roles]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax.bar(x + i * width, vals, width, label=m, color=METHOD_COLORS.get(m, "#888"))
        ax.set_xticks(x + width * len(METHOD_ORDER) / 2)
        ax.set_xticklabels(roles, rotation=20, ha="right")
        ax.set_ylabel("Mean RMSE")
        ax.set_title(f"Role-wise Mean RMSE at K={k_plot} (exploratory; each role = 2 targets)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / "rolewise_rmse.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {plots_dir / 'rolewise_rmse.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(exp_dir: str) -> None:
    exp_path  = Path(exp_dir)
    res_dir   = exp_path / "results"
    plots_dir = exp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    jsonl_path = res_dir / "results_incremental.jsonl"
    assert jsonl_path.exists(), f"JSONL not found: {jsonl_path}"

    print(f"Loading {jsonl_path} ...")
    df = load_jsonl(jsonl_path)
    print(f"  Loaded {len(df)} rows")

    validate(df)

    # 1. tsfm_downstream_results.csv (from JSONL only; no stale data)
    csv_path = res_dir / "tsfm_downstream_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path} ({len(df)} rows)")

    # 2. selected_sensors_by_method.json
    sel_path = res_dir / "selected_sensors_by_method.json"
    rebuild_selected_sensors(df, sel_path)

    # 3. statistical_summary.csv
    summary = compute_summary(df)
    sum_path = res_dir / "statistical_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"  saved {sum_path} ({len(summary)} rows)")

    # 4. statistical_tests.csv
    try:
        tests = compute_wilcoxon(df)
        test_path = res_dir / "statistical_tests.csv"
        with open(test_path, "w") as f:
            f.write("# EXPLORATORY: n=10 targets, low power — do not over-interpret\n")
            tests.to_csv(f, index=False)
        print(f"  saved {test_path} ({len(tests)} rows)")
    except ImportError:
        print("  [SKIP] scipy not available — statistical_tests.csv not generated")
        tests = pd.DataFrame()

    # 5. Plots
    print("Generating plots ...")
    make_plots(df, plots_dir, summary)

    # 6. Print summary tables
    print_critical_summary(df, summary, tests)


# ---------------------------------------------------------------------------
# Critical summary printer
# ---------------------------------------------------------------------------

def print_critical_summary(df: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("CRITICAL RESULT SUMMARY (exploratory; n=10 targets; 91.7% window overlap)")
    print("="*80)

    rk_agg = (
        df[df["method"] == "random_k"]
        .groupby(["target_sensor_id", "top_k"])["RMSE"].mean().reset_index()
    )
    rk_agg["method"] = "random_k"

    to_rmse = df[df["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = df[df["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()

    print(f"\ntarget_only  mean RMSE: {np.mean(list(to_rmse.values())):.4f}")
    print(f"all_features mean RMSE: {np.mean(list(af_rmse.values())):.4f}")

    FS_METHODS = ["random_k", "Pearson", "SparseLinear_L1", "RandomForest", "Mean_CKA_L8", "Lagged_CKA_L8"]

    for k in [5, 10, 20]:
        print(f"\n--- K={k} ---")
        print(f"{'Method':<20} {'mean_RMSE':>10} {'med_RMSE':>10} {'std_RMSE':>10} "
              f"{'mean_MAE':>10} {'%vs_TO':>8} {'%vs_AF':>8} "
              f"{'win_TO':>7} {'win_AF':>7} {'best':>5}")
        for m in ["target_only", "all_features_206"] + FS_METHODS:
            row = summary[(summary["method"] == m) & (summary["top_k"] == k)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"  {m:<20} {r['mean_RMSE']:>10.4f} {r['median_RMSE']:>10.4f} "
                  f"{r['std_RMSE']:>10.4f} {r['mean_MAE']:>10.4f} "
                  f"{r.get('pct_impr_vs_to', float('nan')):>8.2f} "
                  f"{r.get('pct_impr_vs_af', float('nan')):>8.2f} "
                  f"{int(r.get('win_vs_to', 0)):>7d} "
                  f"{int(r.get('win_vs_af', 0)):>7d} "
                  f"{int(r.get('best_method_count', 0)):>5d}")

    if not tests.empty:
        print("\n--- PAIRWISE WILCOXON (exploratory; n=10; interpret cautiously) ---")
        print(f"{'Comparison':<45} {'K':>4} {'diff_A-B':>10} {'win_A':>6} {'win_B':>6} {'p':>8}")
        for _, row in tests.sort_values(["comparison", "top_k"]).iterrows():
            print(f"  {row['comparison']:<43} K={int(row['top_k']):<4} "
                  f"{row['mean_diff_A_minus_B']:>+10.4f} "
                  f"{int(row['win_count_A_better']):>6d} "
                  f"{int(row['win_count_B_better']):>6d} "
                  f"{row['p_value']:>8.4f}")

    print("\n--- ROLE-WISE MEAN RMSE (K=10; each role = 2 targets) ---")
    role_sum = compute_role_summary(df)
    role_k10 = role_sum[role_sum["top_k"] == 10]
    methods_show = ["target_only", "all_features_206", "Pearson", "RandomForest",
                    "SparseLinear_L1", "Mean_CKA_L8", "Lagged_CKA_L8", "random_k"]
    roles = sorted(role_k10["target_role"].unique())
    print(f"{'Method':<22}" + "".join(f"{r[:12]:>14}" for r in roles))
    for m in methods_show:
        vals = []
        for r in roles:
            v = role_k10[(role_k10["method"] == m) & (role_k10["target_role"] == r)]["mean_RMSE"]
            vals.append(f"{v.values[0]:>14.4f}" if len(v) > 0 else f"{'N/A':>14}")
        print(f"  {m:<22}" + "".join(vals))

    print("\n--- LAGGED_CKA BUG REPORT ---")
    print("  Lagged_CKA is DEGENERATE on this data.")
    print("  All pairwise scores = 1.000000 (exact) for every candidate-target pair.")
    print("  Cause: patch-centered gram matrices K_x and K_y are nearly proportional")
    print("         (Frobenius cos-similarity ≈ 0.99999 across all windows).")
    print("  Root: Chronos-2 patch embeddings are rank-1 dominated after within-window")
    print("         centering. Dominant SV: ~700; second SV: ~17 (ratio ≈ 40:1).")
    print("         All sensors share the same leading patch direction (diurnal traffic mode).")
    print("  Effect: stable sort returns candidates in insertion order → [0,1,...,K-1]")
    print("         for all targets. These are df_column indices, NOT ranked sensors.")
    print("  NOT a sorting or mapping bug. The scoring function is mathematically correct")
    print("         but the patch-space CKA is methodologically degenerate here.")
    print("  Fix: center over N windows (not P patches) before computing gram matrices.")
    print("  Lagged_CKA results must be EXCLUDED from scientific conclusions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Path to experiment output directory")
    args = parser.parse_args()
    main(args.exp_dir)
