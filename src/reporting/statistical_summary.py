"""Statistical aggregation and Wilcoxon signed-rank tests for the TSFM experiment.

IMPORTANT: This phase is EXPLORATORY (80/20 train/test split, n=10 targets).
Statistical tests have low power and should not be over-interpreted.

aggregate_results():
    Groups by (method, top_k, layer), computes mean/median RMSE/MAE,
    win counts, and percent improvement over target_only / all_features_N.

run_wilcoxon_tests():
    Runs Wilcoxon signed-rank tests separately for each (top_k, layer)
    combination. Paired samples = target sensors.
    Does NOT pool across K values or layers.

Outputs:
    results/statistical_summary.csv
    results/statistical_tests.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_results_dir(out_dir: Path | None = None) -> Path:
    d = out_dir if out_dir is not None else Path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove replicated-baseline rows and raw per-repeat random_k rows."""
    out = df.copy()
    if "is_replicated_baseline" in out.columns:
        out = out[out["is_replicated_baseline"] != True]
    return out


def _method_base(method_str: str) -> str:
    import re
    return re.sub(r"_L\d+$", "", str(method_str))


def _find_all_features_method(df: pd.DataFrame) -> str | None:
    """Return the first method name starting with 'all_features_', or None."""
    for m in df["method"].unique():
        if str(m).startswith("all_features_"):
            return str(m)
    return None


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

def aggregate_results(results_df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate RMSE/MAE per (method, top_k, layer) across target sensors.

    Also computes:
        - win counts: which method achieves min RMSE per (target, top_k, layer)
        - percent improvement over target_only and all_features_N

    Returns a summary DataFrame and saves to results/statistical_summary.csv.
    """
    df = _clean_df(results_df).copy()
    all_features_method = _find_all_features_method(df)

    # For random_k, aggregate repeats per (target, top_k) before grouping
    rk_mask = df["method"] == "random_k"
    if rk_mask.any():
        rk_agg = (
            df[rk_mask]
            .groupby(["target_sensor_id", "top_k"])[["RMSE", "MAE", "MAPE"]]
            .mean()
            .reset_index()
        )
        rk_agg["method"] = "random_k"
        rk_agg["layer"]  = "N/A"
        for col in df.columns:
            if col not in rk_agg.columns:
                rk_agg[col] = np.nan
        df = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    # Main aggregation: mean/median per (method, top_k, layer)
    grp_cols = ["method", "top_k", "layer"]
    agg = (
        df.groupby(grp_cols)[["RMSE", "MAE"]]
        .agg(
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            std_RMSE=("RMSE", "std"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            n_targets=("RMSE", "count"),
        )
        .reset_index()
    )

    # Win counts: which method has min RMSE per (target, top_k, layer)?
    df_wins = df[df["method"].notna() & df["RMSE"].notna()].copy()
    win_rows = (
        df_wins.loc[df_wins.groupby(["target_sensor_id", "top_k", "layer"])["RMSE"].idxmin()]
        [["method", "top_k", "layer"]]
    )
    win_counts = win_rows.groupby(["method", "top_k", "layer"]).size().reset_index(name="win_count")
    agg = agg.merge(win_counts, on=["method", "top_k", "layer"], how="left")
    agg["win_count"] = agg["win_count"].fillna(0).astype(int)

    # Percent improvement over target_only
    to_rows = df[df["method"] == "target_only"][["target_sensor_id", "RMSE"]].rename(
        columns={"RMSE": "RMSE_target_only"}
    )
    df_imp = df.merge(to_rows, on="target_sensor_id", how="left")
    df_imp["pct_impr_vs_target_only"] = (
        (df_imp["RMSE_target_only"] - df_imp["RMSE"]) / df_imp["RMSE_target_only"] * 100
    )

    # Percent improvement over all_features_N (detect by prefix)
    if all_features_method:
        af_rows = df[df["method"] == all_features_method][["target_sensor_id", "RMSE"]].rename(
            columns={"RMSE": "RMSE_all_features"}
        )
        df_imp = df_imp.merge(af_rows, on="target_sensor_id", how="left")
        df_imp["pct_impr_vs_all_features"] = (
            (df_imp["RMSE_all_features"] - df_imp["RMSE"]) / df_imp["RMSE_all_features"] * 100
        )
    else:
        df_imp["pct_impr_vs_all_features"] = np.nan

    imp_cols = ["pct_impr_vs_target_only", "pct_impr_vs_all_features"]
    imp_agg = df_imp.groupby(grp_cols)[imp_cols].mean().reset_index()
    agg = agg.merge(imp_agg, on=grp_cols, how="left")

    out_path = _get_results_dir(out_dir) / "statistical_summary.csv"
    agg.to_csv(out_path, index=False)
    print(f"[statistical_summary] Saved {out_path}")
    return agg


# ---------------------------------------------------------------------------
# run_wilcoxon_tests
# ---------------------------------------------------------------------------

def run_wilcoxon_tests(results_df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests per (top_k, layer) combination.

    Paired samples = target_sensor_id.
    Tests are run separately for each (top_k, layer) — never pooled.

    Effect size: r = Z / sqrt(N) where Z is approximated from W statistic.

    Returns a DataFrame of test results and saves to results/statistical_tests.csv.
    """
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError:
        warnings.warn("scipy not available — skipping Wilcoxon tests.")
        return pd.DataFrame()

    df = _clean_df(results_df).copy()
    all_features_method = _find_all_features_method(df)

    # Aggregate random_k repeats first
    rk_mask = df["method"] == "random_k"
    if rk_mask.any():
        rk_agg = (
            df[rk_mask]
            .groupby(["target_sensor_id", "top_k"])["RMSE"]
            .mean()
            .reset_index()
        )
        rk_agg["method"] = "random_k"
        rk_agg["layer"]  = "N/A"
        df = pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)

    # Comparisons using _method_base matching (strips _L\d+ suffix).
    # all_features_N is matched dynamically by the detected prefix.
    af_base = all_features_method if all_features_method else "all_features_N"
    comparisons = [
        ("Lagged_CKA",     "target_only"),
        ("Lagged_CKA",     af_base),
        ("Lagged_CKA",     "Pearson"),
        ("Lagged_CKA",     "random_k"),
        ("Mean_CKA",       "Pearson"),
        ("Mean_CKA",       "Lagged_CKA"),
        ("RandomForest",   "Pearson"),
        ("SparseLinear_L1","Pearson"),
    ]

    rows = []
    tk_layer_pairs = df[["top_k", "layer"]].drop_duplicates().values.tolist()

    for top_k, layer in tk_layer_pairs:
        df_tl = df[(df["top_k"] == top_k) & (df["layer"] == layer)]
        if df_tl.empty:
            continue

        for method_a, method_b in comparisons:
            rows_a = df_tl[df_tl["method"].apply(_method_base) == method_a][["target_sensor_id", "RMSE"]]
            rows_b = df_tl[df_tl["method"].apply(_method_base) == method_b][["target_sensor_id", "RMSE"]]

            # target_only and all_features_N are stored at fixed top_k values; match by target
            if method_b in ("target_only", af_base):
                rows_b = df[df["method"].apply(_method_base) == method_b][["target_sensor_id", "RMSE"]]

            merged = rows_a.merge(rows_b, on="target_sensor_id", suffixes=("_a", "_b"))
            if len(merged) < 5:
                continue

            a = merged["RMSE_a"].values
            b = merged["RMSE_b"].values
            n = len(a)
            diffs = a - b

            if np.all(diffs == 0):
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p = _wilcoxon(diffs, alternative="two-sided")
            except Exception:
                continue

            mu    = n * (n + 1) / 4.0
            sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
            Z     = (stat - mu) / sigma if sigma > 0 else 0.0
            r     = abs(Z) / np.sqrt(n) if n > 0 else float("nan")

            rows.append({
                "comparison":         f"{method_a}_vs_{method_b}",
                "top_k":              top_k,
                "layer":              layer,
                "n_pairs":            n,
                "W_statistic":        float(stat),
                "p_value":            float(p),
                "effect_size_r":      float(r),
                "significant_at_05":  bool(p < 0.05),
                "win_count_A":        int((a < b).sum()),
                "mean_improvement":   float(np.mean(b - a)),
                "median_improvement": float(np.median(b - a)),
            })

    test_df = pd.DataFrame(rows)

    out_path = _get_results_dir(out_dir) / "statistical_tests.csv"
    with open(out_path, "w") as f:
        f.write("# EXPLORATORY: n<=10 targets per test, low statistical power\n")
        test_df.to_csv(f, index=False)
    print(f"[statistical_summary] Saved {out_path}")
    return test_df
