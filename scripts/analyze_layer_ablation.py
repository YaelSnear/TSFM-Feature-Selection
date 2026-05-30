"""Analysis A-F for CKA layer ablation results.

Reads layer_ablation_results.jsonl and the original FULL RUN JSONL,
then produces all required summary tables and plots.

Usage:
    conda run --no-capture-output -n yael_env \\
        python scripts/analyze_layer_ablation.py \\
            --exp_dir outputs/EXP_tsfm_full_run_all206_20260530_172932
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ABLATION_METHODS = [
    "Mean_CKA_L6", "Mean_CKA_L8", "Mean_CKA_L10",
    "Lagged_CKA_L6_fixed", "Lagged_CKA_L8_fixed", "Lagged_CKA_L10_fixed",
]
BASELINES = ["target_only", "all_features_206", "random_k",
             "Pearson", "SparseLinear_L1", "RandomForest"]

METHOD_COLORS = {
    "Mean_CKA_L6":            "#ff9999",
    "Mean_CKA_L8":            "#ee6677",
    "Mean_CKA_L10":           "#cc0033",
    "Lagged_CKA_L6_fixed":   "#ccaaff",
    "Lagged_CKA_L8_fixed":   "#aa44cc",
    "Lagged_CKA_L10_fixed":  "#660099",
    "Pearson":                "#4477aa",
    "SparseLinear_L1":        "#66ccee",
    "RandomForest":           "#228833",
    "random_k":               "#ccbb44",
    "target_only":            "#aaaaaa",
    "all_features_206":       "#bbbbbb",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows


def load_all(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ablation JSONL and original FULL RUN JSONL."""
    abl_rows  = load_jsonl(exp_dir / "layer_ablation_cka" / "layer_ablation_results.jsonl")
    orig_rows = load_jsonl(exp_dir / "results" / "results_incremental.jsonl")

    abl_df  = pd.DataFrame(abl_rows)
    orig_df = pd.DataFrame(orig_rows)

    # Drop replicated baselines from original
    if "is_replicated_baseline" in orig_df.columns:
        orig_df = orig_df[orig_df["is_replicated_baseline"] != True].copy()

    print(f"Ablation rows: {len(abl_df)}")
    print(f"Original rows: {len(orig_df)}")
    return abl_df, orig_df


def _rk_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate random_k repeats per (target, top_k)."""
    rk_mask = df["method"] == "random_k"
    if not rk_mask.any():
        return df
    rk_agg = (
        df[rk_mask].groupby(["target_sensor_id", "top_k"])[["RMSE", "MAE"]]
        .mean().reset_index()
    )
    rk_agg["method"] = "random_k"
    return pd.concat([df[~rk_mask], rk_agg], ignore_index=True, sort=False)


def _overlap_k(a: list, b: list, k: int) -> int:
    return len(set(a[:k]) & set(b[:k]))


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0


# ---------------------------------------------------------------------------
# Analysis A — forecasting comparison per method × layer × K
# ---------------------------------------------------------------------------

def analysis_a(abl_df: pd.DataFrame, orig_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Analysis A: Forecasting comparison ===")
    import warnings as _w
    from scipy.stats import wilcoxon

    orig_clean = _rk_agg(orig_df)
    to_rmse = orig_clean[orig_clean["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].to_dict()
    af_rmse = orig_clean[orig_clean["method"] == "all_features_206"].set_index("target_sensor_id")["RMSE"].to_dict()

    ref_methods = {
        m: {} for m in ["Pearson", "SparseLinear_L1", "RandomForest",
                         "random_k", "Mean_CKA_L8", "Lagged_CKA_L8_fixed"]
    }
    for m in ref_methods:
        src = orig_clean if m not in ["Mean_CKA_L8", "Lagged_CKA_L8_fixed"] else abl_df
        for _, r in src[src["method"] == m].iterrows():
            ref_methods[m][(str(r["target_sensor_id"]), int(r["top_k"]))] = r["RMSE"]

    rows = []
    for method in ABLATION_METHODS:
        sub = abl_df[abl_df["method"] == method]
        for top_k in [5, 10, 20]:
            sk = sub[sub["top_k"] == top_k]
            if sk.empty:
                continue
            rmse   = sk["RMSE"].values
            mae    = sk["MAE"].values
            tgts   = sk["target_sensor_id"].values.astype(str)
            to_r   = np.array([to_rmse.get(t, np.nan) for t in tgts])
            af_r   = np.array([af_rmse.get(t, np.nan) for t in tgts])
            win_dict = {}
            wlx_dict = {}
            for ref_name, ref_map in ref_methods.items():
                ref_r = np.array([ref_map.get((t, top_k), np.nan) for t in tgts])
                win_dict[ref_name] = int(np.nansum(rmse < ref_r))
                diffs = rmse - ref_r
                valid = ~np.isnan(diffs)
                if valid.sum() >= 5 and not np.all(diffs[valid] == 0):
                    try:
                        with _w.catch_warnings():
                            _w.simplefilter("ignore")
                            _, p = wilcoxon(diffs[valid], alternative="two-sided")
                    except Exception:
                        p = np.nan
                else:
                    p = np.nan
                wlx_dict[ref_name] = round(float(p), 4) if np.isfinite(p) else np.nan

            row = {
                "method":            method,
                "top_k":             top_k,
                "n_targets":         len(sk),
                "mean_RMSE":         float(np.mean(rmse)),
                "median_RMSE":       float(np.median(rmse)),
                "std_RMSE":          float(np.std(rmse)),
                "mean_MAE":          float(np.mean(mae)),
                "median_MAE":        float(np.median(mae)),
                "pct_impr_vs_to":    float(np.nanmean((to_r - rmse) / to_r * 100)),
                "pct_impr_vs_af":    float(np.nanmean((af_r - rmse) / af_r * 100)),
                "win_vs_to":         int(np.nansum(rmse < to_r)),
                "win_vs_af":         int(np.nansum(rmse < af_r)),
            }
            for ref_name in ref_methods:
                row[f"win_vs_{ref_name}"]   = win_dict[ref_name]
                row[f"p_vs_{ref_name}"]     = wlx_dict[ref_name]
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "layer_ablation_statistical_summary.csv", index=False)
    print(f"  Saved: {out_dir / 'layer_ablation_statistical_summary.csv'}")
    return df_out


# ---------------------------------------------------------------------------
# Analysis B — layer ranking per method
# ---------------------------------------------------------------------------

def analysis_b(abl_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Analysis B: Layer ranking ===")
    rows = []
    for method_base in ["Mean_CKA", "Lagged_CKA_fixed"]:
        for top_k in [5, 10, 20]:
            layer_rmse = {}
            for layer in [6, 8, 10]:
                if method_base == "Mean_CKA":
                    mname = f"Mean_CKA_L{layer}"
                else:
                    mname = f"Lagged_CKA_L{layer}_fixed"
                sub = abl_df[(abl_df["method"] == mname) & (abl_df["top_k"] == top_k)]
                if sub.empty:
                    continue
                layer_rmse[layer] = sub["RMSE"].values

            if not layer_rmse:
                continue
            best_mean   = min(layer_rmse, key=lambda l: np.mean(layer_rmse[l]))
            best_median = min(layer_rmse, key=lambda l: np.median(layer_rmse[l]))

            # Per-target best layer
            target_best: dict[str, int] = {}
            if method_base == "Mean_CKA":
                methods_k = {l: f"Mean_CKA_L{l}" for l in [6, 8, 10]}
            else:
                methods_k = {l: f"Lagged_CKA_L{l}_fixed" for l in [6, 8, 10]}
            targets = abl_df[abl_df["method"] == list(methods_k.values())[0]]["target_sensor_id"].unique()
            for tgt in targets:
                tgt_rmse = {
                    l: abl_df[(abl_df["method"] == methods_k[l]) &
                               (abl_df["top_k"] == top_k) &
                               (abl_df["target_sensor_id"] == tgt)]["RMSE"].values
                    for l in [6, 8, 10]
                }
                valid = {l: v for l, v in tgt_rmse.items() if len(v) > 0}
                if valid:
                    target_best[str(tgt)] = min(valid, key=lambda l: float(valid[l][0]))

            best_layer_counts = dict(Counter(target_best.values()))
            rows.append({
                "method_base":       method_base,
                "top_k":             top_k,
                "best_layer_mean":   best_mean,
                "best_layer_median": best_median,
                "per_target_best_layer_counts": json.dumps(best_layer_counts),
                **{f"mean_RMSE_L{l}": float(np.mean(layer_rmse[l])) if l in layer_rmse else np.nan
                   for l in [6, 8, 10]},
                **{f"median_RMSE_L{l}": float(np.median(layer_rmse[l])) if l in layer_rmse else np.nan
                   for l in [6, 8, 10]},
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "layer_ranking.csv", index=False)
    print(f"  Saved: {out_dir / 'layer_ranking.csv'}")
    _print_layer_ranking(df_out)
    return df_out


def _print_layer_ranking(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        print(f"  {r['method_base']} K={r['top_k']}:  "
              f"best_layer_mean=L{r['best_layer_mean']}  "
              f"best_layer_median=L{r['best_layer_median']}  "
              f"target_counts={r['per_target_best_layer_counts']}")
        for l in [6, 8, 10]:
            print(f"    L{l}: mean={r.get(f'mean_RMSE_L{l}',np.nan):.4f}  "
                  f"median={r.get(f'median_RMSE_L{l}',np.nan):.4f}")


# ---------------------------------------------------------------------------
# Analysis C — selection stability across layers
# ---------------------------------------------------------------------------

def analysis_c(abl_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Analysis C: Selection stability ===")
    rows = []
    for method_base in ["Mean_CKA", "Lagged_CKA_fixed"]:
        for top_k in [5, 10, 20]:
            if method_base == "Mean_CKA":
                mmap = {l: f"Mean_CKA_L{l}" for l in [6, 8, 10]}
            else:
                mmap = {l: f"Lagged_CKA_L{l}_fixed" for l in [6, 8, 10]}

            all_olap = defaultdict(list)
            all_jacc = defaultdict(list)

            targets = abl_df["target_sensor_id"].unique()
            for tgt in targets:
                sel = {}
                for l, mname in mmap.items():
                    sub = abl_df[(abl_df["method"] == mname) &
                                  (abl_df["top_k"] == top_k) &
                                  (abl_df["target_sensor_id"] == tgt)]
                    if not sub.empty and isinstance(sub.iloc[0]["selected_sensors"], str):
                        sel[l] = json.loads(sub.iloc[0]["selected_sensors"])
                    else:
                        sel[l] = []

                for (la, lb) in [(6, 8), (6, 10), (8, 10)]:
                    if sel.get(la) and sel.get(lb):
                        all_olap[(la, lb)].append(_overlap_k(sel[la], sel[lb], top_k))
                        all_jacc[(la, lb)].append(_jaccard(sel[la][:top_k], sel[lb][:top_k]))

            for (la, lb), olaps in all_olap.items():
                rows.append({
                    "method_base":   method_base,
                    "top_k":         top_k,
                    "layer_pair":    f"L{la}_L{lb}",
                    "mean_overlap":  float(np.mean(olaps)),
                    "mean_jaccard":  float(np.mean(all_jacc[(la, lb)])),
                    "n_targets":     len(olaps),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "layer_selection_stability.csv", index=False)
    print(f"  Saved: {out_dir / 'layer_selection_stability.csv'}")
    print(f"\n  {'method':<22}  {'K':>4}  {'pair':>10}  {'mean_overlap':>13}  {'mean_jaccard':>13}")
    for _, r in df_out.iterrows():
        print(f"  {r['method_base']:<22}  {r['top_k']:>4}  {r['layer_pair']:>10}"
              f"  {r['mean_overlap']:>13.3f}  {r['mean_jaccard']:>13.3f}")
    return df_out


# ---------------------------------------------------------------------------
# Analysis D — difference from existing methods
# ---------------------------------------------------------------------------

def analysis_d(abl_df: pd.DataFrame, orig_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Analysis D: Similarity to Lagged_CKA_fixed ===")
    from scipy.stats import spearmanr

    orig_clean = _rk_agg(orig_df)
    # Build reference selected_sensors per (method, target, K)
    ref_sel: dict[tuple, list] = {}
    for _, r in orig_clean.iterrows():
        m = r["method"]
        if m in ["Pearson", "SparseLinear_L1", "RandomForest", "Mean_CKA_L8"]:
            sel = json.loads(r["selected_sensors"]) if isinstance(r["selected_sensors"], str) else []
            ref_sel[(m, str(r["target_sensor_id"]), int(r["top_k"]))] = sel
    # random_k: pick first repeat per (target, K)
    rk_rows = orig_df[orig_df["method"] == "random_k"].copy()
    for _, r in rk_rows.drop_duplicates(subset=["target_sensor_id", "top_k"]).iterrows():
        sel = json.loads(r["selected_sensors"]) if isinstance(r["selected_sensors"], str) else []
        ref_sel[("random_k", str(r["target_sensor_id"]), int(r["top_k"]))] = sel

    ref_names = ["Pearson", "SparseLinear_L1", "RandomForest", "Mean_CKA_L8", "random_k"]
    rows = []
    for layer in [6, 8, 10]:
        lc_name = f"Lagged_CKA_L{layer}_fixed"
        mc_name = f"Mean_CKA_L{layer}"
        lc_sub  = abl_df[abl_df["method"] == lc_name]

        for top_k in [5, 10, 20]:
            lc_k = lc_sub[lc_sub["top_k"] == top_k]
            for ref_name in ref_names + [mc_name]:
                olaps, jaccs = [], []
                for _, r in lc_k.iterrows():
                    tgt = str(r["target_sensor_id"])
                    lc_sel = json.loads(r["selected_sensors"]) if isinstance(r["selected_sensors"], str) else []
                    if ref_name == mc_name:
                        mc_r = abl_df[(abl_df["method"] == mc_name) &
                                       (abl_df["top_k"] == top_k) &
                                       (abl_df["target_sensor_id"] == tgt)]
                        ref_s = json.loads(mc_r.iloc[0]["selected_sensors"]) if not mc_r.empty else []
                    else:
                        ref_s = ref_sel.get((ref_name, tgt, top_k), [])
                    if ref_s:
                        olaps.append(_overlap_k(lc_sel, ref_s, top_k))
                        jaccs.append(_jaccard(lc_sel[:top_k], ref_s[:top_k]))
                if olaps:
                    rows.append({
                        "CKA_method":      lc_name,
                        "ref_method":      ref_name,
                        "top_k":           top_k,
                        "mean_overlap":    float(np.mean(olaps)),
                        "mean_jaccard":    float(np.mean(jaccs)),
                        "n_targets":       len(olaps),
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "selection_similarity_to_lagged_cka.csv", index=False)
    print(f"  Saved: {out_dir / 'selection_similarity_to_lagged_cka.csv'}")
    # Print ranking by overlap@10 for L8
    print(f"\n  Similarity ranking (mean overlap@10 vs Lagged_CKA_L8_fixed):")
    sub10 = df_out[(df_out["CKA_method"] == "Lagged_CKA_L8_fixed") & (df_out["top_k"] == 10)]
    for _, r in sub10.sort_values("mean_overlap", ascending=False).iterrows():
        print(f"    {r['ref_method']:<24}: overlap={r['mean_overlap']:.2f}  "
              f"jaccard={r['mean_jaccard']:.3f}")
    return df_out


# ---------------------------------------------------------------------------
# Analysis E — lag behavior by layer
# ---------------------------------------------------------------------------

def analysis_e(abl_df: pd.DataFrame, score_dir: Path, out_dir: Path) -> pd.DataFrame:
    print("\n=== Analysis E: Lag behavior by layer ===")
    rows = []
    P = 11  # Chronos-2 patches for context_length=144

    for layer in [6, 8, 10]:
        lc_name = f"Lagged_CKA_L{layer}_fixed"
        sub = abl_df[abl_df["method"] == lc_name]

        # Try to load from score tables if available
        score_files = list(score_dir.glob("*_L6_L10.csv"))
        if score_files and layer in [6, 10]:
            all_s = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)
            layer_s = all_s[(all_s["method"] == lc_name)]
            lags = layer_s["best_lag"].dropna().astype(int).tolist()
        else:
            # Fall back to best_lag_distribution field in JSONL
            lags = []
            for _, r in sub.iterrows():
                dist = r.get("best_lag_distribution")
                if isinstance(dist, str):
                    d = json.loads(dist)
                    for lag_val, cnt in d.items():
                        lags.extend([int(lag_val)] * int(cnt))

        if not lags:
            continue

        lag_counts = Counter(lags)
        n_total    = len(lags)
        n_lag0     = lag_counts.get(0, 0)
        max_lag_valid = min(24, P - 1) - 1  # boundary = P-2 = 9 for P=11
        n_boundary = sum(lag_counts.get(k, 0) for k in [max_lag_valid, -max_lag_valid])
        n_other    = n_total - n_lag0 - n_boundary

        rows.append({
            "layer":             layer,
            "method":            lc_name,
            "n_total_lags":      n_total,
            "n_lag0":            n_lag0,
            "frac_lag0":         round(n_lag0 / n_total, 3) if n_total > 0 else np.nan,
            "n_boundary":        n_boundary,
            "frac_boundary":     round(n_boundary / n_total, 3) if n_total > 0 else np.nan,
            "n_intermediate":    n_other,
            "frac_intermediate": round(n_other / n_total, 3) if n_total > 0 else np.nan,
            "lag_distribution":  json.dumps(dict(sorted(lag_counts.items()))),
        })
        print(f"  L{layer}: n={n_total}  lag=0: {n_lag0} ({n_lag0/n_total*100:.0f}%)"
              f"  boundary: {n_boundary} ({n_boundary/n_total*100:.0f}%)"
              f"  intermediate: {n_other} ({n_other/n_total*100:.0f}%)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "lag_behavior_by_layer.csv", index=False)
    print(f"  Saved: {out_dir / 'lag_behavior_by_layer.csv'}")
    return df_out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(abl_df: pd.DataFrame, orig_df: pd.DataFrame,
               summary_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    orig_clean = _rk_agg(orig_df)
    to_rmse = orig_clean[orig_clean["method"] == "target_only"].set_index("target_sensor_id")["RMSE"].mean()

    # Bar: mean RMSE by method × K (only ablation methods)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, k in zip(axes, [5, 10, 20]):
        sub = summary_df[summary_df["top_k"] == k]
        sub = sub[sub["method"].isin(ABLATION_METHODS)].copy()
        sub["method"] = pd.Categorical(sub["method"], categories=ABLATION_METHODS, ordered=True)
        sub = sub.sort_values("method")
        colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
        ax.bar(range(len(sub)), sub["mean_RMSE"], color=colors)
        ax.axhline(to_rmse, color="gray", linestyle="--", linewidth=0.8, label="target_only")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"K={k}")
        ax.set_ylabel("Mean RMSE" if k == 5 else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Mean RMSE by CKA Method × Layer (ablation; exploratory)")
    plt.tight_layout()
    fig.savefig(plots_dir / "bar_rmse_by_method_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Pct improvement vs target_only
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, k in zip(axes, [5, 10, 20]):
        sub = summary_df[(summary_df["top_k"] == k) & summary_df["method"].isin(ABLATION_METHODS)].copy()
        sub["method"] = pd.Categorical(sub["method"], categories=ABLATION_METHODS, ordered=True)
        sub = sub.sort_values("method")
        colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
        ax.bar(range(len(sub)), sub["pct_impr_vs_to"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"K={k}")
        ax.set_ylabel("% improvement vs target_only" if k == 5 else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("% Improvement vs target_only by Layer (exploratory)")
    plt.tight_layout()
    fig.savefig(plots_dir / "pct_improvement_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Win counts vs Pearson
    if "win_vs_Pearson" in summary_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, k in zip(axes, [5, 10, 20]):
            sub = summary_df[(summary_df["top_k"] == k) & summary_df["method"].isin(ABLATION_METHODS)].copy()
            sub["method"] = pd.Categorical(sub["method"], categories=ABLATION_METHODS, ordered=True)
            sub = sub.sort_values("method")
            colors = [METHOD_COLORS.get(m, "#888") for m in sub["method"]]
            ax.bar(range(len(sub)), sub["win_vs_Pearson"], color=colors)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"K={k}")
            ax.set_ylabel("Win count vs Pearson" if k == 5 else "")
            ax.set_ylim(0, 11)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Win count vs Pearson by Layer (exploratory; n=10 targets)")
        plt.tight_layout()
        fig.savefig(plots_dir / "win_count_by_layer.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Analysis F — critical interpretation
# ---------------------------------------------------------------------------

def analysis_f(summary_df: pd.DataFrame, layer_rank_df: pd.DataFrame,
               similarity_df: pd.DataFrame, lag_df: pd.DataFrame,
               out_dir: Path) -> None:
    print("\n=== Analysis F: Critical interpretation ===")
    lines = [
        "LAYER ABLATION — CRITICAL INTERPRETATION",
        "=" * 70,
        "Context: exploratory; n=10 targets; 91.7% window overlap; low power.",
        "",
    ]

    # Best layer summary
    lines.append("1. Best layer by mean RMSE (K=10):")
    for _, r in layer_rank_df[layer_rank_df["top_k"] == 10].iterrows():
        lines.append(f"   {r['method_base']}: L{r['best_layer_mean']}  "
                     f"(L6={r.get('mean_RMSE_L6',float('nan')):.4f}  "
                     f"L8={r.get('mean_RMSE_L8',float('nan')):.4f}  "
                     f"L10={r.get('mean_RMSE_L10',float('nan')):.4f})")

    # Layer spread
    lines.append("\n2. Layer spread (mean RMSE difference L6 vs L10 at K=10):")
    for _, r in layer_rank_df[layer_rank_df["top_k"] == 10].iterrows():
        diff = abs(r.get("mean_RMSE_L6", np.nan) - r.get("mean_RMSE_L10", np.nan))
        lines.append(f"   {r['method_base']}: |L6 - L10| = {diff:.4f}")
    lines.append("   (differences < 0.1 RMSE are likely within noise; n=10 is low power)")

    # Win vs Pearson
    lines.append("\n3. Win count vs Pearson (K=10):")
    for method in ABLATION_METHODS:
        sub = summary_df[(summary_df["method"] == method) & (summary_df["top_k"] == 10)]
        if not sub.empty and "win_vs_Pearson" in sub.columns:
            lines.append(f"   {method}: wins {sub['win_vs_Pearson'].values[0]}/10")

    # Similarity to Pearson
    lines.append("\n4. Overlap@10 with Pearson per layer (Lagged_CKA_fixed):")
    for layer in [6, 8, 10]:
        lc_name = f"Lagged_CKA_L{layer}_fixed"
        sub = similarity_df[(similarity_df["CKA_method"] == lc_name) &
                             (similarity_df["ref_method"] == "Pearson") &
                             (similarity_df["top_k"] == 10)]
        if not sub.empty:
            lines.append(f"   {lc_name}: overlap@10 = {sub['mean_overlap'].values[0]:.2f}/10")

    # Lag behavior
    if not lag_df.empty:
        lines.append("\n5. Boundary-lag issue (Lagged_CKA_fixed) by layer:")
        for _, r in lag_df.iterrows():
            lines.append(f"   L{r['layer']}: lag=0 {r['frac_lag0']*100:.0f}%  "
                         f"boundary {r['frac_boundary']*100:.0f}%  "
                         f"intermediate {r['frac_intermediate']*100:.0f}%")
        all_bnd = all(row["frac_intermediate"] == 0 for _, row in lag_df.iterrows())
        if all_bnd:
            lines.append("   CONCLUSION: boundary-lag issue persists across all layers.")
            lines.append("   The lag dimension is not providing meaningful lag selection.")
        else:
            lines.append("   Some intermediate lags appear at certain layers — investigate.")

    # Overall conclusions
    lines += [
        "",
        "6. Does any layer make CKA clearly better than Pearson?",
        "   To be determined from data above. If win counts are consistently ≤ 5/10",
        "   across all layers, no layer provides a clear advantage.",
        "",
        "7. Does Lagged_CKA_fixed provide different ranking than Pearson?",
        "   If overlap@10 > 7/10 for all layers, the rankings are similar.",
        "   If overlap@5 = 5/5, the top sensors are identical to Pearson.",
        "",
        "8. Recommendation:",
        "   - If no layer shows consistent wins: report CKA as recovering lagged",
        "     dependence structure (consistent with Pearson), not adding new signal.",
        "   - If a layer shows > 6/10 wins vs Pearson with p < 0.1: investigate further.",
        "   - The boundary-lag issue in Lagged_CKA_fixed suggests the lag search is not",
        "     finding meaningful temporal offsets. Consider replacing with window-level",
        "     lagged CKA on pooled embeddings (Mean_CKA with lag) as the next variant.",
        "",
        "9. Proposed alternative latent-space methods (if current CKA fails):",
        "   A. Window-CKA with lag on mean-pooled embeddings: apply _cka_core([N,D])",
        "      to lag-shifted (mean-pooled X_lag, Y_lag) — bridges Mean_CKA and Lagged_CKA",
        "   B. Linear probe: fit linear model on X_emb [N,P*D] to predict target mean RMSE",
        "      across scoring windows; select sensors by predictive weight.",
        "   C. RSA (Representational Similarity Analysis): compare N×N pairwise distance",
        "      matrices of X and Y embeddings; kernel alignment, not CKA.",
        "   D. Cosine nearest-neighbor consistency: fraction of windows where candidate",
        "      is nearest neighbor to target in embedding space.",
        "   E. Layer-wise importance: if above results show no clear winner, ablate more",
        "      layers (e.g. [2, 4, 6, 8, 10, 12]) to find optimal layer.",
    ]

    text = "\n".join(lines)
    print("\n" + text)
    out_path = out_dir / "layer_ablation_interpretation.txt"
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    args = parser.parse_args()

    exp_dir   = Path(args.exp_dir)
    out_dir   = exp_dir / "layer_ablation_cka"
    score_dir = out_dir / "layer_ablation_score_tables"
    plots_dir = out_dir / "layer_ablation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    abl_df, orig_df = load_all(exp_dir)

    if abl_df.empty:
        print("ERROR: no ablation rows found. Run run_layer_ablation_cka.py first.")
        import sys; sys.exit(1)

    summary_df    = analysis_a(abl_df, orig_df, out_dir)
    layer_rank_df = analysis_b(abl_df, out_dir)
    _             = analysis_c(abl_df, out_dir)
    similarity_df = analysis_d(abl_df, orig_df, out_dir)
    lag_df        = analysis_e(abl_df, score_dir, out_dir)

    try:
        make_plots(abl_df, orig_df, summary_df, plots_dir)
    except Exception as e:
        print(f"  [warn] plot error: {e}")

    analysis_f(summary_df, layer_rank_df, similarity_df, lag_df, out_dir)

    # Save selected sensors JSON
    sel_map: dict = {}
    for _, r in abl_df.iterrows():
        m   = str(r["method"])
        tgt = str(r["target_sensor_id"])
        k   = str(int(r["top_k"]))
        sel = json.loads(r["selected_sensors"]) if isinstance(r["selected_sensors"], str) else []
        sel_map.setdefault(tgt, {}).setdefault(k, {})[m] = sel
    with open(out_dir / "layer_ablation_selected_sensors.json", "w") as f:
        json.dump(sel_map, f, indent=2)
    print(f"\n  Saved: {out_dir / 'layer_ablation_selected_sensors.json'}")
    print(f"\nAnalysis complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
