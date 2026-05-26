"""TSFM Feature Selection Experiment — Main Orchestrator.

Usage:
    python run_experiment.py --config configs/config_lag_1h.yaml

Modules executed sequentially:
    0  Init & safety checks (GPU, output directory)
    1  Data setup (METR-LA, sensor selection, windowing, geo adjacency scores)
    2  Latent extraction (Chronos-2, cached to disk, all requested layers)
    3  Representation scoring (Mean_Pooling, Lagged_CKA, Soft_DTW × layers × conditions
                               + Lasso and RF supervised baselines, train-only)
    4  Downstream evaluation (LightGBM on raw traffic, all K in top_ks,
                               Univariate/Geographic/Lasso/RF/Pearson baselines + scored methods)
    5  Reporting (results.csv, selected_sensors.json, 4 publication charts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.real_traffic import (
    ForecastWindows,
    load_metr_la,
    make_forecast_windows,
    select_sensors,
)
from src.evaluation.downstream import (
    compute_ipg_ground_truth,
    evaluate_geographic_baseline,
    evaluate_method,
    evaluate_univariate_baseline,
)
from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
from src.preprocessing.whitening import whiten_embeddings
from src.reporting.plots import (
    plot_layer_ablation,
    plot_overlap_with_geo,
    plot_rmse_vs_k,
    plot_sota_comparison_bar,
)
from src.reporting.saver import save_config
from src.scoring.tsfm_scorers import (
    lagged_cka,
    lasso_fs_scorer,
    mean_pooling_cka,
    pearson_fs_scorer,
    rf_fs_scorer,
    soft_dtw_score,
)


# ---------------------------------------------------------------------------
# Module 0: Initialisation
# ---------------------------------------------------------------------------

def _init(config_path: str) -> tuple[object, Path, str]:
    """Load config, verify GPU, create timestamped output directory."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline requires a GPU. Aborting."
        )
    print(f"[Module 0] GPU: {torch.cuda.get_device_name(0)}")

    cfg       = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path("outputs") / f"EXP_{cfg.experiment.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)

    save_config(config_path, out_dir)
    print(f"[Module 0] Output directory: {out_dir}")
    return cfg, out_dir, timestamp


# ---------------------------------------------------------------------------
# Module 2: Caching helpers
# ---------------------------------------------------------------------------

def _series_cache_key(series: np.ndarray, layer: int) -> str:
    sha = hashlib.sha1(series.tobytes()).hexdigest()
    return f"{sha}_layer{layer}"


def _extract_or_cache(
    extractor: ChronosEmbeddingExtractor,
    windows: np.ndarray,         # [N, context_length]
    layer: int,
    cache_dir: Path,
) -> np.ndarray:
    """Return [N, P, D] embeddings, loading from cache if available."""
    key  = _series_cache_key(windows, layer)
    path = cache_dir / f"{key}.npz"

    if path.exists():
        return np.load(path)["emb"]

    result = extractor.extract_windows(windows, layers=[layer])
    if layer not in result:
        raise RuntimeError(
            f"Layer {layer} was not captured. Check ChronosEmbeddingExtractor hooks."
        )
    emb = result[layer]   # [N, P, D]
    np.savez_compressed(path, emb=emb)
    return emb


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:

    # -----------------------------------------------------------------------
    # Module 0 — Init
    # -----------------------------------------------------------------------
    cfg, out_dir, timestamp = _init(config_path)
    cache_dir = out_dir / "cache"

    layers = cfg.extraction.layers   # e.g. [6, 8, 10]
    gamma  = cfg.scoring.dtw_gamma   # single float

    # -----------------------------------------------------------------------
    # Module 1 — Data setup
    # -----------------------------------------------------------------------
    print("\n[Module 1] Loading METR-LA …")
    data = load_metr_la(cfg.dataset.zip_path)

    target_sensor = data.df.columns[cfg.dataset.target_sensor_idx]
    print(f"  Target sensor : {target_sensor}")

    sensors = select_sensors(
        data,
        target_sensor,
        n_relevant   = cfg.dataset.n_relevant,
        n_distractor = cfg.dataset.n_distractor,
        seed         = cfg.dataset.distractor_seed,
    )
    proxy_relevant_cols = sensors["proxy_relevant"]
    distractor_cols     = sensors["distractor"]
    all_sensor_cols     = proxy_relevant_cols + distractor_cols

    print(f"  proxy_relevant df cols : {proxy_relevant_cols}")
    print(f"  distractor     df cols : {distractor_cols}")

    # Build adjacency scores for all candidate sensors so the geographic
    # baseline can rank them for any K, not just the fixed n_relevant=5.
    target_adj_idx = data.sensor_id_to_idx[target_sensor]
    geo_adj_scores: dict[int, float] = {}
    for col in all_sensor_cols:
        sensor_name = data.df.columns[col]
        adj_idx = data.sensor_id_to_idx.get(sensor_name)
        geo_adj_scores[col] = (
            float(data.adj_matrix[target_adj_idx, adj_idx])
            if adj_idx is not None
            else 0.0
        )

    windows: ForecastWindows = make_forecast_windows(
        data.df,
        target_sensor  = target_sensor,
        context_length = cfg.dataset.context_length,
        horizon        = cfg.dataset.horizon,
        stride         = cfg.dataset.stride,
        max_windows    = cfg.dataset.max_windows,
    )
    N = windows.X_context.shape[0]
    print(f"  Windows created: {N}")

    target_df_col = data.df.columns.get_loc(target_sensor)
    Y_series: np.ndarray = windows.X_context[:, :, target_df_col]   # [N, 144]
    y_target: np.ndarray = windows.y_target                          # [N, 12]

    raw_windows_X: dict[int, np.ndarray] = {
        col: windows.X_context[:, :, col]
        for col in all_sensor_cols
    }

    # IPG ground truth — computed strictly on the train split, no test leakage.
    # Replaces the adjacency-based proxy_relevant_cols for Precision@K / MRR.
    print("  Computing IPG ground truth (train-only) …")
    ipg_relevant_cols = compute_ipg_ground_truth(
        Y_series, raw_windows_X, y_target,
        all_sensor_cols, cfg.evaluation.test_frac,
        n_top=cfg.dataset.n_relevant,
    )
    print(f"  IPG top-{cfg.dataset.n_relevant} sensors: {ipg_relevant_cols}")

    # Pre-flight: report train/test split sizes to confirm statistical validity.
    _n_train = int(N * (1.0 - cfg.evaluation.test_frac))
    _gap     = 144
    _n_test  = N - (_n_train + _gap)
    if _n_test < 0:
        _n_test = N - _n_train   # fallback (gap exceeds remaining rows)
    print(f"  N_total={N}  N_train={_n_train}  gap={_gap}  N_test={_n_test}")
    if _n_test < 400:
        print(f"  [WARNING] N_test={_n_test} < 400 — results may lack statistical power.")

    # -----------------------------------------------------------------------
    # Module 2 — Latent extraction (Chronos-2, all layers, cached)
    # -----------------------------------------------------------------------
    print(f"\n[Module 2] Loading {cfg.extraction.model_id} …")
    extractor = ChronosEmbeddingExtractor(
        model_id = cfg.extraction.model_id,
        layers   = layers,
        pooling  = "none",
    )
    extractor.load()

    # Extract embeddings for every requested layer; cache per (series, layer).
    Y_embs_by_layer: dict[int, np.ndarray] = {}
    for layer in layers:
        print(f"  Extracting target embeddings (layer {layer}) …")
        Y_embs_by_layer[layer] = _extract_or_cache(extractor, Y_series, layer, cache_dir)
    print(f"  Y_emb shape (first layer): {Y_embs_by_layer[layers[0]].shape}")

    X_embs_by_layer: dict[int, dict[int, np.ndarray]] = {}
    for layer in layers:
        X_embs_by_layer[layer] = {}
        for col in all_sensor_cols:
            X_embs_by_layer[layer][col] = _extract_or_cache(
                extractor, raw_windows_X[col], layer, cache_dir
            )
    print(f"  Extracted embeddings for {len(all_sensor_cols)} sensors × {len(layers)} layers.")
    extractor.remove_hooks()

    # -----------------------------------------------------------------------
    # Module 3 — Representation scoring (layer × condition × method)
    #            + supervised baselines (Lasso, RF) fit on train split only
    # -----------------------------------------------------------------------
    print("\n[Module 3] Scoring …")
    max_lag   = cfg.scoring.max_lag
    test_frac = cfg.evaluation.test_frac

    scoring_results: list[dict] = []   # [{layer, method, condition, feat_scores}]

    for layer in layers:
        Y_emb  = Y_embs_by_layer[layer]
        X_embs = X_embs_by_layer[layer]

        # Pre-compute whitened embeddings once per (layer, sensor) to avoid
        # redundant ZCA SVD calls inside the method loop (was 3× per sensor).
        print(f"  Pre-whitening embeddings for layer {layer} …")
        Y_emb_w: np.ndarray = whiten_embeddings(Y_emb)
        X_embs_w: dict[int, np.ndarray] = {
            col: whiten_embeddings(X_embs[col]) for col in all_sensor_cols
        }

        for condition in ["raw", "whitened"]:
            Y_e = Y_emb_w if condition == "whitened" else Y_emb

            method_list: list[tuple[str, object]] = [
                ("Mean_Pooling",        None),
                ("Lagged_CKA",          None),
                (f"Soft_DTW_g{gamma}",  gamma),
            ]

            for method_name, g in method_list:
                print(f"  layer={layer} / {method_name} / {condition} …")
                feat_scores: dict[int, float] = {}

                for col in all_sensor_cols:
                    X_e = X_embs_w[col] if condition == "whitened" else X_embs[col]

                    if method_name == "Mean_Pooling":
                        score = mean_pooling_cka(X_e, Y_e)
                    elif method_name == "Lagged_CKA":
                        score = lagged_cka(X_e, Y_e, max_lag)
                    else:
                        score = soft_dtw_score(X_e, Y_e, float(g))

                    feat_scores[col] = score

                scoring_results.append({
                    "layer":       layer,
                    "method":      method_name,
                    "condition":   condition,
                    "feat_scores": feat_scores,
                })

    # Supervised baselines: fit once on the train portion of the raw windows.
    print("  Lasso supervised scorer (train-only) …")
    lasso_scores = lasso_fs_scorer(raw_windows_X, y_target, all_sensor_cols, test_frac)
    scoring_results.append({
        "layer":       "N/A",
        "method":      "Lasso",
        "condition":   "N/A",
        "feat_scores": lasso_scores,
    })

    print("  RF supervised scorer (train-only) …")
    rf_scores = rf_fs_scorer(raw_windows_X, y_target, all_sensor_cols, test_frac)
    scoring_results.append({
        "layer":       "N/A",
        "method":      "RF",
        "condition":   "N/A",
        "feat_scores": rf_scores,
    })

    print("  Pearson statistical baseline (train-only) …")
    pearson_scores = pearson_fs_scorer(raw_windows_X, Y_series, all_sensor_cols, test_frac)
    scoring_results.append({
        "layer":       "N/A",
        "method":      "Pearson",
        "condition":   "N/A",
        "feat_scores": pearson_scores,
    })

    # -----------------------------------------------------------------------
    # Module 4 — Downstream forecasting (raw traffic Ridge Regression)
    #            Loops over all K values in cfg.evaluation.top_ks.
    #            Records which sensors each method selected at each K.
    # -----------------------------------------------------------------------
    print("\n[Module 4] Downstream evaluation …")
    top_ks = cfg.evaluation.top_ks   # list[int], e.g. [5, 10, 20, 30]

    results_rows: list[dict] = []

    # {str(k): {method_key: [col_idx, ...]}} — dumped to selected_sensors.json
    selected_sensors_map: dict[str, dict[str, list[int]]] = {
        str(k): {} for k in top_ks
    }

    for k in top_ks:
        print(f"\n  [k={k}]")

        # --- Univariate baseline (no sensors selected beyond Y) ---
        print("    Univariate baseline …")
        uni_metrics, uni_sel = evaluate_univariate_baseline(Y_series, y_target, test_frac)
        results_rows.append({
            "k": k, "method": "Univariate", "condition": "N/A", "layer": "N/A",
            **uni_metrics,
        })
        selected_sensors_map[str(k)]["Univariate"] = uni_sel

        # --- Geographic baseline (rank all candidates by adj weight) ---
        print("    Geographic baseline …")
        geo_metrics, geo_sel = evaluate_geographic_baseline(
            Y_series, raw_windows_X, y_target,
            all_sensor_cols, ipg_relevant_cols,
            geo_adj_scores, k, test_frac,
        )
        results_rows.append({
            "k": k, "method": "Geographic", "condition": "N/A", "layer": "N/A",
            **geo_metrics,
        })
        selected_sensors_map[str(k)]["Geographic"] = geo_sel

        # --- All scored methods (TSFM latent + Lasso + RF) ---
        for sr in scoring_results:
            method    = sr["method"]
            condition = sr["condition"]
            layer     = sr["layer"]

            # Build a stable string key for the selected_sensors_map.
            # Normalise Soft_DTW_g* → Soft_DTW so the key is gamma-independent
            # and matches the lookup used in plot_overlap_heatmap.
            norm_method = re.sub(r"^Soft_DTW_g.*", "Soft_DTW", method)
            if condition != "N/A":
                method_key = f"{norm_method}_{condition}_L{layer}"
            else:
                method_key = norm_method

            print(f"    layer={layer} / {method} / {condition} …")
            metrics, sel = evaluate_method(
                feat_scores            = sr["feat_scores"],
                Y_series               = Y_series,
                raw_windows_X          = raw_windows_X,
                y_target               = y_target,
                all_sensor_df_cols     = all_sensor_cols,
                proxy_relevant_df_cols = ipg_relevant_cols,
                top_k                  = k,
                test_frac              = test_frac,
            )
            results_rows.append({
                "k": k, "method": method, "condition": condition, "layer": layer,
                **metrics,
            })
            selected_sensors_map[str(k)][method_key] = sel

    # -----------------------------------------------------------------------
    # Module 5 — Reporting
    # -----------------------------------------------------------------------
    print("\n[Module 5] Saving results …")
    results_df = pd.DataFrame(results_rows)
    csv_path   = out_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    # Dump selected sensor sets for offline Jaccard similarity analysis.
    sensors_path = out_dir / "selected_sensors.json"
    with open(sensors_path, "w") as f:
        json.dump(selected_sensors_map, f, indent=2)
    print(f"  saved {sensors_path}")

    plot_rmse_vs_k(results_df, out_dir)
    plot_sota_comparison_bar(results_df, out_dir)
    plot_layer_ablation(results_df, out_dir)
    plot_overlap_with_geo(selected_sensors_map, results_df, out_dir)

    print(f"\nDone. Results in: {out_dir}")
    display_cols = ["k", "method", "condition", "layer", "RMSE", "MAE", "MAPE", "R2", "MRR"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TSFM Feature Selection — run ablation experiment"
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    main(args.config)
