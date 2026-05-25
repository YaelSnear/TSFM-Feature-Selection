"""TSFM Feature Selection Experiment — Main Orchestrator.

Usage:
    python run_experiment.py --config configs/config_lag_1h.yaml

Modules executed sequentially:
    0  Init & safety checks (GPU, output directory)
    1  Data setup (METR-LA, sensor selection, windowing)
    2  Latent extraction (Chronos-2, cached to disk, all requested layers)
    3  Representation scoring (Mean_Pooling, Lagged_CKA, Soft_DTW × layers × conditions)
    4  Downstream evaluation (Ridge Regression on raw traffic, + 2 baselines)
    5  Reporting (results.csv, 4 publication charts)
"""

from __future__ import annotations

import argparse
import hashlib
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
    evaluate_geographic_baseline,
    evaluate_method,
    evaluate_univariate_baseline,
)
from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
from src.preprocessing.whitening import whiten_embeddings
from src.reporting.plots import (
    plot_layer_comparison,
    plot_mrr_bar,
    plot_precision_vs_rmse,
    plot_rmse_bar,
)
from src.reporting.saver import save_config
from src.scoring.tsfm_scorers import lagged_cka, mean_pooling_cka, soft_dtw_score


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
        n_relevant  = cfg.dataset.n_relevant,
        n_distractor = cfg.dataset.n_distractor,
        seed        = cfg.dataset.distractor_seed,
    )
    proxy_relevant_cols = sensors["proxy_relevant"]
    distractor_cols     = sensors["distractor"]
    all_sensor_cols     = proxy_relevant_cols + distractor_cols

    print(f"  proxy_relevant df cols : {proxy_relevant_cols}")
    print(f"  distractor     df cols : {distractor_cols}")

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
    # -----------------------------------------------------------------------
    print("\n[Module 3] Scoring …")
    max_lag = cfg.scoring.max_lag

    scoring_results: list[dict] = []   # [{layer, method, condition, feat_scores}]

    for layer in layers:
        Y_emb = Y_embs_by_layer[layer]
        X_embs = X_embs_by_layer[layer]

        for condition in ["raw", "whitened"]:
            Y_e = whiten_embeddings(Y_emb) if condition == "whitened" else Y_emb

            method_list: list[tuple[str, object]] = [
                ("Mean_Pooling",        None),
                ("Lagged_CKA",          None),
                (f"Soft_DTW_g{gamma}",  gamma),
            ]

            for method_name, g in method_list:
                print(f"  layer={layer} / {method_name} / {condition} …")
                feat_scores: dict[int, float] = {}

                for col in all_sensor_cols:
                    X_e = (
                        whiten_embeddings(X_embs[col])
                        if condition == "whitened"
                        else X_embs[col]
                    )
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

    # -----------------------------------------------------------------------
    # Module 4 — Downstream forecasting (raw traffic Ridge Regression)
    # -----------------------------------------------------------------------
    print("\n[Module 4] Downstream evaluation …")
    top_k     = cfg.evaluation.top_k
    test_frac = cfg.evaluation.test_frac

    results_rows: list[dict] = []

    # --- Baselines (condition/layer-agnostic, evaluated once) ---
    print("  Univariate baseline …")
    uni_metrics = evaluate_univariate_baseline(Y_series, y_target, test_frac)
    results_rows.append({"method": "Univariate", "condition": "N/A", "layer": "N/A", **uni_metrics})

    print("  Geographic baseline …")
    geo_metrics = evaluate_geographic_baseline(
        Y_series, raw_windows_X, y_target, proxy_relevant_cols, test_frac
    )
    results_rows.append({"method": "Geographic", "condition": "N/A", "layer": "N/A", **geo_metrics})

    # --- Scored methods ---
    for sr in scoring_results:
        method      = sr["method"]
        condition   = sr["condition"]
        layer       = sr["layer"]
        feat_scores = sr["feat_scores"]
        print(f"  layer={layer} / {method} / {condition} …")

        metrics = evaluate_method(
            feat_scores            = feat_scores,
            Y_series               = Y_series,
            raw_windows_X          = raw_windows_X,
            y_target               = y_target,
            all_sensor_df_cols     = all_sensor_cols,
            proxy_relevant_df_cols = proxy_relevant_cols,
            top_k                  = top_k,
            test_frac              = test_frac,
        )
        results_rows.append({"method": method, "condition": condition, "layer": layer, **metrics})

    # -----------------------------------------------------------------------
    # Module 5 — Reporting
    # -----------------------------------------------------------------------
    print("\n[Module 5] Saving results …")
    results_df = pd.DataFrame(results_rows)
    csv_path   = out_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    plot_rmse_bar(results_df, out_dir, timestamp)
    plot_mrr_bar(results_df, out_dir, timestamp)
    plot_layer_comparison(results_df, out_dir)
    plot_precision_vs_rmse(results_df, out_dir, top_k)

    print(f"\nDone. Results in: {out_dir}")
    display_cols = ["method", "condition", "layer", "RMSE", "MAE", "MAPE", "R2", "MRR"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TSFM Feature Selection — run ablation experiment"
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    main(args.config)
