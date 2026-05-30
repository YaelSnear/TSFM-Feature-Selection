"""K=5 forecast sanity for corrected window-centered Lagged CKA.

Runs only if diagnose_lagged_cka_fixed.py Part B passed.
Uses target 717469, all 206 candidates, max_scoring_windows=200,
max_test_windows=50, K=5 only.  Requires GPU.

Usage:
    conda run --no-capture-output -n yael_env \
        python scripts/sanity_lagged_cka_forecast.py \
            --cache_dir outputs/EXP_tsfm_full_run_all206_20260530_172932/cache
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

TARGET_SENSOR_ID = "717469"
TARGET_ROLE      = "role_A_central"
TARGET_DF_COL    = 154
CONTEXT_LENGTH   = 144
STRIDE           = 12
HORIZON          = 12
MAX_WINDOWS      = 9999
MAX_SCORING_WIN  = 200
MAX_TEST_WIN     = 50
MAX_LAG          = 24
LAYER            = 8
TEST_FRAC        = 0.2
CROSS_LEARNING   = False
LAYERS_KEY       = [LAYER]
METHOD_NAME      = "Lagged_CKA_L8_fixed"


def _cache_key(windows_arr: np.ndarray, layers: list[int]) -> str:
    sha = hashlib.sha1(windows_arr.tobytes()).hexdigest()
    return f"{sha}_layers{'_'.join(str(l) for l in sorted(layers))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="outputs/EXP_tsfm_full_run_all206_20260530_172932/cache",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path("outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        print(f"ERROR: cache_dir not found: {cache_dir}")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    from src.data.real_traffic import load_metr_la, make_forecast_windows
    from src.evaluation.tsfm_downstream import TSFMForecaster
    from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
    from src.scoring.tsfm_scorers import lagged_cka_window_centered, precompute_lagged_y_wc

    print(f"[sanity] Target: {TARGET_SENSOR_ID}  K=5  max_scoring={MAX_SCORING_WIN}"
          f"  max_test={MAX_TEST_WIN}  layer={LAYER}  max_lag={MAX_LAG}")

    # ---- Load data ----
    data    = load_metr_la("data/raw/METR-LA.zip")
    df      = data.df
    N_total = len(df)
    n_train = int(N_total * (1.0 - TEST_FRAC))
    gap     = CONTEXT_LENGTH
    n_test_start = n_train + gap

    train_df = df.iloc[:n_train]
    test_df  = df.iloc[n_test_start:]

    sensor_cols = list(df.columns)
    if TARGET_SENSOR_ID not in sensor_cols:
        print(f"ERROR: {TARGET_SENSOR_ID} not found in data")
        sys.exit(1)
    target_df_col = sensor_cols.index(TARGET_SENSOR_ID)

    # ---- Make windows ----
    train_wins = make_forecast_windows(
        train_df,
        target_sensor  = TARGET_SENSOR_ID,
        context_length = CONTEXT_LENGTH,
        horizon        = HORIZON,
        stride         = STRIDE,
        max_windows    = MAX_WINDOWS,
    )
    test_wins = make_forecast_windows(
        test_df,
        target_sensor  = TARGET_SENSOR_ID,
        context_length = CONTEXT_LENGTH,
        horizon        = HORIZON,
        stride         = STRIDE,
        max_windows    = None,
    )

    N_use  = min(MAX_SCORING_WIN, train_wins.X_context.shape[0])
    N_test = min(MAX_TEST_WIN,    test_wins.X_context.shape[0])
    print(f"  scoring windows={N_use}  test windows={N_test}")

    candidate_cols = [c for c in range(df.shape[1]) if c != target_df_col]

    # ---- Load model ----
    model_id = "amazon/chronos-2"
    print(f"\n  Loading {model_id} ...", flush=True)
    extractor = ChronosEmbeddingExtractor(model_id=model_id, layers=LAYERS_KEY, pooling="none")
    extractor.load()

    # ---- Extract / load target embedding ----
    print("  Extracting target embedding ...", flush=True)
    Y_series = train_wins.X_context[:N_use, :, target_df_col]
    target_key = _cache_key(Y_series, LAYERS_KEY)
    target_npz = cache_dir / f"{target_key}.npz"

    if target_npz.exists():
        Y_emb = np.load(target_npz)[f"layer_{LAYER}"]
        print(f"    cache HIT: {Y_emb.shape}")
    else:
        result = extractor.extract_windows(Y_series, layers=LAYERS_KEY)
        Y_emb  = result[LAYER]
        np.savez_compressed(target_npz, **{f"layer_{LAYER}": Y_emb})
        print(f"    computed:  {Y_emb.shape}")

    # ---- Score all 206 candidates ----
    print(f"\n  Scoring {len(candidate_cols)} candidates (window-centered Lagged CKA) ...", flush=True)
    y_cache_wc = precompute_lagged_y_wc(Y_emb, MAX_LAG)
    scores = {}
    missing = 0
    t0 = time.time()
    for i, col in enumerate(candidate_cols):
        X_series = train_wins.X_context[:N_use, :, col]
        cand_key = _cache_key(X_series, LAYERS_KEY)
        cand_npz = cache_dir / f"{cand_key}.npz"
        if cand_npz.exists():
            X_emb = np.load(cand_npz)[f"layer_{LAYER}"]
        else:
            result = extractor.extract_windows(X_series, layers=LAYERS_KEY)
            X_emb  = result[LAYER]
            np.savez_compressed(cand_npz, **{f"layer_{LAYER}": X_emb})
        s, _ = lagged_cka_window_centered(X_emb, Y_emb, MAX_LAG, precomputed_y=y_cache_wc)
        scores[col] = s
    print(f"  Scoring done in {time.time()-t0:.1f}s  (missing cache: {missing})")

    # ---- Select top-5 ----
    top5_cols = sorted(scores, key=scores.__getitem__, reverse=True)[:5]
    print(f"\n  Top-5 selected df_cols: {top5_cols}")
    sensor_id_map = dict(zip(range(df.shape[1]), df.columns))
    print(f"  Top-5 sensor IDs:       {[sensor_id_map.get(c,'?') for c in top5_cols]}")
    print(f"  Scores:                 {[round(scores[c], 6) for c in top5_cols]}")

    # Stop condition: trivial ordering
    if top5_cols == list(range(5)):
        print("\n  STOP: selected_sensors = [0,1,2,3,4] — trivial ordering")
        sys.exit(1)

    # ---- Remove hooks before forecasting ----
    extractor.remove_hooks()

    # ---- Build covariate windows ----
    Y_test_eval = test_wins.X_context[:N_test, :, target_df_col]
    y_test_eval = test_wins.y_target[:N_test]
    cov_wins_test_eval = {
        col: test_wins.X_context[:N_test, :, col]
        for col in candidate_cols
    }

    # ---- Forecast ----
    print(f"\n  Running downstream forecast (K=5, {N_test} test windows) ...", flush=True)
    forecaster = TSFMForecaster(
        pipeline          = extractor._pipe,
        prediction_length = HORIZON,
        cross_learning    = CROSS_LEARNING,
    )
    t_forecast = time.time()
    try:
        metrics = forecaster.evaluate(
            target_windows_test    = Y_test_eval,
            covariate_windows_test = cov_wins_test_eval,
            y_test                 = y_test_eval,
            selected_cols          = top5_cols,
            batch_size             = 64,
        )
    except RuntimeError as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            metrics = forecaster.evaluate(
                target_windows_test    = Y_test_eval,
                covariate_windows_test = cov_wins_test_eval,
                y_test                 = y_test_eval,
                selected_cols          = top5_cols,
                batch_size             = 32,
            )
        else:
            raise
    runtime_forecast = time.time() - t_forecast

    # ---- Validate metrics ----
    rmse = metrics["RMSE"]
    mae  = metrics["MAE"]
    print(f"\n  RMSE={rmse:.4f}  MAE={mae:.4f}  ({runtime_forecast:.1f}s)")

    if not np.isfinite(rmse) or not (1.0 <= rmse <= 100.0):
        print(f"\n  STOP: RMSE={rmse} is NaN or outside [1, 100]")
        sys.exit(1)

    # ---- Save result row ----
    row = {
        "target_sensor_id":       TARGET_SENSOR_ID,
        "target_role":            TARGET_ROLE,
        "method":                 METHOD_NAME,
        "top_k":                  5,
        "layer":                  LAYER,
        "feature_universe_size":  len(candidate_cols),
        "n_covariates":           5,
        "candidate_scope":        f"all_{len(candidate_cols)}",
        "max_scoring_windows_cfg": MAX_SCORING_WIN,
        "n_scoring_windows_used": N_use,
        "n_test_windows_used":    N_test,
        "max_lag":                MAX_LAG,
        "selected_sensors":       json.dumps(top5_cols),
        "repeat_id":              None,
        "is_replicated_baseline": False,
        "RMSE":                   rmse,
        "MAE":                    mae,
        "MAPE":                   metrics.get("MAPE"),
        "mae_per_horizon":        json.dumps([round(x, 6) for x in metrics.get("mae_per_horizon", [])]),
        "rmse_per_horizon":       json.dumps([round(x, 6) for x in metrics.get("rmse_per_horizon", [])]),
        "runtime_seconds":        round(runtime_forecast, 2),
    }
    out_path = out_dir / "sanity_results.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"\n  Saved: {out_path}")

    # ---- GPU cleanup ----
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[sanity] PASS — RMSE={rmse:.4f}  MAE={mae:.4f}  selected={top5_cols}")
    print("Proceed to: request approval for 10-target fixed rerun.")
    sys.exit(0)


if __name__ == "__main__":
    main()
