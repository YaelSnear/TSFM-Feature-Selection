"""Tiny SANITY for CKA layer ablation — technical flow check only.

NOT a scientific experiment. Verifies:
  1. Extraction for layers 6 and 10 works
  2. New cache namespace ({sha}_layers6_10.npz) is safe
  3. Mean_CKA_L6/L10 scores computed
  4. Lagged_CKA_L6/L10_fixed scores computed
  5. selected_sensors are real df_col IDs
  6. Forecasting runs for K=5
  7. Output JSONL created with 4 rows
  8. No NaN/inf scores or RMSE/MAE

SANITY parameters (fixed — do not increase):
  target         = 717469
  layers         = [6, 10]  (L8 NOT re-extracted)
  max_candidates = 20  (first 20 non-target df_cols)
  K              = [5]
  max_scoring    = 50
  max_test       = 20
  max_lag        = 6
  cross_learning = False

Usage:
    conda run --no-capture-output -n yael_env \
        python scripts/sanity_layer_ablation.py \
            --cache_dir outputs/EXP_tsfm_full_run_all206_20260530_172932/cache
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ---- Fixed SANITY parameters ----
TARGET_SENSOR_ID = "717469"
TARGET_ROLE      = "role_A_central"
LAYERS           = [6, 10]
MAX_CANDIDATES   = 20
TOP_K            = 5
MAX_SCORING      = 50
MAX_TEST         = 20
MAX_LAG          = 6
CONTEXT_LENGTH   = 144
HORIZON          = 12
STRIDE           = 12
MAX_WINDOWS      = 9999
TEST_FRAC        = 0.2
CROSS_LEARNING   = False
WINDOW_OVERLAP   = round((CONTEXT_LENGTH - STRIDE) / CONTEXT_LENGTH * 100, 1)


def _cache_key(windows_arr: np.ndarray, layers: list[int]) -> str:
    sha = hashlib.sha1(windows_arr.tobytes()).hexdigest()
    return f"{sha}_layers{'_'.join(str(l) for l in sorted(layers))}"


def _flatten_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, list):
            out[k] = json.dumps([round(x, 6) for x in v])
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="outputs/EXP_tsfm_full_run_all206_20260530_172932/cache",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(
        "outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka_sanity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "sanity_results.jsonl"

    if not cache_dir.exists():
        print(f"ERROR: cache_dir not found: {cache_dir}")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU required.")
        sys.exit(1)

    from src.data.real_traffic import load_metr_la, make_forecast_windows
    from src.evaluation.tsfm_downstream import TSFMForecaster
    from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
    from src.scoring.tsfm_scorers import (
        mean_pooling_cka,
        lagged_cka_window_centered,
        precompute_lagged_y_wc,
    )

    t_start = time.time()
    print(f"[SANITY] target={TARGET_SENSOR_ID}  layers={LAYERS}  "
          f"max_candidates={MAX_CANDIDATES}  K={TOP_K}  "
          f"max_scoring={MAX_SCORING}  max_test={MAX_TEST}  max_lag={MAX_LAG}")

    # ---- Load data ----
    data    = load_metr_la("data/raw/METR-LA.zip")
    df      = data.df
    N_total = len(df)
    n_train = int(N_total * (1.0 - TEST_FRAC))
    gap     = CONTEXT_LENGTH
    n_test_start = n_train + gap
    train_df = df.iloc[:n_train]
    test_df  = df.iloc[n_test_start:]
    sensor_id_map = {i: str(c) for i, c in enumerate(df.columns)}

    sensor_cols = list(df.columns)
    if TARGET_SENSOR_ID not in sensor_cols:
        print(f"ERROR: {TARGET_SENSOR_ID} not found in data")
        sys.exit(1)
    target_df_col = sensor_cols.index(TARGET_SENSOR_ID)

    # ---- Make windows ----
    train_wins = make_forecast_windows(
        train_df, target_sensor=TARGET_SENSOR_ID,
        context_length=CONTEXT_LENGTH, horizon=HORIZON,
        stride=STRIDE, max_windows=MAX_WINDOWS,
    )
    test_wins = make_forecast_windows(
        test_df, target_sensor=TARGET_SENSOR_ID,
        context_length=CONTEXT_LENGTH, horizon=HORIZON,
        stride=STRIDE, max_windows=None,
    )

    N_use  = min(MAX_SCORING, train_wins.X_context.shape[0])
    N_test = min(MAX_TEST, test_wins.X_context.shape[0])
    print(f"  train_wins={train_wins.X_context.shape[0]}  using={N_use}"
          f"  test_wins={test_wins.X_context.shape[0]}  using={N_test}")

    # ---- Candidate cols: first MAX_CANDIDATES non-target ----
    all_nontarget = [c for c in range(df.shape[1]) if c != target_df_col]
    candidate_cols = all_nontarget[:MAX_CANDIDATES]
    print(f"  candidate_cols (first {MAX_CANDIDATES}): {candidate_cols}")

    # ---- Load extractor ----
    model_id  = "amazon/chronos-2"
    print(f"\n  Loading {model_id} (layers={LAYERS}) ...", flush=True)
    extractor = ChronosEmbeddingExtractor(
        model_id=model_id, layers=LAYERS, pooling="none"
    )
    extractor.load()

    # ---- Extract embeddings for target + 20 candidates ----
    print(f"\n  Extracting embeddings (layers={LAYERS}) ...", flush=True)
    all_sensors = [target_df_col] + candidate_cols
    emb_store: dict[int, dict[int, np.ndarray]] = {}  # {col: {layer: emb}}

    t_extract = time.time()
    n_hits = 0
    n_miss = 0
    for col in all_sensors:
        series  = train_wins.X_context[:N_use, :, col]
        key     = _cache_key(series, LAYERS)
        npz_path = cache_dir / f"{key}.npz"
        if npz_path.exists():
            data_npz = np.load(npz_path)
            emb_store[col] = {l: data_npz[f"layer_{l}"] for l in LAYERS}
            n_hits += 1
        else:
            result = extractor.extract_windows(series, layers=LAYERS)
            emb_store[col] = result
            np.savez_compressed(npz_path, **{f"layer_{l}": result[l] for l in LAYERS})
            n_miss += 1

    print(f"  Extraction: {n_miss} computed, {n_hits} cache hits  "
          f"({time.time()-t_extract:.1f}s)")

    # Confirm new cache namespace
    sample_key = _cache_key(train_wins.X_context[:N_use, :, target_df_col], LAYERS)
    sample_old = f"{sample_key.split('_layers')[0]}_layers8.npz"
    new_npz    = cache_dir / f"{sample_key}.npz"
    old_npz    = cache_dir / sample_old
    print(f"\n  Cache namespace check:")
    print(f"    New file: {new_npz.name}  exists={new_npz.exists()}")
    print(f"    Old L8:   {old_npz.name}  exists={old_npz.exists()}")
    if new_npz.name == old_npz.name:
        print("  ERROR: new and old cache keys are identical — namespace conflict!")
        sys.exit(1)
    print("  Cache namespace: SAFE (different files)")

    Y_emb_by_layer = {l: emb_store[target_df_col][l] for l in LAYERS}

    # ---- Score all candidates per layer ----
    scores_by_layer: dict[int, dict[str, dict[int, float]]] = {}
    lags_by_layer:   dict[int, dict[int, int]] = {}

    errors = []

    for layer in LAYERS:
        Y_emb   = Y_emb_by_layer[layer]
        y_cache = precompute_lagged_y_wc(Y_emb, MAX_LAG)

        mc_scores: dict[int, float] = {}
        lc_scores: dict[int, float] = {}
        lc_lags:   dict[int, int]   = {}

        for col in candidate_cols:
            X_emb = emb_store[col][layer]
            mc_scores[col] = mean_pooling_cka(X_emb, Y_emb)
            s, l = lagged_cka_window_centered(X_emb, Y_emb, MAX_LAG, precomputed_y=y_cache)
            lc_scores[col] = s
            lc_lags[col]   = l

        scores_by_layer[layer] = {
            f"Mean_CKA_L{layer}":          mc_scores,
            f"Lagged_CKA_L{layer}_fixed":  lc_scores,
        }
        lags_by_layer[layer] = lc_lags

        # Checks
        for method_name, sc in scores_by_layer[layer].items():
            vals = np.array(list(sc.values()))
            if np.any(~np.isfinite(vals)):
                errors.append(f"{method_name}: NaN/inf scores")
            if vals.std() <= 1e-6:
                errors.append(f"{method_name}: degenerate (std={vals.std():.2e})")

        print(f"\n  Layer {layer}:")
        for method_name, sc in scores_by_layer[layer].items():
            vals = np.array(list(sc.values()))
            print(f"    {method_name:<30}  min={vals.min():.4f}  max={vals.max():.4f}"
                  f"  std={vals.std():.4f}")

    if errors:
        print(f"\n  STOP: {errors}")
        sys.exit(1)

    # ---- Remove hooks before forecasting ----
    extractor.remove_hooks()

    # ---- Build test data ----
    Y_test_eval = test_wins.X_context[:N_test, :, target_df_col]
    y_test_eval = test_wins.y_target[:N_test]
    cov_wins_test = {col: test_wins.X_context[:N_test, :, col] for col in candidate_cols}

    forecaster = TSFMForecaster(
        pipeline=extractor._pipe, prediction_length=HORIZON,
        cross_learning=CROSS_LEARNING,
    )

    # ---- Forecast for each (layer, method) ----
    rows = []
    print(f"\n  Forecasting (K={TOP_K}, {N_test} test windows) ...", flush=True)

    for layer in LAYERS:
        for method_name, sc in scores_by_layer[layer].items():
            top_k_cols = sorted(sc, key=sc.__getitem__, reverse=True)[:TOP_K]
            sel_ids    = [sensor_id_map.get(c, "?") for c in top_k_cols]

            # Check: not trivial ordering
            if top_k_cols == list(range(TOP_K)):
                print(f"  [warn] {method_name}: selected = {top_k_cols} (trivial ordering)")

            t_f = time.time()
            metrics = forecaster.evaluate(
                target_windows_test    = Y_test_eval,
                covariate_windows_test = cov_wins_test,
                y_test                 = y_test_eval,
                selected_cols          = top_k_cols,
                batch_size             = 64,
            )
            rt = round(time.time() - t_f, 2)

            rmse = metrics["RMSE"]
            mae  = metrics["MAE"]
            print(f"    {method_name:<30}  sel={top_k_cols}  "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  ({rt}s)")

            if not np.isfinite(rmse) or not (1.0 <= rmse <= 100.0):
                errors.append(f"{method_name}: RMSE={rmse} implausible")
                continue

            row = {
                "target_sensor_id":       TARGET_SENSOR_ID,
                "target_role":            TARGET_ROLE,
                "downstream_model":       "Chronos-2",
                "frozen":                 True,
                "feature_universe_size":  MAX_CANDIDATES,
                "context_length":         CONTEXT_LENGTH,
                "stride":                 STRIDE,
                "window_overlap_pct":     WINDOW_OVERLAP,
                "max_scoring_windows_cfg": MAX_SCORING,
                "n_scoring_windows_used": N_use,
                "max_test_windows_cfg":   MAX_TEST,
                "n_test_windows_used":    N_test,
                "method":                 method_name,
                "top_k":                  TOP_K,
                "layer":                  layer,
                "mode":                   "scored",
                "n_covariates":           TOP_K,
                "candidate_scope":        f"first_{MAX_CANDIDATES}",
                "baseline_scope":         "N/A",
                "selected_sensors":       json.dumps(top_k_cols),
                "selected_sensor_ids":    json.dumps(sel_ids),
                "repeat_id":              None,
                "is_replicated_baseline": False,
                "runtime_seconds":        rt,
                "max_lag":                MAX_LAG,
                **_flatten_metrics(metrics),
            }
            rows.append(row)

    # ---- Cleanup ----
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Final checks ----
    if errors:
        print(f"\n  STOP conditions triggered: {errors}")
        sys.exit(1)
    if len(rows) != 4:
        print(f"  ERROR: expected 4 rows, got {len(rows)}")
        sys.exit(1)

    # ---- Write JSONL ----
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    # ---- Report ----
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SANITY COMPLETE  ({t_total:.0f}s)")
    print(f"{'='*60}")
    print(f"  Output: {jsonl_path}  ({len(rows)} rows)")
    print(f"\n  Results:")
    print(f"  {'method':<32}  {'layer':>6}  {'sel_sensors':>35}  {'RMSE':>8}  {'MAE':>8}")
    for r in rows:
        sel = json.loads(r['selected_sensors'])
        print(f"  {r['method']:<32}  {r['layer']:>6}  {str(sel):>35}"
              f"  {r['RMSE']:>8.4f}  {r['MAE']:>8.4f}")

    print(f"\n  Score distributions:")
    for layer in LAYERS:
        for method_name, sc in scores_by_layer[layer].items():
            vals = np.array(list(sc.values()))
            print(f"    {method_name:<32}  min={vals.min():.4f}  "
                  f"max={vals.max():.4f}  std={vals.std():.4f}")

    print(f"\n  best_lag distribution (Lagged_CKA_fixed):")
    for layer in LAYERS:
        lags = list(lags_by_layer[layer].values())
        lag_cnt = dict(sorted(Counter(lags).items()))
        n_lag0 = lag_cnt.get(0, 0)
        max_valid = min(MAX_LAG, 11 - 1) - 1  # boundary lag
        n_bnd  = lag_cnt.get(max_valid, 0) + lag_cnt.get(-max_valid, 0)
        n_other = len(lags) - n_lag0 - n_bnd
        print(f"    L{layer}: {lag_cnt}  (lag=0: {n_lag0}  boundary: {n_bnd}"
              f"  intermediate: {n_other})")

    print(f"\n  Cache namespace: {cache_dir.name}/{_cache_key(train_wins.X_context[:N_use,:,target_df_col], LAYERS)}.npz")
    print(f"  Safe to launch FULL SBATCH: YES (all checks passed)")


if __name__ == "__main__":
    main()
