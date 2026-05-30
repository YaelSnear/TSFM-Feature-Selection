"""Full CKA layer ablation: layers [6, 8, 10] on all 10 FULL RUN targets.

Runs Mean_CKA and Lagged_CKA_fixed for layers [6, 10] (new).
Copies existing Mean_CKA_L8 and Lagged_CKA_L8_fixed rows from existing files.
Reuses embedding cache where possible.

Does NOT overwrite any existing FULL RUN files.

Usage (via SBATCH — do not run interactively for full run):
    python scripts/run_layer_ablation_cka.py \\
        --cache_dir  outputs/EXP_tsfm_full_run_all206_20260530_172932/cache \\
        --orig_jsonl outputs/EXP_tsfm_full_run_all206_20260530_172932/results/results_incremental.jsonl \\
        --fixed_jsonl outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun/results_incremental_fixed.jsonl \\
        --out_dir    outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka
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
import pandas as pd

# ---- Parameters matching FULL RUN exactly ----
LAYERS_EXTRACT  = [6, 8, 10]    # extracted together; L8 used for caching only
LAYERS_NEW      = [6, 10]       # new scoring + forecasting only
MAX_SCORING     = 1000
MAX_LAG         = 24
TOP_KS          = [5, 10, 20]
TEST_FRAC       = 0.2
N_PER_ROLE      = 2
CONTEXT_LENGTH  = 144
HORIZON         = 12
STRIDE          = 12
MAX_WINDOWS     = 9999
CROSS_LEARNING  = False
WINDOW_OVERLAP  = round((CONTEXT_LENGTH - STRIDE) / CONTEXT_LENGTH * 100, 1)


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


def _append_row(row: dict, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",   required=True)
    parser.add_argument("--orig_jsonl",  required=True)
    parser.add_argument("--fixed_jsonl", required=True)
    parser.add_argument("--out_dir",     required=True)
    args = parser.parse_args()

    cache_dir   = Path(args.cache_dir)
    orig_path   = Path(args.orig_jsonl)
    fixed_path  = Path(args.fixed_jsonl)
    out_dir     = Path(args.out_dir)
    score_dir   = out_dir / "layer_ablation_score_tables"
    log_dir     = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    jsonl_path  = out_dir / "layer_ablation_results.jsonl"

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    from src.data.real_traffic import load_metr_la, make_forecast_windows
    from src.data.target_selection import select_target_sensors
    from src.evaluation.tsfm_downstream import TSFMForecaster
    from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
    from src.scoring.tsfm_scorers import (
        mean_pooling_cka,
        lagged_cka_window_centered,
        precompute_lagged_y_wc,
    )

    t_exp = time.time()

    # ---- Phase 0: Copy existing L8 rows ----
    existing_keys: set[tuple] = set()   # (target, method, top_k) already written

    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    existing_keys.add((str(r["target_sensor_id"]), str(r["method"]), str(r["top_k"])))
                except Exception:
                    pass
        print(f"[resume] {len(existing_keys)} rows already in JSONL")

    # Copy Mean_CKA_L8 from original JSONL (read-only)
    n_copied = 0
    if orig_path.exists():
        with open(orig_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("method") == "Mean_CKA_L8" and not r.get("is_replicated_baseline", False):
                    key = (str(r["target_sensor_id"]), "Mean_CKA_L8", str(r["top_k"]))
                    if key not in existing_keys:
                        r["source"] = "existing_full_run"
                        _append_row(r, jsonl_path)
                        existing_keys.add(key)
                        n_copied += 1
    print(f"[phase0] Copied {n_copied} Mean_CKA_L8 rows from original JSONL")

    # Copy Lagged_CKA_L8_fixed from fixed rerun JSONL (read-only)
    n_copied2 = 0
    if fixed_path.exists():
        with open(fixed_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("method") == "Lagged_CKA_L8_fixed":
                    key = (str(r["target_sensor_id"]), "Lagged_CKA_L8_fixed", str(r["top_k"]))
                    if key not in existing_keys:
                        r["source"] = "existing_lagged_cka_fixed_rerun"
                        _append_row(r, jsonl_path)
                        existing_keys.add(key)
                        n_copied2 += 1
    print(f"[phase0] Copied {n_copied2} Lagged_CKA_L8_fixed rows from fixed rerun JSONL")

    # ---- Phase 1: Setup ----
    print("\nLoading METR-LA ...", flush=True)
    data    = load_metr_la("data/raw/METR-LA.zip")
    df      = data.df
    N_total = len(df)
    n_train = int(N_total * (1.0 - TEST_FRAC))
    n_test_start = n_train + CONTEXT_LENGTH
    train_df = df.iloc[:n_train]
    test_df  = df.iloc[n_test_start:]
    sensor_id_map = {i: str(c) for i, c in enumerate(df.columns)}

    role_dict, _, _ = select_target_sensors(data, train_df, n_per_role=N_PER_ROLE, seed=42)
    target_list = [s for sensors in role_dict.values() for s in sensors]
    print(f"Targets ({len(target_list)}): {[t['sensor_id'] for t in target_list]}")

    print(f"\nLoading Chronos-2 (layers={LAYERS_EXTRACT}) ...", flush=True)
    extractor = ChronosEmbeddingExtractor(
        model_id="amazon/chronos-2", layers=LAYERS_EXTRACT, pooling="none"
    )
    extractor.load()

    # ---- Phase 2: Per-target loop ----
    for t_idx, target_info in enumerate(target_list):
        target_sensor_id = str(target_info["sensor_id"])
        target_role      = target_info["role"]
        target_df_col    = target_info["df_col"]
        t_target = time.time()
        print(f"\n[{t_idx+1}/{len(target_list)}] {target_sensor_id} ({target_role})",
              flush=True)

        # Check if all new work done for this target
        new_methods = ([f"Mean_CKA_L{l}" for l in LAYERS_NEW] +
                       [f"Lagged_CKA_L{l}_fixed" for l in LAYERS_NEW])
        all_done = all(
            (target_sensor_id, m, str(k)) in existing_keys
            for m in new_methods for k in TOP_KS
        )
        if all_done:
            print("  [RESUME] all new rows done — skipping", flush=True)
            continue

        # Make windows
        train_wins = make_forecast_windows(
            train_df, target_sensor=target_sensor_id,
            context_length=CONTEXT_LENGTH, horizon=HORIZON,
            stride=STRIDE, max_windows=MAX_WINDOWS,
        )
        test_wins = make_forecast_windows(
            test_df, target_sensor=target_sensor_id,
            context_length=CONTEXT_LENGTH, horizon=HORIZON,
            stride=STRIDE, max_windows=None,
        )
        N_use  = min(MAX_SCORING, train_wins.X_context.shape[0])
        N_test = test_wins.X_context.shape[0]
        print(f"  scoring={N_use}  test={N_test}", flush=True)

        candidate_cols        = [c for c in range(df.shape[1]) if c != target_df_col]
        feature_universe_size = len(candidate_cols)
        if feature_universe_size != 206:
            print(f"  STOP: feature_universe_size={feature_universe_size}, expected 206")
            sys.exit(1)

        # Extract embeddings at LAYERS_EXTRACT = [6,8,10] together
        print(f"  Extracting {len(candidate_cols)+1} sensors at layers {LAYERS_EXTRACT} ...",
              flush=True)
        t_extract = time.time()
        emb_store: dict[int, dict[int, np.ndarray]] = {}  # {col: {layer: emb}}
        n_hits = n_miss = 0

        for col in [target_df_col] + candidate_cols:
            series   = train_wins.X_context[:N_use, :, col]
            key      = _cache_key(series, LAYERS_EXTRACT)
            npz_path = cache_dir / f"{key}.npz"
            if npz_path.exists():
                d = np.load(npz_path)
                emb_store[col] = {l: d[f"layer_{l}"] for l in LAYERS_EXTRACT}
                n_hits += 1
            else:
                result = extractor.extract_windows(series, layers=LAYERS_EXTRACT)
                emb_store[col] = result
                np.savez_compressed(npz_path, **{f"layer_{l}": result[l] for l in LAYERS_EXTRACT})
                n_miss += 1

        print(f"  Extraction done in {time.time()-t_extract:.1f}s  "
              f"(hits={n_hits}  misses={n_miss})", flush=True)

        # Score per new layer
        all_layer_scores: dict[str, dict[int, float]] = {}  # method_name -> {col: score}
        all_layer_lags:   dict[str, dict[int, int]]   = {}
        all_score_rows:   list[dict] = []

        for layer in LAYERS_NEW:
            Y_emb   = emb_store[target_df_col][layer]
            y_cache = precompute_lagged_y_wc(Y_emb, MAX_LAG)

            mc_name = f"Mean_CKA_L{layer}"
            lc_name = f"Lagged_CKA_L{layer}_fixed"

            mc_scores: dict[int, float] = {}
            lc_scores: dict[int, float] = {}
            lc_lags:   dict[int, int]   = {}

            print(f"  Scoring layer {layer} ...", flush=True)
            t_score = time.time()
            for col in candidate_cols:
                X_emb = emb_store[col][layer]
                mc_scores[col] = mean_pooling_cka(X_emb, Y_emb)
                s, l = lagged_cka_window_centered(X_emb, Y_emb, MAX_LAG, precomputed_y=y_cache)
                lc_scores[col] = s
                lc_lags[col]   = l

            elapsed = time.time() - t_score
            mc_arr = np.array(list(mc_scores.values()))
            lc_arr = np.array(list(lc_scores.values()))
            print(f"    {mc_name}: min={mc_arr.min():.4f}  max={mc_arr.max():.4f}"
                  f"  std={mc_arr.std():.4f}  ({elapsed:.1f}s)", flush=True)
            print(f"    {lc_name}: min={lc_arr.min():.4f}  max={lc_arr.max():.4f}"
                  f"  std={lc_arr.std():.4f}", flush=True)

            # Degeneracy checks
            for name, arr in [(mc_name, mc_arr), (lc_name, lc_arr)]:
                if np.any(~np.isfinite(arr)):
                    print(f"  STOP: {name} has NaN/inf")
                    sys.exit(1)
                if arr.std() <= 1e-6:
                    print(f"  STOP: {name} degenerate (std={arr.std():.2e})")
                    sys.exit(1)

            all_layer_scores[mc_name] = mc_scores
            all_layer_scores[lc_name] = lc_scores
            all_layer_lags[lc_name]   = lc_lags

            # Build per-target score tables for this layer
            mc_order = sorted(mc_scores, key=mc_scores.__getitem__, reverse=True)
            for rank_i, c in enumerate(mc_order):
                all_score_rows.append({
                    "target_sensor_id": target_sensor_id, "target_role": target_role,
                    "layer": layer, "method": mc_name,
                    "candidate_df_col": c, "candidate_sensor_id": sensor_id_map.get(c, "?"),
                    "score": mc_scores[c], "rank": rank_i + 1, "best_lag": float("nan"),
                    "n_scoring_windows_used": N_use, "max_lag": MAX_LAG,
                })
            lc_order = sorted(lc_scores, key=lc_scores.__getitem__, reverse=True)
            for rank_i, c in enumerate(lc_order):
                all_score_rows.append({
                    "target_sensor_id": target_sensor_id, "target_role": target_role,
                    "layer": layer, "method": lc_name,
                    "candidate_df_col": c, "candidate_sensor_id": sensor_id_map.get(c, "?"),
                    "score": lc_scores[c], "rank": rank_i + 1, "best_lag": lc_lags[c],
                    "n_scoring_windows_used": N_use, "max_lag": MAX_LAG,
                })

        # Save per-target score table
        if all_score_rows:
            sdf = pd.DataFrame(all_score_rows)
            sdf.to_csv(score_dir / f"{target_sensor_id}_L6_L10.csv", index=False)

        # Remove hooks before forecasting
        extractor.remove_hooks()

        forecaster = TSFMForecaster(
            pipeline=extractor._pipe, prediction_length=HORIZON,
            cross_learning=CROSS_LEARNING,
        )
        Y_test_eval = test_wins.X_context[:N_test, :, target_df_col]
        y_test_eval = test_wins.y_target[:N_test]
        cov_wins_test = {col: test_wins.X_context[:N_test, :, col] for col in candidate_cols}

        # Forecast per new (method, K)
        for method_name, sc in all_layer_scores.items():
            layer = int(method_name.split("_L")[1].split("_")[0])
            sc_order = sorted(sc, key=sc.__getitem__, reverse=True)

            # best_lag summary
            if method_name in all_layer_lags:
                lag_dist = dict(sorted(Counter(all_layer_lags[method_name].values()).items())[:5])
            else:
                lag_dist = {}

            mc_arr = np.array(list(sc.values()))
            row_base = {
                "target_sensor_id":       target_sensor_id,
                "target_role":            target_role,
                "downstream_model":       "Chronos-2",
                "frozen":                 True,
                "feature_universe_size":  feature_universe_size,
                "context_length":         CONTEXT_LENGTH,
                "stride":                 STRIDE,
                "window_overlap_pct":     WINDOW_OVERLAP,
                "max_scoring_windows_cfg": MAX_SCORING,
                "n_scoring_windows_used": N_use,
                "max_test_windows_cfg":   None,
                "n_test_windows_used":    N_test,
                "method":                 method_name,
                "layer":                  layer,
                "mode":                   "scored",
                "candidate_scope":        f"all_{feature_universe_size}",
                "baseline_scope":         "N/A",
                "repeat_id":              None,
                "is_replicated_baseline": False,
                "source":                 "layer_ablation_new",
                "score_min":              float(mc_arr.min()),
                "score_max":              float(mc_arr.max()),
                "score_mean":             float(mc_arr.mean()),
                "score_std":              float(mc_arr.std()),
                "best_lag_distribution":  json.dumps(lag_dist),
                "max_lag":                MAX_LAG,
            }

            for top_k in TOP_KS:
                key = (target_sensor_id, method_name, str(top_k))
                if key in existing_keys:
                    print(f"  [RESUME] {method_name} K={top_k}", flush=True)
                    continue

                top_k_cols = sc_order[:top_k]
                if top_k_cols == list(range(top_k)):
                    print(f"  STOP: {method_name} K={top_k} selected trivially {top_k_cols}")
                    sys.exit(1)

                t_f = time.time()
                try:
                    metrics = forecaster.evaluate(
                        target_windows_test    = Y_test_eval,
                        covariate_windows_test = cov_wins_test,
                        y_test                 = y_test_eval,
                        selected_cols          = top_k_cols,
                        batch_size             = 256,
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e) or "memory" in str(e).lower():
                        torch.cuda.empty_cache(); gc.collect()
                        metrics = forecaster.evaluate(
                            target_windows_test    = Y_test_eval,
                            covariate_windows_test = cov_wins_test,
                            y_test                 = y_test_eval,
                            selected_cols          = top_k_cols,
                            batch_size             = 64,
                        )
                    else:
                        raise
                rt = round(time.time() - t_f, 2)

                rmse = metrics["RMSE"]
                if not np.isfinite(rmse) or not (1.0 <= rmse <= 100.0):
                    print(f"  STOP: {method_name} K={top_k} RMSE={rmse} implausible")
                    sys.exit(1)

                print(f"  {method_name} K={top_k}: RMSE={rmse:.4f}  "
                      f"MAE={metrics['MAE']:.4f}  ({rt}s)", flush=True)

                row = {
                    **row_base,
                    "top_k":              top_k,
                    "n_covariates":       top_k,
                    "selected_sensors":   json.dumps(top_k_cols),
                    "selected_sensor_ids": json.dumps(
                        [sensor_id_map.get(c, "?") for c in top_k_cols]
                    ),
                    "runtime_seconds":    rt,
                    **_flatten_metrics(metrics),
                }
                _append_row(row, jsonl_path)
                existing_keys.add(key)

        # GPU cleanup + re-register hooks
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if t_idx < len(target_list) - 1:
            blocks = extractor._find_blocks()
            extractor._register_hooks(blocks, LAYERS_EXTRACT)

        print(f"  Target done in {time.time()-t_target:.1f}s", flush=True)

    # ---- Build combined all_scores.csv ----
    score_files = list(score_dir.glob("*_L6_L10.csv"))
    if score_files:
        combined_scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)
        combined_scores.to_csv(score_dir / "all_scores.csv", index=False)
        print(f"\nSaved combined score table: {score_dir / 'all_scores.csv'}")

    # ---- Final validation ----
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    n_rows    = len(rows)
    n_targets = len(set(r["target_sensor_id"] for r in rows))
    methods   = sorted(set(r["method"] for r in rows))
    any_nan   = any(not np.isfinite(r.get("RMSE", float("nan"))) for r in rows)
    t_total   = time.time() - t_exp

    print(f"\n[validation] rows={n_rows} (expected 180)  targets={n_targets}  "
          f"methods={methods}  any_nan={any_nan}")
    if n_rows != 180:
        print(f"  WARNING: expected 180 rows, got {n_rows}")
    if any_nan:
        print("  ERROR: NaN in RMSE")
        sys.exit(1)
    print(f"\nDone. Total: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"Results: {jsonl_path}")


if __name__ == "__main__":
    main()
