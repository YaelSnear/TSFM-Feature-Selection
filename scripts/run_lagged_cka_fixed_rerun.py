"""Full 10-target rerun of corrected window-centered Lagged CKA.

Runs ONLY Lagged_CKA_L8_fixed for all 10 role-selected targets with the exact
same settings as the original FULL RUN. Reuses the existing embedding cache.

Does NOT rerun: target_only, all_features_206, random_k, Pearson,
SparseLinear_L1, RandomForest, or Mean_CKA_L8.

Does NOT write to the original results_incremental.jsonl.

Output rows are fully schema-compatible with the original FULL RUN.

Usage:
    conda run --no-capture-output -n yael_env \
        python scripts/run_lagged_cka_fixed_rerun.py \
            --cache_dir outputs/EXP_tsfm_full_run_all206_20260530_172932/cache \
            --out_dir   outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun
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
import pandas as pd

# ---- Parameters matching the original FULL RUN exactly ----
MAX_SCORING_WIN  = 1000
MAX_LAG          = 24
LAYER            = 8
LAYERS_KEY       = [LAYER]
TOP_KS           = [5, 10, 20]
TEST_FRAC        = 0.2
N_PER_ROLE       = 2
CONTEXT_LENGTH   = 144
HORIZON          = 12
STRIDE           = 12
MAX_WINDOWS      = 9999
CROSS_LEARNING   = False
METHOD_NAME      = "Lagged_CKA_L8_fixed"
WINDOW_OVERLAP   = round((CONTEXT_LENGTH - STRIDE) / CONTEXT_LENGTH * 100, 1)  # 91.7


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
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument(
        "--out_dir",
        default="outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    from src.data.real_traffic import load_metr_la, make_forecast_windows
    from src.data.target_selection import select_target_sensors
    from src.evaluation.tsfm_downstream import TSFMForecaster
    from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
    from src.scoring.tsfm_scorers import lagged_cka_window_centered, precompute_lagged_y_wc

    jsonl_path = out_dir / "results_incremental_fixed.jsonl"

    # ---- Resume: load completed (target, top_k) pairs ----
    completed_keys: set[tuple] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed_keys.add((str(r["target_sensor_id"]), str(r["top_k"])))
                except Exception:
                    pass
        print(f"[resume] {len(completed_keys)} completed (target, K) pairs")

    # ---- Load data ----
    print("Loading METR-LA ...", flush=True)
    data    = load_metr_la("data/raw/METR-LA.zip")
    df      = data.df
    N_total = len(df)
    n_train = int(N_total * (1.0 - TEST_FRAC))
    gap     = CONTEXT_LENGTH
    n_test_start = n_train + gap
    train_df = df.iloc[:n_train]
    test_df  = df.iloc[n_test_start:]
    sensor_id_map = {i: str(c) for i, c in enumerate(df.columns)}
    print(f"  N_total={N_total}  n_train={n_train}  n_test_start={n_test_start}")

    # ---- Select same 10 targets (seed=42, n_per_role=2) ----
    role_dict, _, _ = select_target_sensors(data, train_df, n_per_role=N_PER_ROLE, seed=42)
    target_list = [s for sensors in role_dict.values() for s in sensors]
    print(f"Targets ({len(target_list)}): {[t['sensor_id'] for t in target_list]}")

    # ---- Load Chronos-2 ----
    model_id = "amazon/chronos-2"
    print(f"Loading {model_id} ...", flush=True)
    extractor = ChronosEmbeddingExtractor(model_id=model_id, layers=LAYERS_KEY, pooling="none")
    extractor.load()

    t_exp = time.time()

    for t_idx, target_info in enumerate(target_list):
        target_sensor_id = str(target_info["sensor_id"])
        target_role      = target_info["role"]
        target_df_col    = target_info["df_col"]
        t_target = time.time()
        print(f"\n[{t_idx+1}/{len(target_list)}] {target_sensor_id} ({target_role})", flush=True)

        # Check if all K values already complete
        if all((target_sensor_id, str(k)) in completed_keys for k in TOP_KS):
            print("  [RESUME] all K values done — skipping")
            continue

        # ---- Make windows ----
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

        N_train_orig = train_wins.X_context.shape[0]
        N_test_orig  = test_wins.X_context.shape[0]
        N_use        = min(MAX_SCORING_WIN, N_train_orig)
        N_test       = N_test_orig
        print(f"  train={N_train_orig}  scoring_use={N_use}  test={N_test}", flush=True)

        candidate_cols        = [c for c in range(df.shape[1]) if c != target_df_col]
        feature_universe_size = len(candidate_cols)

        # ---- Load / extract target embedding ----
        Y_series  = train_wins.X_context[:N_use, :, target_df_col]
        tgt_key   = _cache_key(Y_series, LAYERS_KEY)
        tgt_npz   = cache_dir / f"{tgt_key}.npz"
        if tgt_npz.exists():
            Y_emb = np.load(tgt_npz)[f"layer_{LAYER}"]
            print(f"  target emb: cache HIT {Y_emb.shape}", flush=True)
        else:
            result = extractor.extract_windows(Y_series, layers=LAYERS_KEY)
            Y_emb  = result[LAYER]
            np.savez_compressed(tgt_npz, **{f"layer_{LAYER}": Y_emb})
            print(f"  target emb: computed   {Y_emb.shape}", flush=True)

        # ---- Score all 206 candidates ----
        print(f"  Scoring {len(candidate_cols)} candidates ...", flush=True)
        y_cache_wc = precompute_lagged_y_wc(Y_emb, MAX_LAG)
        scores     = {}
        best_lags  = {}
        n_cache_miss = 0
        t_score = time.time()

        for col in candidate_cols:
            X_series = train_wins.X_context[:N_use, :, col]
            cand_key = _cache_key(X_series, LAYERS_KEY)
            cand_npz = cache_dir / f"{cand_key}.npz"
            if cand_npz.exists():
                X_emb = np.load(cand_npz)[f"layer_{LAYER}"]
            else:
                result  = extractor.extract_windows(X_series, layers=LAYERS_KEY)
                X_emb   = result[LAYER]
                np.savez_compressed(cand_npz, **{f"layer_{LAYER}": X_emb})
                n_cache_miss += 1

            s, l = lagged_cka_window_centered(X_emb, Y_emb, MAX_LAG, precomputed_y=y_cache_wc)
            scores[col]    = s
            best_lags[col] = l

        if n_cache_miss > 0:
            print(f"  [warn] {n_cache_miss} cache misses (embeddings recomputed and cached)")

        score_values = np.array(list(scores.values()))
        print(f"  Scoring done in {time.time()-t_score:.1f}s  "
              f"score=[{score_values.min():.4f},{score_values.max():.4f}] "
              f"std={score_values.std():.4f}", flush=True)

        # Hard stop: degenerate scores
        if score_values.std() <= 1e-6:
            print(f"  STOP: degenerate scores (std={score_values.std():.2e})")
            sys.exit(1)
        if np.any(~np.isfinite(score_values)):
            print("  STOP: NaN or inf in scores")
            sys.exit(1)
        if feature_universe_size != 206:
            print(f"  STOP: feature_universe_size={feature_universe_size}, expected 206")
            sys.exit(1)

        # ---- Save per-target score table ----
        score_order = sorted(scores, key=scores.__getitem__, reverse=True)
        score_rows = [
            {
                "target_sensor_id":       target_sensor_id,
                "target_role":            target_role,
                "candidate_df_col":       c,
                "candidate_sensor_id":    sensor_id_map.get(c, "?"),
                "score":                  scores[c],
                "rank":                   rank_i + 1,
                "best_lag":               best_lags[c],
                "n_scoring_windows_used": N_use,
                "max_lag":                MAX_LAG,
                "layer":                  LAYER,
            }
            for rank_i, c in enumerate(score_order)
        ]
        pd.DataFrame(score_rows).to_csv(
            out_dir / f"lagged_cka_fixed_scores_{target_sensor_id}.csv", index=False
        )

        # ---- Remove hooks before forecasting ----
        extractor.remove_hooks()

        forecaster = TSFMForecaster(
            pipeline=extractor._pipe, prediction_length=HORIZON,
            cross_learning=CROSS_LEARNING,
        )

        Y_test_eval = test_wins.X_context[:N_test, :, target_df_col]
        y_test_eval = test_wins.y_target[:N_test]
        cov_wins_test = {col: test_wins.X_context[:N_test, :, col] for col in candidate_cols}

        # Best-lag distribution summary (top 3 lags)
        from collections import Counter
        lag_counts = Counter(best_lags.values())
        lag_summary = dict(sorted(lag_counts.items(), key=lambda x: -x[1])[:5])

        # ---- Forecast for each K ----
        row_base = {
            "target_sensor_id":       target_sensor_id,
            "target_role":            target_role,
            "downstream_model":       "Chronos-2",
            "frozen":                 True,
            "feature_universe_size":  feature_universe_size,
            "context_length":         CONTEXT_LENGTH,
            "stride":                 STRIDE,
            "window_overlap_pct":     WINDOW_OVERLAP,
            "max_scoring_windows_cfg": MAX_SCORING_WIN,
            "n_scoring_windows_used": N_use,
            "max_test_windows_cfg":   None,
            "n_test_windows_used":    N_test,
            "method":                 METHOD_NAME,
            "layer":                  LAYER,
            "mode":                   "scored",
            "candidate_scope":        f"all_{feature_universe_size}",
            "baseline_scope":         "N/A",
            "repeat_id":              None,
            "is_replicated_baseline": False,
            "score_min":              float(score_values.min()),
            "score_max":              float(score_values.max()),
            "score_mean":             float(score_values.mean()),
            "score_std":              float(score_values.std()),
            "best_lag_distribution":  json.dumps(lag_summary),
        }

        for top_k in TOP_KS:
            resume_key = (target_sensor_id, str(top_k))
            if resume_key in completed_keys:
                print(f"  [RESUME] K={top_k} done", flush=True)
                continue

            top_k_cols = score_order[:top_k]

            # Hard stop: trivial ordering
            if top_k_cols == list(range(top_k)):
                print(f"  STOP: selected_sensors = {top_k_cols} — trivial ordering at K={top_k}")
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
                    torch.cuda.empty_cache()
                    gc.collect()
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

            # Hard stop: bad RMSE
            rmse = metrics["RMSE"]
            if not (np.isfinite(rmse) and 1.0 <= rmse <= 100.0):
                print(f"  STOP: RMSE={rmse} implausible at K={top_k}")
                sys.exit(1)

            print(f"  K={top_k}: RMSE={rmse:.4f}  MAE={metrics['MAE']:.4f}  ({rt}s)",
                  flush=True)

            row = {
                **row_base,
                "top_k":              top_k,
                "n_covariates":       top_k,
                "selected_sensors":   json.dumps(top_k_cols),
                "selected_sensor_ids": json.dumps([sensor_id_map.get(c, "?") for c in top_k_cols]),
                "runtime_seconds":    rt,
                **_flatten_metrics(metrics),
            }
            _append_row(row, jsonl_path)
            completed_keys.add(resume_key)

        # ---- GPU cleanup + re-register hooks ----
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if t_idx < len(target_list) - 1:
            blocks = extractor._find_blocks()
            extractor._register_hooks(blocks, LAYERS_KEY)

        print(f"  Target done in {time.time()-t_target:.1f}s", flush=True)

    # ---- Final validation ----
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    n_rows    = len(rows)
    n_targets = len(set(r["target_sensor_id"] for r in rows))
    any_nan   = any(not np.isfinite(r["RMSE"]) or not np.isfinite(r["MAE"]) for r in rows)
    print(f"\n[validation] rows={n_rows}  targets={n_targets}  any_nan={any_nan}")
    if n_rows != 30:
        print(f"  WARNING: expected 30 rows, got {n_rows}")
    if n_targets != 10:
        print(f"  WARNING: expected 10 targets, got {n_targets}")
    if any_nan:
        print("  ERROR: NaN in RMSE/MAE")
        sys.exit(1)

    t_total = time.time() - t_exp
    print(f"\nDone. Total: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"Results: {jsonl_path}")


if __name__ == "__main__":
    main()
