"""TSFM Feature Selection — Frozen Chronos-2 Downstream Experiment.

Research question: Can a model-aware FS method select a small Top-K subset from all
available non-target sensors and improve or compete with using all features in a frozen TSFM?

Two run modes controlled by config:
  SANITY  — technical verification only (max_candidates cap applied)
  FULL RUN — scientific experiment (all 206 non-target sensors)

Window overlap note: context_length=144, stride=12 → 91.7% overlap between adjacent
windows. Windows are not independent statistical samples. Paired statistical unit = target sensor.

Usage:
    conda activate yael_env
    python run_experiment_tsfm.py --config configs/config_tsfm_sanity_a.yaml
    python run_experiment_tsfm.py --config configs/config_tsfm_sanity_b.yaml
    python run_experiment_tsfm.py --config configs/config_tsfm_all206_minimal.yaml

Modules:
    0  Init (GPU, output dir, config, file logging)
    1  Data loading + train/test split
    2  Target sensor selection (role-aware)
    3  Model loading + covariate API smoke test
    4  Per-target loop:
         4a  Forecast windows (train + test)
         4b  Feature universe construction (all non-target sensors; optional SANITY cap)
         4c  Chronos embedding extraction (train scoring windows; cached)
         4d  Representation scoring (Pearson, Mean_CKA, Lagged_CKA, etc.; train only)
         4e  Remove hooks before forecasting
         4f  TSFM downstream evaluation (frozen Chronos-2)
         4g  GPU cleanup + hooks re-register for next target
    5  Save results CSV + JSON
    6  Statistical summary + Wilcoxon tests
    7  Publication plots

cross_learning = False  (confirmed by diagnose_cross_learning.py)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.real_traffic import (
    load_metr_la,
    make_forecast_windows,
)
from src.data.target_selection import select_target_sensors
from src.evaluation.tsfm_downstream import TSFMForecaster, smoke_test_covariate_api
from src.extraction.chronos_hooks import ChronosEmbeddingExtractor
from src.reporting.saver import save_config
from src.reporting.tsfm_plots import (
    plot_bar_rmse_by_method,
    plot_bar_mae_by_method,
    plot_pct_improvement_vs_target_only,
    plot_pct_improvement_vs_all_features,
    plot_win_count_by_method,
    plot_rolewise_rmse,
)
from src.reporting.statistical_summary import aggregate_results, run_wilcoxon_tests
from src.scoring.tsfm_scorers import (
    lagged_cka,
    precompute_lagged_y,
    mean_pooling_cka,
    pearson_fs_scorer,
    rf_fs_scorer,
    sparse_linear_fs_scorer,
    soft_dtw_score,
)

CROSS_LEARNING: bool = False  # confirmed by diagnose_cross_learning.py

# ---------------------------------------------------------------------------
# Module-level log file (opened in _init, closed at experiment end)
# ---------------------------------------------------------------------------
_log_file = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


# ---------------------------------------------------------------------------
# Caching helpers — multi-layer, one forward pass per batch
# ---------------------------------------------------------------------------

def _multi_layer_cache_key(series: np.ndarray, layers: list[int]) -> str:
    sha = hashlib.sha1(series.tobytes()).hexdigest()
    return f"{sha}_layers{'_'.join(str(l) for l in sorted(layers))}"


def _extract_or_cache_multi(
    extractor: ChronosEmbeddingExtractor,
    windows: np.ndarray,
    layers: list[int],
    cache_dir: Path,
    batch_size: int = 256,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """Return ({layer: [N,P,D]}, cache_hit, n_embed_batches) for all requested layers.

    All layers captured in a single forward pass per batch.  Cache key =
    SHA1(series.tobytes()) + sorted layer list — shared across targets within a run.
    """
    key  = _multi_layer_cache_key(windows, layers)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        data = np.load(path)
        return {l: data[f"layer_{l}"] for l in layers}, True, 0

    N = len(windows)
    all_embs: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    n_batches = 0
    for start in range(0, N, batch_size):
        batch  = windows[start : start + batch_size]
        result = extractor.extract_windows(batch, layers=layers)
        for l in layers:
            if l not in result:
                raise RuntimeError(f"Layer {l} not captured. Check hook registration.")
            all_embs[l].append(result[l])
        n_batches += 1

    embs = {l: np.concatenate(all_embs[l], axis=0) for l in layers}
    np.savez_compressed(path, **{f"layer_{l}": embs[l] for l in layers})
    return embs, False, n_batches


# ---------------------------------------------------------------------------
# Incremental saving helpers
# ---------------------------------------------------------------------------

def _append_row(row: dict, path: Path) -> None:
    """Append one result row to the JSONL file (condition-level persistence)."""
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _make_resume_key(target_sensor_id: str, method: str, top_k, layer, repeat_id) -> tuple:
    """Stable key for resume deduplication.

    layer="N/A" for non-embedding methods.
    repeat_id="N/A" for non-random methods.
    repeat_id=str(r) for random_k (integer as string).
    """
    return (
        str(target_sensor_id),
        str(method),
        str(top_k),
        str(layer) if layer is not None else "N/A",
        str(repeat_id) if repeat_id is not None else "N/A",
    )


# ---------------------------------------------------------------------------
# CUDA-safe forecasting wrapper
# ---------------------------------------------------------------------------

def _safe_evaluate(
    forecaster: TSFMForecaster,
    *,
    target_windows_test: np.ndarray,
    covariate_windows_test: dict,
    y_test: np.ndarray,
    selected_cols: list,
    batch_size: int = 256,
    label: str = "",
) -> dict:
    """Call forecaster.evaluate() with OOM-safe retry at smaller batch_size."""
    import torch
    try:
        return forecaster.evaluate(
            target_windows_test    = target_windows_test,
            covariate_windows_test = covariate_windows_test,
            y_test                 = y_test,
            selected_cols          = selected_cols,
            batch_size             = batch_size,
        )
    except RuntimeError as e:
        err_str = str(e)
        _log(f"  [CUDA ERROR{' ' + label if label else ''}] {err_str[:200]}")
        if "CUDA" in err_str or "out of memory" in err_str.lower():
            _log(f"  [RETRY] Clearing CUDA cache, retrying with batch_size=64 ...")
            torch.cuda.empty_cache()
            gc.collect()
            return forecaster.evaluate(
                target_windows_test    = target_windows_test,
                covariate_windows_test = covariate_windows_test,
                y_test                 = y_test,
                selected_cols          = selected_cols,
                batch_size             = 64,
            )
        raise


# ---------------------------------------------------------------------------
# Module 0: Init
# ---------------------------------------------------------------------------

def _init(config_path: str):
    global _log_file
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This pipeline requires a GPU.")

    cfg       = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path("outputs") / f"EXP_{cfg.experiment.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    _log_file = open(log_dir / "experiment.log", "w")

    _log(f"[Module 0] GPU: {torch.cuda.get_device_name(0)}")

    is_sanity = getattr(cfg.experiment, "debug_mode", False)
    if is_sanity:
        _log("[SANITY] This is a SANITY run only. Not for scientific conclusions.")
    else:
        _log("[Module 0] EXPLORATORY RUN — 80/20 train/test split.")
        _log("           No K or layer choices should be made based on test results.")

    save_config(config_path, out_dir)
    _log(f"[Module 0] Output directory: {out_dir}")
    return cfg, out_dir, timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_metrics(m: dict) -> dict:
    """Serialize per-horizon lists to JSON strings for CSV storage."""
    out = {}
    for k, v in m.items():
        if isinstance(v, list):
            out[k] = json.dumps([round(x, 6) for x in v])
        else:
            out[k] = v
    return out


def _log_cuda_memory(label: str = "") -> None:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            res   = torch.cuda.memory_reserved()  / 1e6
            _log(f"  [CUDA{' ' + label if label else ''}] alloc={alloc:.0f}MB reserved={res:.0f}MB")
    except Exception:
        pass


def _log_pre_forecast(target_id: str, method: str, top_k, layer, n_cov: int,
                      n_test: int, batch_size: int = 256, repeat_id=None) -> None:
    rep_str = f" repeat={repeat_id}" if repeat_id is not None else ""
    _log(f"  [forecast] target={target_id} | method={method} | K={top_k} | layer={layer}"
         f"{rep_str} | n_cov={n_cov} | n_test={n_test} | batch={batch_size}")
    _log_cuda_memory()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:

    # -----------------------------------------------------------------------
    # Module 0 — Init
    # -----------------------------------------------------------------------
    cfg, out_dir, timestamp = _init(config_path)
    cache_dir   = out_dir / "cache"
    results_dir = out_dir / "results"
    plots_dir   = out_dir / "plots"
    jsonl_path  = results_dir / "results_incremental.jsonl"

    t_experiment_start = time.time()
    timing: dict[str, float]    = defaultdict(float)
    timing_4d: dict[str, float] = defaultdict(float)
    cache_hits: int    = 0
    cache_misses: int  = 0
    embed_batches: int = 0
    predict_calls: int = 0

    layers              = cfg.extraction.layers
    top_ks              = cfg.evaluation.top_ks
    test_frac           = cfg.evaluation.test_frac
    pilot_mode          = getattr(cfg.experiment, "pilot_mode", False)
    max_cands           = getattr(cfg.experiment, "max_candidates", None)
    rk_repeats          = getattr(cfg.experiment, "random_k_repeats", 3)
    n_per_role          = getattr(cfg.experiment, "n_per_role", 2)
    active_methods      = list(getattr(cfg.scoring, "active_methods",
                                       ["Pearson", "Mean_CKA", "Lagged_CKA"]))
    max_scoring_windows = getattr(cfg.experiment, "max_scoring_windows", None)
    max_test_windows    = getattr(cfg.evaluation,  "max_test_windows",    None)
    is_sanity           = getattr(cfg.experiment, "debug_mode", False)

    # -----------------------------------------------------------------------
    # Module 1 — Data loading + train/test split
    # -----------------------------------------------------------------------
    _log("\n[Module 1] Loading METR-LA …")
    t0 = time.time()
    data    = load_metr_la(cfg.dataset.zip_path)
    N_total = len(data.df)
    n_train = int(N_total * (1.0 - test_frac))
    gap     = cfg.dataset.context_length
    n_test_start = n_train + gap

    train_df = data.df.iloc[:n_train]
    test_df  = data.df.iloc[n_test_start:]

    # Window overlap statistics
    ctx    = cfg.dataset.context_length
    stride = cfg.dataset.stride
    window_overlap_pct = round((ctx - stride) / ctx * 100, 1)
    _log(f"  N_total={N_total}  N_train={n_train}  gap={gap}  N_test={len(test_df)}")
    _log(f"  context_length={ctx}  stride={stride}  window_overlap={window_overlap_pct}%")
    _log(f"  NOTE: {window_overlap_pct}% overlap — windows are not independent statistical samples.")
    _log(f"  [Module 1 done in {time.time()-t0:.1f}s]")

    # -----------------------------------------------------------------------
    # Module 2 — Target sensor selection
    # -----------------------------------------------------------------------
    _log("\n[Module 2] Selecting target sensors …")
    t0 = time.time()
    role_dict, rankings_df, sensor_role_map = select_target_sensors(
        data, train_df, n_per_role=n_per_role, seed=42
    )
    _log(f"  [Module 2 done in {time.time()-t0:.1f}s]")

    if pilot_mode:
        target_list: list[dict] = role_dict.get("role_A_central", [])[:1]
        _log(f"[Module 2] PILOT/SANITY mode: 1 target (role_A_central only)")
    else:
        target_list = [s for sensors in role_dict.values() for s in sensors]

    _log(f"[Module 2] Targets: {[t['sensor_id'] for t in target_list]}")

    # -----------------------------------------------------------------------
    # Module 3 — Load Chronos-2 + smoke test
    # -----------------------------------------------------------------------
    _log(f"\n[Module 3] Loading {cfg.extraction.model_id} …")
    import torch
    t0 = time.time()
    extractor = ChronosEmbeddingExtractor(
        model_id = cfg.extraction.model_id,
        layers   = layers,
        pooling  = "none",
    )
    extractor.load()
    _log(f"[Module 3] Pipeline type: {type(extractor._pipe).__name__}")
    try:
        param = next(extractor._pipe.model.parameters())
        _log(f"[Module 3] Model parameter device: {param.device}")
        if str(param.device) == "cpu":
            _log("[Module 3] WARNING: model parameters are on CPU — inference will be slow!")
    except Exception as e:
        _log(f"[Module 3] Could not check parameter device: {e}")
    _log(f"  [Module 3 model load done in {time.time()-t0:.1f}s]")

    t0 = time.time()
    smoke_test_covariate_api(
        extractor._pipe,
        context_length = cfg.dataset.context_length,
        horizon        = cfg.dataset.horizon,
        cross_learning = CROSS_LEARNING,
    )
    _log(f"  [Smoke test done in {time.time()-t0:.1f}s]")

    # -----------------------------------------------------------------------
    # Resume: load completed keys from existing JSONL
    # -----------------------------------------------------------------------
    completed_keys: set[tuple]          = set()
    completed_rows_by_key: dict[tuple, dict] = {}
    if jsonl_path.exists():
        _log(f"[Resume] Loading completed rows from {jsonl_path} …")
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if not r.get("is_replicated_baseline"):
                        key = _make_resume_key(
                            r.get("target_sensor_id", ""),
                            r.get("method", ""),
                            r.get("top_k", ""),
                            r.get("layer", "N/A"),
                            r.get("repeat_id", "N/A"),
                        )
                        completed_keys.add(key)
                        completed_rows_by_key[key] = r
                except json.JSONDecodeError:
                    pass
        _log(f"[Resume] Found {len(completed_keys)} completed rows (old naming may differ — verify before trusting)")

    # -----------------------------------------------------------------------
    # Module 4 — Per-target loop
    # -----------------------------------------------------------------------
    _log("\n[Module 4] Per-target evaluation …")

    results_rows: list[dict]   = []
    selected_sensors_map: dict = {}
    target_feature_universe: dict = {}   # {target_sensor_id: list of candidate sensor IDs}

    for t_idx, target_info in enumerate(target_list):
        target_sensor_id = target_info["sensor_id"]
        target_role      = target_info["role"]
        target_df_col    = target_info["df_col"]
        t_target_start   = time.time()
        predict_calls_this_target = 0
        _log(f"\n  [Target {t_idx+1}/{len(target_list)}] {target_sensor_id} (role={target_role})")

        # ------------------------------------------------------------------
        # 4a — Forecast windows (train and test split separately)
        # ------------------------------------------------------------------
        _log(f"  [4a] Making forecast windows …")
        t_phase = time.time()
        train_wins = make_forecast_windows(
            train_df,
            target_sensor  = target_sensor_id,
            context_length = cfg.dataset.context_length,
            horizon        = cfg.dataset.horizon,
            stride         = cfg.dataset.stride,
            max_windows    = cfg.dataset.max_windows,
        )
        N_train_wins_orig = train_wins.X_context.shape[0]

        test_wins = make_forecast_windows(
            test_df,
            target_sensor  = target_sensor_id,
            context_length = cfg.dataset.context_length,
            horizon        = cfg.dataset.horizon,
            stride         = cfg.dataset.stride,
            max_windows    = None,
        )
        N_test_wins_orig = test_wins.X_context.shape[0]
        _log(f"    original_n_train_windows={N_train_wins_orig}  original_n_test_windows={N_test_wins_orig}")
        _log(f"  [4a done in {time.time()-t_phase:.1f}s]")

        Y_train = train_wins.X_context[:, :, target_df_col]
        y_train = train_wins.y_target
        Y_test  = test_wins.X_context[:, :, target_df_col]
        y_test  = test_wins.y_target

        # Subsample train windows for scoring/embedding only
        if max_scoring_windows and max_scoring_windows < len(Y_train):
            _log(f"    scoring_windows: {len(Y_train)} → {max_scoring_windows} (first N, temporal order)")
            Y_train_score = Y_train[:max_scoring_windows]
            y_train_score = y_train[:max_scoring_windows]
        else:
            Y_train_score = Y_train
            y_train_score = y_train
        n_scoring_windows_used = len(Y_train_score)

        # Cap test windows for downstream forecasting
        if max_test_windows and max_test_windows < len(Y_test):
            _log(f"    test_windows: {len(Y_test)} → {max_test_windows} (first N, temporal order)")
            Y_test_eval = Y_test[:max_test_windows]
            y_test_eval = y_test[:max_test_windows]
        else:
            Y_test_eval = Y_test
            y_test_eval = y_test
        n_test_windows_used = len(Y_test_eval)
        N_eval_wins = n_test_windows_used

        _log(f"    n_scoring_windows_used={n_scoring_windows_used}  n_test_windows_used={n_test_windows_used}")
        if N_eval_wins < 50:
            _log(f"  [WARNING] very few test windows ({N_eval_wins}) for {target_sensor_id}")

        # ------------------------------------------------------------------
        # 4b — Feature universe construction
        #      All non-target sensors, with optional SANITY cap.
        # ------------------------------------------------------------------
        _log(f"  [4b] Building feature universe …")
        t_phase = time.time()

        all_nontarget_cols = [col for col in range(data.df.shape[1]) if col != target_df_col]
        n_all_nontarget    = len(all_nontarget_cols)

        if max_cands and n_all_nontarget > max_cands:
            _log(f"    SANITY cap: feature universe = first {max_cands} of {n_all_nontarget} non-target sensors "
                 f"(deterministic, technical cap only)")
            candidate_cols = all_nontarget_cols[:max_cands]
        else:
            candidate_cols = all_nontarget_cols

        feature_universe_size = len(candidate_cols)
        all_features_name     = f"all_features_{feature_universe_size}"
        _log(f"  [4b done in {time.time()-t_phase:.1f}s]  "
             f"feature_universe_size={feature_universe_size}  name={all_features_name}")

        # Save feature universe (sensor IDs) for reproducibility
        target_feature_universe[target_sensor_id] = [
            data.df.columns[col] for col in candidate_cols
        ]

        # Build covariate windows (views into X_context, not copies)
        cov_wins_train: dict[int, np.ndarray] = {
            col: train_wins.X_context[:, :, col]
            for col in candidate_cols
        }
        cov_wins_test: dict[int, np.ndarray] = {
            col: test_wins.X_context[:, :, col]
            for col in candidate_cols
        }
        cov_wins_test_eval: dict[int, np.ndarray] = {
            col: cov_wins_test[col][:n_test_windows_used] for col in candidate_cols
        }
        cov_wins_train_score: dict[int, np.ndarray] = {
            col: cov_wins_train[col][:n_scoring_windows_used] for col in candidate_cols
        }

        # ------------------------------------------------------------------
        # row_base — fields common to every result row for this target
        # ------------------------------------------------------------------
        row_base = {
            "target_sensor_id":       target_sensor_id,
            "target_role":            target_role,
            "downstream_model":       "Chronos-2",
            "frozen":                 True,
            "feature_universe_size":  feature_universe_size,
            "context_length":         cfg.dataset.context_length,
            "stride":                 cfg.dataset.stride,
            "window_overlap_pct":     window_overlap_pct,
            "max_scoring_windows_cfg": max_scoring_windows,
            "n_scoring_windows_used":  n_scoring_windows_used,
            "max_test_windows_cfg":    max_test_windows,
            "n_test_windows_used":     n_test_windows_used,
        }

        # ------------------------------------------------------------------
        # 4c — Embedding extraction (scoring windows; cached globally by sensor)
        # ------------------------------------------------------------------
        _log(f"  [4c] Extracting embeddings (layers={layers}, N_scoring={n_scoring_windows_used}) …")
        t_phase = time.time()

        _log(f"    target ({target_sensor_id}) …")
        Y_embs_all, hit, nb = _extract_or_cache_multi(extractor, Y_train_score, layers, cache_dir)
        if hit: cache_hits += 1
        else:   cache_misses += 1; embed_batches += nb
        Y_embs_by_layer: dict[int, np.ndarray] = Y_embs_all
        _log(f"    Y_emb shape (layer {layers[0]}): {Y_embs_by_layer[layers[0]].shape}")

        X_embs_by_layer: dict[int, dict[int, np.ndarray]] = {l: {} for l in layers}
        for ci, col in enumerate(candidate_cols):
            if ci % 20 == 0:
                _log(f"    cov {ci}/{len(candidate_cols)} …")
            cov_win = cov_wins_train_score[col]
            embs_all, hit, nb = _extract_or_cache_multi(extractor, cov_win, layers, cache_dir)
            if hit: cache_hits += 1
            else:   cache_misses += 1; embed_batches += nb
            for l in layers:
                X_embs_by_layer[l][col] = embs_all[l]

        t_4c = time.time() - t_phase
        timing["4c_total"] += t_4c
        _log(f"  [4c done in {t_4c:.1f}s]  "
             f"{len(candidate_cols)} candidates × {len(layers)} layers  "
             f"cache hits={cache_hits}  misses={cache_misses}")

        # ------------------------------------------------------------------
        # 4d — Representation scoring (train windows only; no whitening)
        # ------------------------------------------------------------------
        _log(f"  [4d] Scoring …")
        t_phase = time.time()
        scoring_results: list[dict] = []

        _use_mean_cka  = "Mean_CKA" in active_methods or "Mean_Pooling" in active_methods
        _use_rf        = "RandomForest" in active_methods
        _mean_cka_name = "Mean_CKA" if "Mean_CKA" in active_methods else "Mean_Pooling"

        for layer in layers:
            Y_emb  = Y_embs_by_layer[layer]
            X_embs = X_embs_by_layer[layer]

            if _use_mean_cka:
                t0 = time.time()
                feat_scores_mp: dict[int, float] = {}
                for col in candidate_cols:
                    feat_scores_mp[col] = mean_pooling_cka(X_embs[col], Y_emb)
                scoring_results.append({
                    "method": _mean_cka_name, "layer": layer,
                    "feat_scores": feat_scores_mp,
                })
                timing_4d[_mean_cka_name] += time.time() - t0

            if "Lagged_CKA" in active_methods:
                # Precompute Y Gram matrices once per layer — reused for all candidates
                y_lag_cache = precompute_lagged_y(Y_emb, cfg.scoring.max_lag)
                t0 = time.time()
                feat_scores_lc: dict[int, float] = {}
                for col in candidate_cols:
                    feat_scores_lc[col] = lagged_cka(
                        X_embs[col], Y_emb, cfg.scoring.max_lag,
                        precomputed_y=y_lag_cache,
                    )
                scoring_results.append({
                    "method": "Lagged_CKA", "layer": layer,
                    "feat_scores": feat_scores_lc,
                })
                timing_4d["Lagged_CKA"] += time.time() - t0

            if "Soft_DTW" in active_methods:
                t0 = time.time()
                feat_scores_dtw: dict[int, float] = {}
                for col in candidate_cols:
                    feat_scores_dtw[col] = soft_dtw_score(
                        X_embs[col], Y_emb, float(cfg.scoring.dtw_gamma)
                    )
                scoring_results.append({
                    "method": f"Soft_DTW_g{cfg.scoring.dtw_gamma}", "layer": layer,
                    "feat_scores": feat_scores_dtw,
                })
                timing_4d["Soft_DTW"] += time.time() - t0

        # Supervised baselines (no layer; train windows only)
        if "Pearson" in active_methods:
            t0 = time.time()
            pearson_scores = pearson_fs_scorer(
                cov_wins_train_score, Y_train_score, candidate_cols, test_frac=0.0
            )
            scoring_results.append({
                "method": "Pearson", "layer": "N/A", "feat_scores": pearson_scores,
            })
            timing_4d["Pearson"] += time.time() - t0

        if "SparseLinear_L1" in active_methods:
            t0 = time.time()
            sl1_scores = sparse_linear_fs_scorer(
                cov_wins_train_score, y_train_score, candidate_cols, test_frac=0.0
            )
            scoring_results.append({
                "method": "SparseLinear_L1", "layer": "N/A", "feat_scores": sl1_scores,
            })
            timing_4d["SparseLinear_L1"] += time.time() - t0

        if _use_rf:
            t0 = time.time()
            rf_scores = rf_fs_scorer(
                cov_wins_train_score, y_train_score, candidate_cols, test_frac=0.0
            )
            scoring_results.append({
                "method": "RandomForest", "layer": "N/A", "feat_scores": rf_scores,
            })
            elapsed_rf = time.time() - t0
            timing_4d["RandomForest"] += elapsed_rf
            _log(f"    [4d] RandomForest done in {elapsed_rf:.1f}s  ({len(candidate_cols)} cands)")

        t_4d = time.time() - t_phase
        timing["4d_total"] += t_4d
        _log(f"  [4d done in {t_4d:.1f}s]  "
             + "  ".join(f"{m}:{v:.1f}s" for m, v in timing_4d.items()))

        # ------------------------------------------------------------------
        # 4e — Remove hooks before forecasting
        # ------------------------------------------------------------------
        extractor.remove_hooks()
        _log(f"  [4e] Hooks removed.")

        # ------------------------------------------------------------------
        # 4f — TSFM downstream evaluation
        # ------------------------------------------------------------------
        t_4f_start = time.time()
        forecaster = TSFMForecaster(
            pipeline          = extractor._pipe,
            prediction_length = cfg.dataset.horizon,
            cross_learning    = CROSS_LEARNING,
        )

        # ---- target_only (no covariates) ----
        to_key = _make_resume_key(target_sensor_id, "target_only", 0, "N/A", "N/A")
        if to_key in completed_keys:
            _log(f"  [RESUME] target_only — loaded from checkpoint")
            row_to = completed_rows_by_key[to_key]
            results_rows.append(row_to)
            metrics_to = {k: row_to[k] for k in ["RMSE", "MAE", "MAPE", "mae_per_horizon", "rmse_per_horizon"] if k in row_to}
        else:
            _log(f"  [target_only] …")
            _log_pre_forecast(target_sensor_id, "target_only", 0, "N/A", 0, N_eval_wins)
            t0 = time.time()
            metrics_to = _safe_evaluate(
                forecaster,
                target_windows_test    = Y_test_eval,
                covariate_windows_test = {},
                y_test                 = y_test_eval,
                selected_cols          = [],
                label                  = "target_only",
            )
            predict_calls += 1; predict_calls_this_target += 1
            t_to = time.time() - t0
            _log(f"    target_only: RMSE={metrics_to['RMSE']:.4f}  MAE={metrics_to['MAE']:.4f}  ({t_to:.1f}s)")
            row_to = {
                **row_base,
                "method": "target_only", "top_k": 0, "layer": "N/A",
                "mode": "target_only", "n_covariates": 0,
                "candidate_scope": "N/A", "baseline_scope": "target_only",
                "selected_sensors": "[]", "repeat_id": None,
                "is_replicated_baseline": False,
                "runtime_seconds": round(t_to, 2),
                **_flatten_metrics(metrics_to),
            }
            results_rows.append(row_to)
            _append_row(row_to, jsonl_path)
            completed_keys.add(to_key)
            completed_rows_by_key[to_key] = row_to

        # Replicated target_only rows for K-axis plots (copy only, not recomputed)
        for k in top_ks:
            results_rows.append({
                **row_base,
                "method": "target_only", "top_k": k, "layer": "N/A",
                "mode": "target_only", "n_covariates": 0,
                "candidate_scope": "N/A", "baseline_scope": "target_only",
                "selected_sensors": "[]", "repeat_id": None,
                "is_replicated_baseline": True, "runtime_seconds": 0.0,
                **_flatten_metrics(metrics_to),
            })

        # ---- all_features_N (all non-target sensors in the universe) ----
        af_key = _make_resume_key(target_sensor_id, all_features_name, feature_universe_size, "N/A", "N/A")
        if af_key in completed_keys:
            _log(f"  [RESUME] {all_features_name} — loaded from checkpoint")
            row_af = completed_rows_by_key[af_key]
            results_rows.append(row_af)
            metrics_af = {k: row_af[k] for k in ["RMSE", "MAE", "MAPE", "mae_per_horizon", "rmse_per_horizon"] if k in row_af}
        else:
            _log(f"  [{all_features_name} ({feature_universe_size} sensors)] …")
            _log_pre_forecast(target_sensor_id, all_features_name, feature_universe_size,
                              "N/A", feature_universe_size, N_eval_wins)
            t0 = time.time()
            metrics_af = _safe_evaluate(
                forecaster,
                target_windows_test    = Y_test_eval,
                covariate_windows_test = cov_wins_test_eval,
                y_test                 = y_test_eval,
                selected_cols          = candidate_cols,
                label                  = all_features_name,
            )
            predict_calls += 1; predict_calls_this_target += 1
            t_af = time.time() - t0
            _log(f"    {all_features_name}: RMSE={metrics_af['RMSE']:.4f}  MAE={metrics_af['MAE']:.4f}  ({t_af:.1f}s)")
            row_af = {
                **row_base,
                "method": all_features_name, "top_k": feature_universe_size, "layer": "N/A",
                "mode": "all_features", "n_covariates": feature_universe_size,
                "candidate_scope": "N/A", "baseline_scope": "all_non_target_sensors",
                "selected_sensors": json.dumps(candidate_cols), "repeat_id": None,
                "is_replicated_baseline": False, "runtime_seconds": round(t_af, 2),
                **_flatten_metrics(metrics_af),
            }
            results_rows.append(row_af)
            _append_row(row_af, jsonl_path)
            completed_keys.add(af_key)
            completed_rows_by_key[af_key] = row_af

        # Replicated all_features rows for K-axis plots
        for k in top_ks:
            results_rows.append({
                **row_base,
                "method": all_features_name, "top_k": k, "layer": "N/A",
                "mode": "all_features", "n_covariates": feature_universe_size,
                "candidate_scope": "N/A", "baseline_scope": "all_non_target_sensors",
                "selected_sensors": json.dumps(candidate_cols), "repeat_id": None,
                "is_replicated_baseline": True, "runtime_seconds": 0.0,
                **_flatten_metrics(metrics_af),
            })

        # ---- random_k ----
        _log(f"  [random_k × {rk_repeats} repeats] …")
        for top_k in top_ks:
            for r in range(rk_repeats):
                rk_key = _make_resume_key(target_sensor_id, "random_k", top_k, "N/A", r)
                if rk_key in completed_keys:
                    results_rows.append(completed_rows_by_key[rk_key])
                    continue
                rng_r = np.random.default_rng(42 + r)
                k_size = min(top_k, len(candidate_cols))
                rand_cols = rng_r.choice(candidate_cols, size=k_size, replace=False).tolist()
                _log_pre_forecast(target_sensor_id, "random_k", top_k, "N/A", k_size, N_eval_wins, repeat_id=r)
                m = _safe_evaluate(
                    forecaster,
                    target_windows_test    = Y_test_eval,
                    covariate_windows_test = cov_wins_test_eval,
                    y_test                 = y_test_eval,
                    selected_cols          = rand_cols,
                    label                  = f"random_k K={top_k} r={r}",
                )
                predict_calls += 1; predict_calls_this_target += 1
                row_rk = {
                    **row_base,
                    "method": "random_k", "top_k": top_k, "layer": "N/A",
                    "mode": "random_k", "n_covariates": k_size,
                    "candidate_scope": f"all_{feature_universe_size}",
                    "baseline_scope": "N/A",
                    "selected_sensors": json.dumps(rand_cols),
                    "repeat_id": r,
                    "is_replicated_baseline": False, "runtime_seconds": 0.0,
                    **_flatten_metrics(m),
                }
                results_rows.append(row_rk)
                _append_row(row_rk, jsonl_path)
                completed_keys.add(rk_key)
                completed_rows_by_key[rk_key] = row_rk

        # ---- Scored methods ----
        _log(f"  [scored methods × {len(top_ks)} K values] …")
        for sr in scoring_results:
            method    = sr["method"]
            layer     = sr["layer"]
            scores    = sr["feat_scores"]

            norm_method = re.sub(r"^Soft_DTW_g.*", "Soft_DTW", method)
            if layer not in ("N/A", None):
                method_key = f"{norm_method}_L{layer}"
            else:
                method_key = norm_method

            for top_k in top_ks:
                top_k_cols = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
                sm_key = _make_resume_key(target_sensor_id, method_key, top_k, layer, "N/A")
                if sm_key in completed_keys:
                    results_rows.append(completed_rows_by_key[sm_key])
                    selected_sensors_map \
                        .setdefault(target_sensor_id, {}) \
                        .setdefault(str(top_k), {})[method_key] = top_k_cols
                    continue

                _log_pre_forecast(target_sensor_id, method_key, top_k, layer, top_k, N_eval_wins)
                t0 = time.time()
                metrics_m = _safe_evaluate(
                    forecaster,
                    target_windows_test    = Y_test_eval,
                    covariate_windows_test = cov_wins_test_eval,
                    y_test                 = y_test_eval,
                    selected_cols          = top_k_cols,
                    label                  = f"{method_key} K={top_k}",
                )
                predict_calls += 1; predict_calls_this_target += 1
                row_m = {
                    **row_base,
                    "method": method_key, "top_k": top_k, "layer": layer,
                    "mode": "scored", "n_covariates": top_k,
                    "candidate_scope": f"all_{feature_universe_size}",
                    "baseline_scope": "N/A",
                    "selected_sensors": json.dumps(top_k_cols),
                    "repeat_id": None,
                    "is_replicated_baseline": False,
                    "runtime_seconds": round(time.time() - t0, 2),
                    **_flatten_metrics(metrics_m),
                }
                results_rows.append(row_m)
                _append_row(row_m, jsonl_path)
                completed_keys.add(sm_key)
                completed_rows_by_key[sm_key] = row_m

                selected_sensors_map \
                    .setdefault(target_sensor_id, {}) \
                    .setdefault(str(top_k), {})[method_key] = top_k_cols

        t_4f = time.time() - t_4f_start
        timing["4f_total"] += t_4f
        t_elapsed = time.time() - t_target_start
        _log(f"  Target {target_sensor_id} done in {t_elapsed:.1f}s  "
             f"(4f forecasting: {t_4f:.1f}s  predict_calls_this_target={predict_calls_this_target})")

        # ------------------------------------------------------------------
        # 4g — GPU cleanup + hooks re-register for next target
        # ------------------------------------------------------------------
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if t_idx < len(target_list) - 1:
            blocks = extractor._find_blocks()
            extractor._register_hooks(blocks, layers)
            _log(f"  [4g] GPU cleanup done. Hooks re-registered for next target.")
        else:
            _log(f"  [4g] GPU cleanup done. Last target — hooks not re-registered.")

    # -----------------------------------------------------------------------
    # Module 5 — Save results
    # -----------------------------------------------------------------------
    _log("\n[Module 5] Saving results …")
    results_df = pd.DataFrame(results_rows)
    csv_path   = results_dir / "tsfm_downstream_results.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"  saved {csv_path}  ({len(results_df)} rows)")

    sensors_path = results_dir / "selected_sensors_by_method.json"
    with open(sensors_path, "w") as f:
        json.dump(selected_sensors_map, f, indent=2)
    _log(f"  saved {sensors_path}")

    # Save feature universe per target
    fu_path = results_dir / "candidate_sensors_by_target.json"
    with open(fu_path, "w") as f:
        json.dump(target_feature_universe, f, indent=2)
    _log(f"  saved {fu_path}  (feature universe per target)")

    # Copy role/ranking files to results_dir
    for fname in ["selected_target_sensors_by_role.json", "target_role_rankings_full.csv"]:
        src = Path("outputs") / fname
        if src.exists():
            shutil.copy2(src, results_dir / fname)
            _log(f"  copied {fname} → results_dir")

    # -----------------------------------------------------------------------
    # Module 6 — Statistical summary
    # -----------------------------------------------------------------------
    _log("\n[Module 6] Statistical summary …")
    try:
        agg_df  = aggregate_results(results_df, out_dir=results_dir)
        test_df_stat = run_wilcoxon_tests(results_df, out_dir=results_dir)
        _log(f"  Aggregate summary: {len(agg_df)} rows")
        if not test_df_stat.empty:
            _log(f"  Wilcoxon tests: {len(test_df_stat)} comparisons")
    except Exception as e:
        _log(f"  [warn] Statistical summary failed: {e}")

    # -----------------------------------------------------------------------
    # Module 7 — Plots
    # -----------------------------------------------------------------------
    _log("\n[Module 7] Generating plots …")
    try:
        plot_bar_rmse_by_method(results_df, plots_dir)
        plot_bar_mae_by_method(results_df, plots_dir)
        plot_pct_improvement_vs_target_only(results_df, plots_dir)
        plot_pct_improvement_vs_all_features(results_df, plots_dir)
        plot_win_count_by_method(results_df, plots_dir)
        plot_rolewise_rmse(results_df, plots_dir)
        _log(f"  Plots saved to {plots_dir}")
    except Exception as e:
        _log(f"  [warn] Plot generation failed: {e}")

    # -----------------------------------------------------------------------
    # Module 8 — experiment_description.md + runtime_report.json
    # -----------------------------------------------------------------------
    t_total = time.time() - t_experiment_start
    timing["total"] = t_total

    is_sanity = getattr(cfg.experiment, "debug_mode", False)
    desc_path = out_dir / "experiment_description.md"
    active_methods_str = ", ".join(active_methods)
    n_targets_actual   = len(target_list)
    target_ids_str     = ", ".join(t["sensor_id"] for t in target_list)
    unique_methods     = sorted(results_df["method"].unique()) if not results_df.empty else []

    with open(desc_path, "w") as f:
        f.write(f"# Experiment: {cfg.experiment.name}\n\n")
        if is_sanity:
            f.write("**This is a SANITY run only. It is not used for scientific conclusions.**\n\n")
        else:
            f.write("**EXPLORATORY RUN — 80/20 train/test split. "
                    "No K or layer choices should be made based on test results.**\n\n")
        f.write(f"Run ID: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Research Question\n")
        f.write("Can a model-aware FS method select a small Top-K subset from all available "
                "non-target sensors and improve or compete with using all features in a frozen TSFM?\n\n")
        f.write("## Dataset\n")
        f.write(f"METR-LA — 207 sensors, 34,272 timestamps at 5-min resolution.\n")
        f.write(f"Train/test split: 80/20 exploratory.\n")
        f.write(f"Context length: {cfg.dataset.context_length}. Horizon: {cfg.dataset.horizon}. Stride: {cfg.dataset.stride}.\n")
        f.write(f"Window overlap: {window_overlap_pct}%. Windows are not independent statistical samples.\n")
        f.write(f"Scoring windows: {n_scoring_windows_used} (capped from original train windows). "
                "First N in temporal order.\n\n")
        f.write("## Targets\n")
        f.write(f"{n_targets_actual} role-selected sensors: {target_ids_str}.\n\n")
        f.write("## Feature Universe\n")
        f.write(f"All non-target sensors: {feature_universe_size} sensors")
        if max_cands:
            f.write(f" (SANITY cap: first {max_cands} of {data.df.shape[1]-1})")
        f.write(".\n\n")
        f.write("## Layer\n")
        f.write(f"{layers}. Layer 8 is a pragmatic default; not optimized in this run. "
                "Layer ablation [6, 8, 10] is a future step.\n\n")
        f.write("## Scoring Methods\n")
        f.write(f"{active_methods_str}\n\n")
        f.write("## Method Names in Results\n")
        f.write(", ".join(unique_methods) + "\n\n")
        f.write("## K Values\n")
        f.write(f"{top_ks}\n\n")
        f.write("## Runtime\n")
        f.write(f"Total: {t_total:.0f}s ({t_total/60:.1f} min).\n\n")
        f.write("## Limitations\n")
        f.write("- Exploratory 80/20 split only. Do not use for confirmatory claims.\n")
        f.write("- n=10 targets per Wilcoxon test — low statistical power.\n")
        f.write("- No validation split; K and layer choices must not be made from test results.\n")
        f.write("- Window overlap ~91.7%: not independent samples.\n")
    _log(f"  saved {desc_path}")

    report_path = out_dir / "runtime_report.json"
    runtime_report = {
        "total_runtime_s":          round(t_total, 1),
        "timing_per_stage":         {k: round(v, 1) for k, v in timing.items()},
        "timing_4d_per_method":     {k: round(v, 1) for k, v in timing_4d.items()},
        "cache_hits":               cache_hits,
        "cache_misses":             cache_misses,
        "embed_batches_total":      embed_batches,
        "predict_calls_total":      predict_calls,
        "n_targets":                n_targets_actual,
        "layers":                   layers,
        "feature_universe_size":    feature_universe_size,
        "max_scoring_windows":      max_scoring_windows,
        "n_scoring_windows_used":   n_scoring_windows_used,
        "max_test_windows":         max_test_windows,
        "n_test_windows_used":      n_test_windows_used,
        "active_methods":           active_methods,
        "window_overlap_pct":       window_overlap_pct,
        "is_sanity":                is_sanity,
    }
    with open(report_path, "w") as f:
        json.dump(runtime_report, f, indent=2)
    _log(f"  saved {report_path}")

    _log(f"\nDone. Experiment directory: {out_dir}")
    _log(f"  CSV:    {csv_path}")
    _log(f"  Plots:  {plots_dir}/")
    _log(f"  Total runtime: {t_total:.0f}s ({t_total/60:.1f} min)")
    _log(f"  Cache: {cache_hits} hits / {cache_misses} misses  |  "
         f"embed batches: {embed_batches}  |  predict calls: {predict_calls}")

    # Summary table (non-replicated, non-repeat rows)
    display_cols = ["target_sensor_id", "method", "top_k", "layer", "n_covariates", "RMSE", "MAE"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    if "is_replicated_baseline" in results_df.columns:
        non_rep = results_df[results_df["is_replicated_baseline"] != True]
    else:
        non_rep = results_df
    non_rep = non_rep[non_rep["repeat_id"].isna()]
    _log("\n" + non_rep[display_cols].sort_values(["target_sensor_id", "method", "top_k"])
                                      .to_string(index=False))

    if _log_file is not None:
        _log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TSFM Feature Selection — frozen Chronos-2 downstream evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    main(args.config)
