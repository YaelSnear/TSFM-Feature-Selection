"""Diagnostic for corrected window-centered Lagged CKA variant.

Part A — Anonymous mathematical diagnostic using cached .npz files.
Part B — Target-aware score-only diagnostic with real sensor mapping +
          Lagged_Pearson and Lagged_MI baselines.

Usage:
    conda run --no-capture-output -n yael_env \
        python scripts/diagnose_lagged_cka_fixed.py \
            --cache_dir outputs/EXP_tsfm_full_run_all206_20260530_172932/cache
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.scoring.tsfm_scorers import (
    lagged_cka,
    lagged_cka_window_centered,
    precompute_lagged_y_wc,
)

TARGET_SENSOR_ID = "717469"
TARGET_ROLE      = "role_A_central"
TARGET_DF_COL    = 154       # derived from all_features_206 gap analysis
CONTEXT_LENGTH   = 144
STRIDE           = 12
MAX_WINDOWS      = 9999
MAX_SCORING_WIN  = 1000   # must match original FULL RUN to hit cache
MAX_LAG          = 24
LAYER            = 8
TEST_FRAC        = 0.2
LAYERS_KEY       = [LAYER]

_stop_triggered = False


def _stop(msg: str) -> None:
    global _stop_triggered
    print(f"\n  STOP CONDITION: {msg}", flush=True)
    _stop_triggered = True


def _cache_key(windows_arr: np.ndarray, layers: list[int]) -> str:
    sha = hashlib.sha1(windows_arr.tobytes()).hexdigest()
    return f"{sha}_layers{'_'.join(str(l) for l in sorted(layers))}"


def _overlap_at_k(set_a: list, set_b: list, k: int) -> int:
    return len(set(set_a[:k]) & set(set_b[:k]))


# ---------------------------------------------------------------------------
# Part A — Anonymous mathematical diagnostic
# ---------------------------------------------------------------------------

def part_a(cache_dir: Path) -> bool:
    print("\n" + "="*70)
    print("PART A — Anonymous Mathematical Diagnostic")
    print("="*70)
    print(f"Cache dir: {cache_dir}")

    files = sorted(cache_dir.glob("*.npz"))
    if len(files) < 5:
        print(f"  ERROR: fewer than 5 cache files found ({len(files)})")
        return False

    print(f"  Total cache files: {len(files)}")
    Y0 = np.load(files[0])["layer_8"]
    N, P, D = Y0.shape
    print(f"  Embedding shape: N={N}, P={P}, D={D}")

    candidates = files[1:]
    n_cands = len(candidates)

    # --- Old lagged_cka (patch-centered, broken) ---
    print(f"\n  Computing old lagged_cka for {n_cands} candidates ...", flush=True)
    t0 = time.time()
    old_scores = np.array([
        lagged_cka(np.load(f)["layer_8"], Y0, max_lag=MAX_LAG)
        for f in candidates
    ])
    print(f"  Old lagged_cka: min={old_scores.min():.6f}  max={old_scores.max():.6f}"
          f"  mean={old_scores.mean():.6f}  std={old_scores.std():.8f}  "
          f"({time.time()-t0:.1f}s)")
    if not np.all(np.abs(old_scores - 1.0) < 1e-4):
        print("  [warn] Some old scores != 1.0 — degeneracy pattern may differ from expected")

    # --- New lagged_cka_window_centered ---
    print(f"\n  Computing lagged_cka_window_centered for {n_cands} candidates ...", flush=True)
    t0 = time.time()
    y_cache_wc = precompute_lagged_y_wc(Y0, MAX_LAG)
    new_scores = []
    new_lags   = []
    for f in candidates:
        X = np.load(f)["layer_8"]
        s, l = lagged_cka_window_centered(X, Y0, MAX_LAG, precomputed_y=y_cache_wc)
        new_scores.append(s)
        new_lags.append(l)
    new_scores = np.array(new_scores)
    new_lags   = np.array(new_lags)
    elapsed = time.time() - t0

    print(f"\n  Corrected scores (axis=0): {elapsed:.1f}s")
    print(f"    min={new_scores.min():.6f}  max={new_scores.max():.6f}"
          f"  mean={new_scores.mean():.6f}  std={new_scores.std():.6f}")
    n_unique = len(set(round(s, 6) for s in new_scores))
    print(f"    Unique scores (6dp): {n_unique} of {n_cands}")

    # NaN/inf check
    if np.any(~np.isfinite(new_scores)):
        _stop("NaN or inf in corrected scores")
        return False

    # Degeneracy checks
    if new_scores.std() <= 1e-6:
        _stop(f"std={new_scores.std():.2e} ≤ 1e-6 — still degenerate")
        return False
    if new_scores.max() - new_scores.min() <= 1e-4:
        _stop(f"range={new_scores.max()-new_scores.min():.2e} ≤ 1e-4 — still degenerate")
        return False
    if np.all(np.abs(new_scores - 1.0) < 1e-4):
        _stop("All corrected scores ≈ 1.0 — still degenerate")
        return False

    # Best-lag distribution
    lag_dist = dict(sorted(Counter(new_lags.tolist()).items()))
    print(f"\n  best_lag distribution (file 0 as target):")
    for lag_val, cnt in lag_dist.items():
        print(f"    lag={lag_val:+3d}: {cnt} candidates")

    # Top-20 / bottom-20 by file index
    order = np.argsort(-new_scores)
    print(f"\n  Top-20 candidates (file index, score):")
    for idx in order[:20]:
        print(f"    file={idx+1:3d}  score={new_scores[idx]:.6f}  lag={new_lags[idx]:+3d}"
              f"  [{candidates[idx].name[:20]}...]")
    print(f"\n  Bottom-20 candidates (file index, score):")
    for idx in order[-20:][::-1]:
        print(f"    file={idx+1:3d}  score={new_scores[idx]:.6f}  lag={new_lags[idx]:+3d}")

    top5_file0 = set(order[:5].tolist())
    if top5_file0 == {0, 1, 2, 3, 4}:
        _stop("Top-5 indices = {0,1,2,3,4} — trivial ordering")
        return False

    # Ranking stability: repeat with file[50] as target
    if len(files) > 51:
        Y50 = np.load(files[50])["layer_8"]
        y_cache_wc50 = precompute_lagged_y_wc(Y50, MAX_LAG)
        scores50 = []
        for f in candidates:
            if f == files[50]:
                scores50.append(-np.inf)
                continue
            X = np.load(f)["layer_8"]
            s, _ = lagged_cka_window_centered(X, Y50, MAX_LAG, precomputed_y=y_cache_wc50)
            scores50.append(s)
        scores50 = np.array(scores50)
        top5_file50 = set(np.argsort(-scores50)[:5].tolist())
        print(f"\n  Ranking stability check:")
        print(f"    Top-5 with file[0] as target:  {sorted(top5_file0)}")
        print(f"    Top-5 with file[50] as target: {sorted(top5_file50)}")
        if top5_file0 == top5_file50:
            _stop("Top-5 identical for two different targets — ranking not target-dependent")
            return False
        print(f"    Overlap: {len(top5_file0 & top5_file50)}/5  [PASS: rankings differ]")

    print("\n  Part A: PASS — scores are non-degenerate and target-dependent.")
    return True


# ---------------------------------------------------------------------------
# Part B — Target-aware score-only diagnostic
# ---------------------------------------------------------------------------

def _lagged_pearson_raw(X_wins: np.ndarray, Y_wins: np.ndarray, max_lag: int
                        ) -> tuple[float, int]:
    """Max abs Pearson correlation over lags. X_wins/Y_wins: [N, context_length]."""
    N, T = Y_wins.shape
    actual_max_lag = min(max_lag, T - 2)
    best_score = 0.0
    best_lag   = 0
    for k in range(-actual_max_lag, actual_max_lag + 1):
        if k > 0:
            xv = X_wins[:, k:]
            yv = Y_wins[:, :T - k]
        elif k < 0:
            xv = X_wins[:, :T + k]
            yv = Y_wins[:, -k:]
        else:
            xv, yv = X_wins, Y_wins
        xv_c = xv - xv.mean(axis=1, keepdims=True)
        yv_c = yv - yv.mean(axis=1, keepdims=True)
        x_n  = np.sqrt((xv_c ** 2).sum(axis=1))
        y_n  = np.sqrt((yv_c ** 2).sum(axis=1))
        denom = x_n * y_n
        corrs = np.where(denom > 1e-10, (xv_c * yv_c).sum(axis=1) / denom, 0.0)
        score = float(np.mean(np.abs(corrs)))
        if score > best_score:
            best_score = score
            best_lag   = k
    return best_score, best_lag


def part_b(cache_dir: Path, out_dir: Path) -> bool:
    print("\n" + "="*70)
    print("PART B — Target-Aware Score-Only Diagnostic")
    print("="*70)
    print(f"Target: {TARGET_SENSOR_ID} (df_col={TARGET_DF_COL}, {TARGET_ROLE})")
    print(f"max_scoring_windows={MAX_SCORING_WIN}, max_lag={MAX_LAG}, layer={LAYER}")

    # --- Load METR-LA data ---
    from src.data.real_traffic import load_metr_la, make_forecast_windows
    print("\n  Loading METR-LA ...", flush=True)
    data = load_metr_la("data/raw/METR-LA.zip")
    df   = data.df
    N_total = len(df)
    n_train = int(N_total * (1.0 - TEST_FRAC))
    train_df = df.iloc[:n_train]

    # Verify target df_col
    sensor_cols = list(df.columns)
    if TARGET_SENSOR_ID not in sensor_cols:
        print(f"  ERROR: {TARGET_SENSOR_ID} not found in DataFrame columns")
        return False
    actual_df_col = sensor_cols.index(TARGET_SENSOR_ID)
    if actual_df_col != TARGET_DF_COL:
        print(f"  [info] df_col reconfirmed: {actual_df_col} (expected {TARGET_DF_COL})")
    target_df_col = actual_df_col

    # --- Make train forecast windows ---
    print("  Making train forecast windows ...", flush=True)
    train_wins = make_forecast_windows(
        train_df,
        target_sensor   = TARGET_SENSOR_ID,
        context_length  = CONTEXT_LENGTH,
        horizon         = 12,
        stride          = STRIDE,
        max_windows     = MAX_WINDOWS,
    )
    N_wins = train_wins.X_context.shape[0]
    N_use  = min(MAX_SCORING_WIN, N_wins)
    print(f"  Train windows: {N_wins}, using first {N_use}")

    Y_wins_raw = train_wins.X_context[:N_use, :, target_df_col]  # [N_use, 144]

    # Candidate df_cols
    candidate_cols = [c for c in range(df.shape[1]) if c != target_df_col]
    print(f"  Candidates: {len(candidate_cols)} sensors")

    # --- Load target embedding from cache ---
    target_series = train_wins.X_context[:N_use, :, target_df_col]  # [N_use, 144]
    target_key    = _cache_key(target_series, LAYERS_KEY)
    target_npz    = cache_dir / f"{target_key}.npz"

    if not target_npz.exists():
        print(f"  ERROR: Target cache file not found: {target_npz}")
        print(f"         The original FULL RUN used N={MAX_SCORING_WIN} windows.")
        print(f"         If N_use was changed, restore MAX_SCORING_WIN to 1000.")
        return False

    Y_emb = np.load(target_npz)[f"layer_{LAYER}"]
    print(f"  Target embedding shape: {Y_emb.shape}")

    # --- Score all candidates ---
    print(f"\n  Scoring {len(candidate_cols)} candidates ...", flush=True)
    y_cache_wc = precompute_lagged_y_wc(Y_emb, MAX_LAG)

    scores_lc   = {}
    lags_lc     = {}
    scores_lp   = {}
    lags_lp     = {}
    missing_cache = 0

    t0 = time.time()
    for i, col in enumerate(candidate_cols):
        if i % 50 == 0:
            print(f"    {i}/{len(candidate_cols)} ...", flush=True)

        # Raw windows for Lagged_Pearson
        X_wins_raw = train_wins.X_context[:N_use, :, col]  # [N_use, 144]

        # Lagged_CKA_fixed via cache
        cand_key = _cache_key(X_wins_raw, LAYERS_KEY)
        cand_npz = cache_dir / f"{cand_key}.npz"
        if not cand_npz.exists():
            missing_cache += 1
            continue
        X_emb = np.load(cand_npz)[f"layer_{LAYER}"]
        s_lc, l_lc = lagged_cka_window_centered(X_emb, Y_emb, MAX_LAG, precomputed_y=y_cache_wc)
        scores_lc[col] = s_lc
        lags_lc[col]   = l_lc

        # Lagged_Pearson (raw windows)
        s_lp, l_lp = _lagged_pearson_raw(X_wins_raw, Y_wins_raw, MAX_LAG)
        scores_lp[col] = s_lp
        lags_lp[col]   = l_lp

    t_score = time.time() - t0
    if missing_cache > 0:
        print(f"  [warn] {missing_cache} candidates had no cache file and were skipped")

    scored_cols = sorted(scores_lc.keys())
    n_scored    = len(scored_cols)
    print(f"\n  Scored {n_scored} candidates in {t_score:.1f}s")

    if n_scored < 10:
        print("  ERROR: too few candidates scored (cache likely mismatched)")
        return False

    # --- Lagged_MI ---
    scores_mi = {}
    lags_mi   = {}
    mi_available = False
    try:
        from sklearn.feature_selection import mutual_info_regression
        print(f"\n  Computing Lagged_MI for {n_scored} candidates ...", flush=True)
        t_mi = time.time()
        Y_flat = Y_wins_raw.mean(axis=1)  # [N_use] — mean over context
        for i, col in enumerate(scored_cols):
            X_raw = train_wins.X_context[:N_use, :, col]
            best_mi  = -np.inf
            best_lag_mi = 0
            for k in range(-min(MAX_LAG, 5), min(MAX_LAG, 5) + 1):  # ±5 lag limit for MI speed
                T = X_raw.shape[1]
                if k > 0:
                    xv = X_raw[:, k:].mean(axis=1)
                elif k < 0:
                    xv = X_raw[:, :T+k].mean(axis=1)
                else:
                    xv = X_raw.mean(axis=1)
                mi_val = float(mutual_info_regression(
                    xv.reshape(-1, 1), Y_flat, random_state=0
                )[0])
                if mi_val > best_mi:
                    best_mi = mi_val
                    best_lag_mi = k
            scores_mi[col] = best_mi
            lags_mi[col]   = best_lag_mi
            if i % 50 == 0:
                elapsed_mi = time.time() - t_mi
                rate = (i + 1) / (elapsed_mi + 1e-6)
                est_remaining = (n_scored - i - 1) / max(rate, 1e-6)
                if est_remaining > 300:
                    print(f"  [Lagged_MI] Estimated {est_remaining:.0f}s remaining — SKIPPING (too slow)")
                    scores_mi = {}
                    lags_mi   = {}
                    break
        else:
            mi_available = len(scores_mi) == n_scored
            if mi_available:
                print(f"  Lagged_MI done in {time.time()-t_mi:.1f}s (lag range ±5)")
    except ImportError:
        print("  [Lagged_MI] sklearn not available — skipping")

    # --- Degeneracy checks on Lagged_CKA_fixed ---
    lc_arr = np.array([scores_lc[c] for c in scored_cols])
    if np.any(~np.isfinite(lc_arr)):
        _stop("NaN or inf in Lagged_CKA_fixed scores")
        return False
    if lc_arr.std() <= 1e-6:
        _stop(f"std={lc_arr.std():.2e} ≤ 1e-6 — still degenerate")
        return False
    if np.all(np.abs(lc_arr - 1.0) < 1e-4):
        _stop("All corrected scores ≈ 1.0 — still degenerate")
        return False

    # --- Build ranked lists ---
    lc_order = sorted(scored_cols, key=lambda c: scores_lc[c], reverse=True)
    lp_order = sorted(scored_cols, key=lambda c: scores_lp[c], reverse=True)
    lc_ranks = {c: i+1 for i, c in enumerate(lc_order)}
    lp_ranks = {c: i+1 for i, c in enumerate(lp_order)}

    selected_k5  = lc_order[:5]
    selected_k10 = lc_order[:10]
    selected_k20 = lc_order[:20]

    # Check trivial ordering
    for k, sel in [(5, selected_k5), (10, selected_k10), (20, selected_k20)]:
        if sel == list(range(k)):
            _stop(f"Top-{k} selected = {sel} — trivial df_col ordering")
            return False

    # --- Print score distribution ---
    print(f"\n  Lagged_CKA_fixed score distribution:")
    print(f"    min={lc_arr.min():.6f}  max={lc_arr.max():.6f}"
          f"  mean={lc_arr.mean():.6f}  std={lc_arr.std():.6f}")
    print(f"    Unique scores (6dp): {len(set(round(s,6) for s in lc_arr))}")

    sensor_id_map = dict(zip(range(df.shape[1]), df.columns))

    print(f"\n  Top-20 candidates (Lagged_CKA_fixed):")
    print(f"  {'df_col':>7}  {'sensor_id':>10}  {'score':>10}  {'rank':>5}  {'best_lag':>9}")
    for c in lc_order[:20]:
        print(f"  {c:>7}  {sensor_id_map.get(c,'?'):>10}  {scores_lc[c]:>10.6f}"
              f"  {lc_ranks[c]:>5}  {lags_lc[c]:>+9d}")

    print(f"\n  Bottom-20 candidates (Lagged_CKA_fixed):")
    print(f"  {'df_col':>7}  {'sensor_id':>10}  {'score':>10}  {'rank':>5}  {'best_lag':>9}")
    for c in lc_order[-20:][::-1]:
        print(f"  {c:>7}  {sensor_id_map.get(c,'?'):>10}  {scores_lc[c]:>10.6f}"
              f"  {lc_ranks[c]:>5}  {lags_lc[c]:>+9d}")

    print(f"\n  Selected sensors:")
    print(f"    K=5:  df_cols={selected_k5}")
    print(f"          sensor_ids={[sensor_id_map.get(c,'?') for c in selected_k5]}")
    print(f"    K=10: df_cols={selected_k10}")
    print(f"    K=20: df_cols={selected_k20}")

    lag_dist = dict(sorted(Counter(lags_lc[c] for c in scored_cols).items()))
    print(f"\n  best_lag distribution:")
    for lag_val, cnt in lag_dist.items():
        bar = "#" * min(cnt, 40)
        print(f"    lag={lag_val:+3d}: {cnt:3d}  {bar}")

    # --- Lagged_Pearson comparison ---
    lp_arr = np.array([scores_lp[c] for c in scored_cols])
    from scipy.stats import spearmanr as _spearman, pearsonr as _pearson

    rho_lp, p_lp = _spearman(lc_arr, lp_arr)
    olap5  = _overlap_at_k(lc_order, lp_order, 5)
    olap10 = _overlap_at_k(lc_order, lp_order, 10)
    olap20 = _overlap_at_k(lc_order, lp_order, 20)

    print(f"\n  Lagged_Pearson comparison:")
    print(f"    Spearman rho (LC_fixed vs LP): {rho_lp:.4f}  p={p_lp:.4f}")
    print(f"    Overlap@5:  {olap5}/5")
    print(f"    Overlap@10: {olap10}/10")
    print(f"    Overlap@20: {olap20}/20")

    if mi_available:
        mi_arr = np.array([scores_mi[c] for c in scored_cols])
        mi_order = sorted(scored_cols, key=lambda c: scores_mi[c], reverse=True)
        rho_mi, p_mi = _spearman(lc_arr, mi_arr)
        olap5_mi  = _overlap_at_k(lc_order, mi_order, 5)
        olap10_mi = _overlap_at_k(lc_order, mi_order, 10)
        olap20_mi = _overlap_at_k(lc_order, mi_order, 20)
        print(f"\n  Lagged_MI comparison:")
        print(f"    Spearman rho (LC_fixed vs MI): {rho_mi:.4f}  p={p_mi:.4f}")
        print(f"    Overlap@5:  {olap5_mi}/5")
        print(f"    Overlap@10: {olap10_mi}/10")
        print(f"    Overlap@20: {olap20_mi}/20")
    else:
        print("\n  Lagged_MI: skipped (runtime too long or not available)")

    # --- Correlation with raw variance and embedding norm ---
    raw_var  = np.array([train_wins.X_context[:N_use, :, c].var() for c in scored_cols])
    emb_norm = np.array([
        np.load(cache_dir / f"{_cache_key(train_wins.X_context[:N_use, :, c], LAYERS_KEY)}.npz")
        [f"layer_{LAYER}"].mean(axis=0).mean()  # mean embedding value as proxy for norm
        if (cache_dir / f"{_cache_key(train_wins.X_context[:N_use, :, c], LAYERS_KEY)}.npz").exists()
        else np.nan
        for c in scored_cols
    ])
    valid_norm = np.isfinite(emb_norm)

    r_var, _ = _pearson(lc_arr, raw_var)
    print(f"\n  Correlations with potential confounders:")
    print(f"    Pearson(LC_fixed, raw_variance):   {r_var:.4f}")
    if valid_norm.sum() > 10:
        r_norm, _ = _pearson(lc_arr[valid_norm], emb_norm[valid_norm])
        print(f"    Pearson(LC_fixed, emb_mean_val):   {r_norm:.4f}")
    else:
        print(f"    Pearson(LC_fixed, emb_norm): skipped (embedding files unavailable)")

    # --- Save score table ---
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for rank_i, c in enumerate(lc_order):
        score_rows.append({
            "target_sensor_id":     TARGET_SENSOR_ID,
            "target_role":          TARGET_ROLE,
            "candidate_df_col":     c,
            "candidate_sensor_id":  sensor_id_map.get(c, "?"),
            "score":                scores_lc[c],
            "rank":                 rank_i + 1,
            "best_lag":             lags_lc[c],
            "n_scoring_windows_used": N_use,
            "max_lag":              MAX_LAG,
            "layer":                LAYER,
        })
    score_df = pd.DataFrame(score_rows)
    score_path = out_dir / "lagged_cka_fixed_scores_diagnostic.csv"
    score_df.to_csv(score_path, index=False)
    print(f"\n  Saved: {score_path}")

    # --- Save comparison table ---
    cmp_rows = []
    mi_cols_present = mi_available
    for c in scored_cols:
        row = {
            "candidate_df_col":          c,
            "candidate_sensor_id":       sensor_id_map.get(c, "?"),
            "Lagged_CKA_fixed_score":    scores_lc[c],
            "Lagged_CKA_fixed_rank":     lc_ranks[c],
            "Lagged_CKA_fixed_best_lag": lags_lc[c],
            "Lagged_Pearson_score":      scores_lp[c],
            "Lagged_Pearson_rank":       lp_ranks[c],
            "Lagged_Pearson_best_lag":   lags_lp[c],
            "Lagged_MI_score":           scores_mi.get(c, float("nan")),
            "Lagged_MI_rank":            (sorted(scored_cols, key=lambda x: scores_mi.get(x, -1), reverse=True).index(c) + 1) if mi_available else float("nan"),
            "Lagged_MI_best_lag":        lags_mi.get(c, float("nan")),
        }
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_path = out_dir / "lagged_sanity_score_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"  Saved: {cmp_path}")

    if _stop_triggered:
        return False

    print("\n  Part B: PASS — target-aware diagnostic complete.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="outputs/EXP_tsfm_full_run_all206_20260530_172932/cache",
        help="Path to original FULL RUN cache directory"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path("outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun")

    if not cache_dir.exists():
        print(f"ERROR: cache_dir not found: {cache_dir}")
        sys.exit(1)

    ok_a = part_a(cache_dir)
    if not ok_a or _stop_triggered:
        print("\n[DIAGNOSTIC] Part A FAILED. Stopping.")
        sys.exit(1)

    ok_b = part_b(cache_dir, out_dir)
    if not ok_b or _stop_triggered:
        print("\n[DIAGNOSTIC] Part B FAILED. Stopping.")
        sys.exit(1)

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE — Both parts passed.")
    print("Proceed to: conda run -n yael_env python scripts/sanity_lagged_cka_forecast.py")
    print("="*70)
    sys.exit(0)


if __name__ == "__main__":
    main()
