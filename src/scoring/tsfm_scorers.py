"""TSFM latent-space feature scorers.

Methods operating on [N, P, D] patch embeddings:

1. mean_pooling_cka  — pool patches → [N, D], standard CKA across N windows.
2. lagged_cka        — slide WITHIN window patches for lags [-max_lag, max_lag];
                       CKA per window averaged across N windows, take best lag score.
3. soft_dtw_score    — per-window soft-DTW on [P, D] patch sequences, averaged.

Helpers:
   precompute_lagged_y — precompute Y Gram matrices for all lags once per target/layer,
                         so lagged_cka can skip recomputing them for every candidate.

Statistical feature-selection baselines (fit on train split only; no test leakage):

4. pearson_fs_scorer       — mean absolute Pearson correlation per sensor.
5. rf_fs_scorer            — RandomForest MDI importance per sensor.
6. sparse_linear_fs_scorer — Fixed-alpha L1 (Lasso) linear feature ranking.
                             Not CV-tuned; designed for fast feature ranking.

All functions return a single float score per sensor (higher = more relevant),
except the dict-returning scorers which return {col_idx: score}.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# CKA core
# ---------------------------------------------------------------------------

def _cka_core(Hx: np.ndarray, Hy: np.ndarray) -> float:
    """Linear CKA between two 2-D representation matrices [n, d].

    Both matrices are centred internally. Returns a score in [0, 1].
    Returns 0.0 when the denominator is numerically zero.
    """
    Hxc = Hx - Hx.mean(axis=0)
    Hyc = Hy - Hy.mean(axis=0)
    num   = np.linalg.norm(Hxc.T @ Hyc, "fro") ** 2
    denom = (np.linalg.norm(Hxc.T @ Hxc, "fro") *
             np.linalg.norm(Hyc.T @ Hyc, "fro"))
    return float(num / denom) if denom > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Method 1: Mean-Pooling CKA
# ---------------------------------------------------------------------------

def mean_pooling_cka(X_emb: np.ndarray, Y_emb: np.ndarray) -> float:
    """Pool patch dimension then compute CKA on the window representations.

    Args:
        X_emb : [N, P, D] candidate feature embeddings
        Y_emb : [N, P, D] target embeddings

    Returns:
        CKA score in [0, 1].
    """
    Hx = X_emb.mean(axis=1)   # [N, D]
    Hy = Y_emb.mean(axis=1)   # [N, D]
    return _cka_core(Hx, Hy)


# ---------------------------------------------------------------------------
# Method 2: Lagged CKA — precompute helper + scorer
# ---------------------------------------------------------------------------

def precompute_lagged_y(Y_emb: np.ndarray, max_lag: int) -> dict:
    """Precompute Y Gram matrices for all lags to avoid redundant computation.

    When lagged_cka is called for every candidate with the same Y_emb,
    the Y-side Gram matrices (K_y, norm_y) are identical across candidates.
    Calling this once per target/layer and passing the result to lagged_cka
    avoids recomputing them C times (once per candidate).

    The mathematical definition of Lagged_CKA is unchanged.

    Args:
        Y_emb   : [N, P, D] target embeddings
        max_lag : maximum patch shift (capped to P-1 internally)

    Returns:
        {k: (K_y [N, P', P'], norm_y [N])} for all valid lag values k.
    """
    N, P, D = Y_emb.shape
    actual_max_lag = min(max_lag, P - 1)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(-actual_max_lag, actual_max_lag + 1):
        if k > 0:
            Y_lag = Y_emb[:, :P - k, :]
        elif k < 0:
            Y_lag = Y_emb[:, -k:, :]
        else:
            Y_lag = Y_emb
        if Y_lag.shape[1] < 2:
            continue
        Yc  = Y_lag - Y_lag.mean(axis=1, keepdims=True)
        K_y = np.matmul(Yc, Yc.swapaxes(-1, -2))              # [N, P', P']
        out[k] = (K_y, np.sqrt((K_y ** 2).sum(axis=(1, 2))))  # norm_y [N]
    return out


def lagged_cka(
    X_emb: np.ndarray,
    Y_emb: np.ndarray,
    max_lag: int,
    precomputed_y: dict | None = None,
) -> float:
    """Sliding-window CKA over the patch dimension within each window.

    Vectorized using the Gram matrix identity: for P-by-D matrices Hxc, Hyc
    (centered over the patch dimension):
        ||Hxc.T @ Hyc||_F^2 = trace(K_x @ K_y)   where K_x = Hxc @ Hxc.T [P', P']
        ||Hxc.T @ Hxc||_F   = ||K_x||_F

    With P=11, D=768 (Chronos-2 default for context_length=144), K_x is 11×11.
    Batching all N windows as [N, P', P'] reduces N individual scalar CKA calls
    to a single batched matmul, eliminating the Python-level window loop.

    For lag k > 0: X_lag = X_emb[:, k:, :],  Y_lag = Y_emb[:, :P-k, :]
    For lag k = 0: X_lag = X_emb,             Y_lag = Y_emb
    For lag k < 0: X_lag = X_emb[:, :P+k, :], Y_lag = Y_emb[:, -k:, :]

    Args:
        X_emb         : [N, P, D]
        Y_emb         : [N, P, D]
        max_lag       : maximum patch shift; capped to P-1 so slices are never empty
        precomputed_y : optional dict from precompute_lagged_y(Y_emb, max_lag).
                        When provided, K_y and norm_y are reused instead of recomputed.

    Returns:
        Best CKA score over all tested lags in [-actual_max_lag, actual_max_lag].
    """
    N, P, D = X_emb.shape
    actual_max_lag = min(max_lag, P - 1)
    best_score = float("-inf")

    for k in range(-actual_max_lag, actual_max_lag + 1):
        if k > 0:
            X_lag = X_emb[:, k:, :]
            Y_lag = Y_emb[:, :P - k, :]
        elif k < 0:
            X_lag = X_emb[:, :P + k, :]
            Y_lag = Y_emb[:, -k:, :]
        else:
            X_lag, Y_lag = X_emb, Y_emb

        P_prime = X_lag.shape[1]
        if P_prime < 2:
            continue

        # Center over patch axis (axis=1) for each window simultaneously
        Xc = X_lag - X_lag.mean(axis=1, keepdims=True)   # [N, P', D]

        # Gram matrices via batched matmul: [N, P', D] × [N, D, P'] → [N, P', P']
        K_x = np.matmul(Xc, Xc.swapaxes(-1, -2))

        if precomputed_y is not None and k in precomputed_y:
            K_y, norm_y = precomputed_y[k]
        else:
            Yc  = Y_lag - Y_lag.mean(axis=1, keepdims=True)
            K_y = np.matmul(Yc, Yc.swapaxes(-1, -2))
            norm_y = np.sqrt((K_y ** 2).sum(axis=(1, 2)))

        # trace(K_x @ K_y) per window via element-wise product + sum
        num    = (K_x * K_y).sum(axis=(1, 2))             # [N]
        norm_x = np.sqrt((K_x ** 2).sum(axis=(1, 2)))     # [N]  = ||K_x||_F

        denom = norm_x * norm_y
        cka_batch = np.where(denom > 1e-10, num / np.maximum(denom, 1e-10), 0.0)
        lag_score = float(cka_batch.mean())
        if lag_score > best_score:
            best_score = lag_score

    return best_score if best_score > float("-inf") else 0.0


# ---------------------------------------------------------------------------
# Method 2b: Window-centered Lagged CKA variant
# ---------------------------------------------------------------------------
# Background: the original lagged_cka centers over the patch axis (axis=1),
# which makes all sensors' patch Gram matrices nearly proportional on
# correlated sensor networks (e.g. METR-LA traffic), causing all scores = 1.0.
#
# Fix: center over the window axis (axis=0) instead.
#
# Object centered:  X_lag[n, p, d]  for window n, patch p, embedding dim d.
#
# Old (broken):   Xc[n,p,d] = X_lag[n,p,d] − mean_{p}(X_lag[n,:,d])
#   → removes mean patch within each window → rank-1 structure shared by all
#   sensors → K_x ≈ c·K_y → CKA = 1.0 for every candidate.
#
# Fixed:          Xc[n,p,d] = X_lag[n,p,d] − mean_{n}(X_lag[:,p,d])
#   → removes temporal mean for each (patch, dim) position → K_x[n] captures
#   window n's deviation from the population mean → different sensors deviate
#   differently → discriminative CKA scores.
#
# What Gram matrices are compared:
#   K_x[n] = Xc[n] @ Xc[n].T  — P'×P' patch covariance of window n's
#   temporal deviation.  CKA per window n = trace(K_x[n]@K_y[n]) /
#   (‖K_x[n]‖_F · ‖K_y[n]‖_F).  Score = mean over N windows.
#
# Lag alignment: unchanged.  For k>0: X_lag=X_emb[:,k:,:], Y_lag=Y_emb[:,:P-k,:].
# Centering is applied to the lag-aligned views, not to the full embeddings.
#
# This is a variant, NOT standard CKA (which operates on [N,D] matrices).
# Name in reports: "window-centered Lagged CKA variant".
#
# precomputed_y cache MUST come from precompute_lagged_y_wc, not from
# precompute_lagged_y (which uses axis=1 centering).


def precompute_lagged_y_wc(Y_emb: np.ndarray, max_lag: int) -> dict:
    """Y-side Gram precomputation for lagged_cka_window_centered.

    Centers over the window axis (axis=0):
        Yc[n, p, d] = Y_lag[n, p, d] − mean_{n}(Y_lag[:, p, d])

    Do NOT mix with precompute_lagged_y, which uses axis=1 centering.

    Args:
        Y_emb   : [N, P, D] target embeddings
        max_lag : maximum patch shift (capped to P-1)

    Returns:
        {k: (K_y [N, P', P'], norm_y [N])} for all valid lags k.
    """
    N, P, D = Y_emb.shape
    actual_max_lag = min(max_lag, P - 1)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(-actual_max_lag, actual_max_lag + 1):
        if k > 0:
            Y_lag = Y_emb[:, :P - k, :]
        elif k < 0:
            Y_lag = Y_emb[:, -k:, :]
        else:
            Y_lag = Y_emb
        if Y_lag.shape[1] < 2:
            continue
        Yc  = Y_lag - Y_lag.mean(axis=0, keepdims=True)           # center over N
        K_y = np.matmul(Yc, Yc.swapaxes(-1, -2))                  # [N, P', P']
        out[k] = (K_y, np.sqrt((K_y ** 2).sum(axis=(1, 2))))      # norm_y [N]
    return out


def lagged_cka_window_centered(
    X_emb: np.ndarray,
    Y_emb: np.ndarray,
    max_lag: int,
    precomputed_y: dict | None = None,
) -> tuple[float, int]:
    """Window-centered Lagged CKA variant.

    Fixes the patch-centering degeneracy in lagged_cka (where axis=1 centering
    makes all scores = 1.0 on correlated sensor networks) by centering over the
    window axis (axis=0) instead of the patch axis (axis=1).

    All other logic — lag alignment, Gram matrix formula, max_lag clipping,
    and per-window CKA averaging — is unchanged from lagged_cka.

    Args:
        X_emb         : [N, P, D] candidate embeddings
        Y_emb         : [N, P, D] target embeddings
        max_lag       : maximum patch shift (capped to P-1)
        precomputed_y : optional dict from precompute_lagged_y_wc(Y_emb, max_lag).
                        Must NOT be a dict from precompute_lagged_y (axis=1).

    Returns:
        (best_score, best_lag) — best window-averaged CKA over all valid lags
        and the lag index at which it was achieved.
    """
    N, P, D = X_emb.shape
    actual_max_lag = min(max_lag, P - 1)
    best_score = float("-inf")
    best_lag   = 0

    for k in range(-actual_max_lag, actual_max_lag + 1):
        if k > 0:
            X_lag = X_emb[:, k:, :]
            Y_lag = Y_emb[:, :P - k, :]
        elif k < 0:
            X_lag = X_emb[:, :P + k, :]
            Y_lag = Y_emb[:, -k:, :]
        else:
            X_lag, Y_lag = X_emb, Y_emb

        P_prime = X_lag.shape[1]
        if P_prime < 2:
            continue

        Xc  = X_lag - X_lag.mean(axis=0, keepdims=True)           # center over N
        K_x = np.matmul(Xc, Xc.swapaxes(-1, -2))                  # [N, P', P']

        if precomputed_y is not None and k in precomputed_y:
            K_y, norm_y = precomputed_y[k]
        else:
            Yc  = Y_lag - Y_lag.mean(axis=0, keepdims=True)        # center over N
            K_y = np.matmul(Yc, Yc.swapaxes(-1, -2))
            norm_y = np.sqrt((K_y ** 2).sum(axis=(1, 2)))

        num    = (K_x * K_y).sum(axis=(1, 2))                     # [N]
        norm_x = np.sqrt((K_x ** 2).sum(axis=(1, 2)))             # [N]
        denom  = norm_x * norm_y
        cka_batch  = np.where(denom > 1e-10, num / np.maximum(denom, 1e-10), 0.0)
        lag_score  = float(cka_batch.mean())
        if lag_score > best_score:
            best_score = lag_score
            best_lag   = k

    score = best_score if best_score > float("-inf") else 0.0
    return score, best_lag


# ---------------------------------------------------------------------------
# Method 3: Soft-DTW
# ---------------------------------------------------------------------------

def soft_dtw_score(X_emb: np.ndarray, Y_emb: np.ndarray, gamma: float) -> float:
    """Average negated soft-DTW distance over per-window patch sequences.

    Soft-DTW is a distance (lower = more similar).  We negate it so that
    higher score = more relevant, consistent with CKA conventions.

    Args:
        X_emb : [N, P, D] — each window's patch sequence is passed to soft_dtw
        Y_emb : [N, P, D]
        gamma : soft-DTW smoothing parameter

    Returns:
        Mean of -soft_dtw(X_emb[w], Y_emb[w], gamma) over all N windows.
    """
    from tslearn.metrics import soft_dtw

    scores = [
        -soft_dtw(X_emb[w], Y_emb[w], gamma=gamma)
        for w in range(X_emb.shape[0])
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Method 4: Pearson correlation feature selector (train-only, no leakage)
# ---------------------------------------------------------------------------

def pearson_fs_scorer(
    raw_windows_X: dict[int, np.ndarray],
    Y_series: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    gap: int = 144,
) -> dict[int, float]:
    """Score all candidate sensors by absolute Pearson correlation with the target
    on the train split only (no leakage).

    For each training window, computes the absolute Pearson correlation between
    the 144-step context of the candidate sensor and the target sensor.  The
    per-sensor score is the mean of these per-window values across all N_train
    windows, giving a single time-averaged correlation measure.

    Args:
        raw_windows_X      : {col_idx: [N, 144]} raw context windows per sensor
        Y_series           : [N, 144] raw context windows for the target
        all_sensor_df_cols : ordered list of candidate df column indices
        test_frac          : fraction of windows reserved for testing
        gap                : unused here but kept for API consistency

    Returns:
        {col_idx: mean_absolute_pearson_correlation}
    """
    N = Y_series.shape[0]
    n_train = int(N * (1.0 - test_frac))

    Y_train = Y_series[:n_train]                         # [n_train, 144]
    Y_c = Y_train - Y_train.mean(axis=1, keepdims=True)  # centre each row
    Y_norm = np.sqrt((Y_c ** 2).sum(axis=1))             # [n_train]

    scores: dict[int, float] = {}
    for col in all_sensor_df_cols:
        X_train = raw_windows_X[col][:n_train]                # [n_train, 144]
        X_c     = X_train - X_train.mean(axis=1, keepdims=True)
        X_norm  = np.sqrt((X_c ** 2).sum(axis=1))             # [n_train]

        numer   = (X_c * Y_c).sum(axis=1)      # [n_train]
        denom   = X_norm * Y_norm              # [n_train]

        corr    = np.zeros_like(numer)
        valid   = denom > 1e-10
        np.divide(numer, denom, out=corr, where=valid)         # [n_train]
        scores[col] = float(np.mean(np.abs(corr)))
    return scores


# ---------------------------------------------------------------------------
# Method 5: Random Forest supervised feature selector (train-only, no leakage)
# ---------------------------------------------------------------------------

def rf_fs_scorer(
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    gap: int = 144,
) -> dict[int, float]:
    """Score all candidate sensors with RandomForestRegressor fit on train split only.

    Constructs X_train by horizontally concatenating the 144-step context
    windows for every candidate sensor. After fitting, each sensor's score
    is the sum of MDI feature importances for its 144 features.

    Args:
        raw_windows_X      : {col_idx: [N, 144]} raw context windows per sensor
        y_target           : [N, 12] forecast targets
        all_sensor_df_cols : ordered list of candidate df column indices
        test_frac          : fraction of windows reserved for testing
        gap                : unused here but kept for API consistency

    Returns:
        {col_idx: aggregated_rf_importance_score}
    """
    from sklearn.ensemble import RandomForestRegressor

    N = next(iter(raw_windows_X.values())).shape[0]
    n_train = int(N * (1.0 - test_frac))

    X_train = np.hstack([raw_windows_X[col][:n_train] for col in all_sensor_df_cols])
    y_train = y_target[:n_train].mean(axis=1)   # [n_train] — mean over 12 horizons

    model = RandomForestRegressor(
        n_estimators=50, max_depth=5, n_jobs=-1, random_state=42
    )  # 50 trees for FULL RUN; bump to 100 only with explicit approval
    model.fit(X_train, y_train)

    importances = model.feature_importances_   # [144 * n_sensors]
    scores: dict[int, float] = {}
    for i, col in enumerate(all_sensor_df_cols):
        scores[col] = float(np.sum(importances[i * 144 : (i + 1) * 144]))
    return scores


# ---------------------------------------------------------------------------
# Method 6: SparseLinear_L1 — fixed-alpha L1 linear feature ranking
# ---------------------------------------------------------------------------

def sparse_linear_fs_scorer(
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    gap: int = 144,
) -> dict[int, float]:
    """Fixed-alpha L1 linear feature ranking on StandardScaler-normalized data.

    Not CV-tuned; designed for fast feature ranking, not prediction accuracy.
    alpha=0.01 is a documented default for StandardScaler-normalized data.

    Constructs X_train by horizontally concatenating the context windows for
    every candidate sensor. Each sensor's score is the sum of absolute Lasso
    coefficients for its 144 features after fitting.

    Args:
        raw_windows_X      : {col_idx: [N, 144]} raw context windows per sensor
        y_target           : [N, 12] forecast targets
        all_sensor_df_cols : ordered list of candidate df column indices
        test_frac          : fraction of windows reserved for testing
        gap                : unused here but kept for API consistency

    Returns:
        {col_idx: aggregated_sparse_linear_score}
    """
    import time
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    N = next(iter(raw_windows_X.values())).shape[0]
    n_train = int(N * (1.0 - test_frac))

    X_train = np.hstack([raw_windows_X[col][:n_train] for col in all_sensor_df_cols])
    y_train = y_target[:n_train].mean(axis=1)   # [n_train] — mean over 12 horizons

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = Lasso(
        alpha=0.01,
        max_iter=5000,
        tol=1e-3,
        selection="random",
        random_state=42,
    )

    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X_train_s, y_train)
        n_conv_warns = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))

    converged = n_conv_warns == 0
    print(
        f"  [SparseLinear_L1] alpha=0.01 max_iter=5000 tol=1e-3 selection=random "
        f"StandardScaler=yes  n_features={X_train_s.shape[1]}  "
        f"n_iter={model.n_iter_}  converged={'yes' if converged else f'NO ({n_conv_warns} warnings)'}  "
        f"runtime={time.time()-t0:.1f}s",
        flush=True,
    )
    if not converged:
        print(
            f"  [SparseLinear_L1] WARNING: {n_conv_warns} ConvergenceWarning(s). "
            f"Scores may be unreliable. Consider excluding from FULL RUN.",
            flush=True,
        )

    coef = model.coef_   # [144 * n_sensors]
    scores: dict[int, float] = {}
    for i, col in enumerate(all_sensor_df_cols):
        scores[col] = float(np.sum(np.abs(coef[i * 144 : (i + 1) * 144])))
    return scores
