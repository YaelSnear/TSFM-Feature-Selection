"""TSFM latent-space feature scorers.

Three scoring methods, each accepting Raw or Whitened [N, P, D] embeddings:

1. mean_pooling_cka  — pool patches → [N, D], then standard CKA.
2. lagged_cka        — slide WITHIN window patches for lags [-max_lag, max_lag],
                       CKA per window averaged across N windows, take max lag score.
3. soft_dtw_score    — per-window soft-DTW on [P, D] patch sequences, averaged.

Two supervised SOTA baselines (fit on train portion only to prevent leakage):

4. lasso_fs_scorer   — LassoCV(cv=3) on flattened raw windows; aggregate |coef| per sensor.
5. rf_fs_scorer      — RandomForestRegressor on flattened raw windows; aggregate importances.

All functions return a single float score per sensor (higher = more relevant),
except lasso_fs_scorer and rf_fs_scorer which return {col_idx: score} dicts.
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
# Method 1: Mean-Pooling CKA (baseline)
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
# Method 2: Lagged CKA (within-window patch sliding)
# ---------------------------------------------------------------------------

def lagged_cka(X_emb: np.ndarray, Y_emb: np.ndarray, max_lag: int) -> float:
    """Sliding-window CKA over the patch dimension within each window.

    For lag k > 0: hx = X_emb[w, k:, :],  hy = Y_emb[w, :P-k, :]
    For lag k = 0: hx = X_emb[w],          hy = Y_emb[w]
    For lag k < 0: hx = X_emb[w, :P+k, :], hy = Y_emb[w, -k:, :]

    For each k, compute CKA per window and average across all N windows.
    Return the maximum score over all tested lags in [-actual_max_lag, actual_max_lag].

    Args:
        X_emb   : [N, P, D]
        Y_emb   : [N, P, D]
        max_lag : maximum patch shift requested; capped to P-1 so slices are never empty

    Returns:
        Best CKA score over all tested lags.
    """
    N, P, D = X_emb.shape

    # Cap max_lag to P-1: any larger shift produces zero-row slices on at least
    # one side, making CKA undefined. A shift of P-1 already leaves only 1 patch,
    # which the inner guard (< 2) will skip, so the effective range shrinks
    # gracefully when P is small.
    actual_max_lag = min(max_lag, P - 1)

    best_score = float("-inf")

    for k in range(-actual_max_lag, actual_max_lag + 1):
        window_scores = []
        for w in range(N):
            if k > 0:
                hx = X_emb[w, k:, :]        # [P-k, D]
                hy = Y_emb[w, :P - k, :]    # [P-k, D]
            elif k < 0:
                hx = X_emb[w, :P + k, :]    # [P+k, D]  (k is negative)
                hy = Y_emb[w, -k:, :]       # [P+k, D]
            else:
                hx = X_emb[w]               # [P, D]
                hy = Y_emb[w]               # [P, D]

            # Guard: require at least 2 rows in BOTH slices before calling CKA.
            # The capped loop prevents asymmetric shapes, but this check is a
            # defensive fallback in case P is very small (P <= 2).
            if hx.shape[0] < 2 or hy.shape[0] < 2:
                continue
            window_scores.append(_cka_core(hx, hy))

        if window_scores:
            lag_score = float(np.mean(window_scores))
            if lag_score > best_score:
                best_score = lag_score

    return best_score if best_score > float("-inf") else 0.0


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
# Method 4: Lasso supervised feature selector (train-only, no leakage)
# ---------------------------------------------------------------------------

def lasso_fs_scorer(
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    gap: int = 144,
) -> dict[int, float]:
    """Score all candidate sensors with LassoCV fit on the train split only.

    Constructs X_train by horizontally concatenating the 144-step context
    windows for every candidate sensor (column order = all_sensor_df_cols).
    The target is the mean over the 12 forecast horizons (scalar per window).

    After fitting, each sensor's score is the sum of absolute values of its
    144 Lasso coefficients, giving a single relevance measure per sensor.

    Args:
        raw_windows_X      : {col_idx: [N, 144]} raw context windows per sensor
        y_target           : [N, 12] forecast targets
        all_sensor_df_cols : ordered list of candidate df column indices
        test_frac          : fraction of windows reserved for testing
        gap                : context-overlap gap excluded between train and test

    Returns:
        {col_idx: aggregated_lasso_score}
    """
    from sklearn.linear_model import LassoCV

    N = next(iter(raw_windows_X.values())).shape[0]
    n_train = int(N * (1.0 - test_frac))

    X_train = np.hstack([raw_windows_X[col][:n_train] for col in all_sensor_df_cols])
    y_train = y_target[:n_train].mean(axis=1)   # [n_train] — mean over 12 horizons

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LassoCV(cv=3, max_iter=5000)
    model.fit(X_train_s, y_train)

    coef = model.coef_   # [144 * n_sensors]
    scores: dict[int, float] = {}
    for i, col in enumerate(all_sensor_df_cols):
        scores[col] = float(np.sum(np.abs(coef[i * 144 : (i + 1) * 144])))
    return scores


# ---------------------------------------------------------------------------
# Method 5: Random Forest supervised feature selector (train-only, no leakage)
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

        # Compute correlation only where both series have nonzero variance;
        # use np.divide with out/where to avoid the spurious division-by-zero
        # warning that np.where produces by evaluating both branches first.
        corr    = np.zeros_like(numer)
        valid   = denom > 1e-10
        np.divide(numer, denom, out=corr, where=valid)         # [n_train]
        scores[col] = float(np.mean(np.abs(corr)))
    return scores


def rf_fs_scorer(
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    gap: int = 144,
) -> dict[int, float]:
    """Score all candidate sensors with RandomForestRegressor fit on train split only.

    Constructs X_train identically to lasso_fs_scorer. After fitting, each
    sensor's score is the sum of MDI feature importances for its 144 features.

    Args:
        raw_windows_X      : {col_idx: [N, 144]} raw context windows per sensor
        y_target           : [N, 12] forecast targets
        all_sensor_df_cols : ordered list of candidate df column indices
        test_frac          : fraction of windows reserved for testing
        gap                : context-overlap gap excluded between train and test

    Returns:
        {col_idx: aggregated_rf_importance_score}
    """
    from sklearn.ensemble import RandomForestRegressor

    N = next(iter(raw_windows_X.values())).shape[0]
    n_train = int(N * (1.0 - test_frac))

    X_train = np.hstack([raw_windows_X[col][:n_train] for col in all_sensor_df_cols])
    y_train = y_target[:n_train].mean(axis=1)   # [n_train] — mean over 12 horizons

    model = RandomForestRegressor(
        n_estimators=100, max_depth=5, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_   # [144 * n_sensors]
    scores: dict[int, float] = {}
    for i, col in enumerate(all_sensor_df_cols):
        scores[col] = float(np.sum(importances[i * 144 : (i + 1) * 144]))
    return scores
