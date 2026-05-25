"""TSFM latent-space feature scorers.

Three scoring methods, each accepting Raw or Whitened [N, P, D] embeddings:

1. mean_pooling_cka  — pool patches → [N, D], then standard CKA.
2. lagged_cka        — slide WITHIN window patches for lags [-max_lag, max_lag],
                       CKA per window averaged across N windows, take max lag score.
3. soft_dtw_score    — per-window soft-DTW on [P, D] patch sequences, averaged.

All functions return a single float (higher = more similar / more relevant).
"""

from __future__ import annotations

import numpy as np


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
