"""Equivalence test: vectorized lagged_cka vs original Python-loop reference.

Run this after changing src/scoring/tsfm_scorers.py to verify the vectorized
implementation produces numerically identical results to the original loop.

Usage:
    conda run --no-capture-output -n yael_env python scripts/test_lagged_cka_equiv.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ---------------------------------------------------------------------------
# Reference implementation (original Python loop, kept here for comparison)
# ---------------------------------------------------------------------------

def _cka_core_ref(Hx: np.ndarray, Hy: np.ndarray) -> float:
    Hxc = Hx - Hx.mean(axis=0)
    Hyc = Hy - Hy.mean(axis=0)
    num   = np.linalg.norm(Hxc.T @ Hyc, "fro") ** 2
    denom = (np.linalg.norm(Hxc.T @ Hxc, "fro") *
             np.linalg.norm(Hyc.T @ Hyc, "fro"))
    return float(num / denom) if denom > 1e-10 else 0.0


def lagged_cka_reference(X_emb: np.ndarray, Y_emb: np.ndarray, max_lag: int) -> float:
    """Original Python-loop implementation — reference only, not used in experiments."""
    N, P, D = X_emb.shape
    actual_max_lag = min(max_lag, P - 1)
    best_score = float("-inf")
    for k in range(-actual_max_lag, actual_max_lag + 1):
        window_scores = []
        for w in range(N):
            if k > 0:
                hx = X_emb[w, k:, :]
                hy = Y_emb[w, :P - k, :]
            elif k < 0:
                hx = X_emb[w, :P + k, :]
                hy = Y_emb[w, -k:, :]
            else:
                hx = X_emb[w]
                hy = Y_emb[w]
            if hx.shape[0] < 2 or hy.shape[0] < 2:
                continue
            window_scores.append(_cka_core_ref(hx, hy))
        if window_scores:
            lag_score = float(np.mean(window_scores))
            if lag_score > best_score:
                best_score = lag_score
    return best_score if best_score > float("-inf") else 0.0


# ---------------------------------------------------------------------------
# Equivalence test
# ---------------------------------------------------------------------------

def test_equivalence() -> None:
    from src.scoring.tsfm_scorers import lagged_cka

    rng = np.random.default_rng(42)
    max_lag = 5
    tol = 1e-4  # relative error tolerance

    test_cases = [
        (5,  4,  8,   "tiny (N=5, P=4, D=8)"),
        (20, 6,  16,  "small (N=20, P=6, D=16)"),
        (50, 11, 64,  "medium (N=50, P=11, D=64)"),
        (100, 11, 128, "near-real (N=100, P=11, D=128)"),
    ]

    all_passed = True

    for N, P, D, name in test_cases:
        X = rng.standard_normal((N, P, D)).astype(np.float32)
        Y = rng.standard_normal((N, P, D)).astype(np.float32)

        ref = lagged_cka_reference(X, Y, max_lag)
        vec = lagged_cka(X, Y, max_lag)

        rel_err = abs(ref - vec) / (abs(ref) + 1e-12)
        status = "PASS" if rel_err < tol else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {status} [{name}]  ref={ref:.6f}  vec={vec:.6f}  rel_err={rel_err:.2e}")

    # Edge case: P=2 (minimum valid, max_lag capped to 1)
    X2 = rng.standard_normal((5, 2, 4)).astype(np.float32)
    Y2 = rng.standard_normal((5, 2, 4)).astype(np.float32)
    ref2 = lagged_cka_reference(X2, Y2, max_lag=10)
    vec2 = lagged_cka(X2, Y2, max_lag=10)
    rel_err2 = abs(ref2 - vec2) / (abs(ref2) + 1e-12)
    status2 = "PASS" if rel_err2 < tol else "FAIL"
    if status2 == "FAIL":
        all_passed = False
    print(f"  {status2} [P=2 edge, max_lag capped to P-1=1]  ref={ref2:.6f}  vec={vec2:.6f}  rel_err={rel_err2:.2e}")

    # All-zero input (degenerate — should return 0.0 for both)
    X0 = np.zeros((10, 5, 8), dtype=np.float32)
    Y0 = rng.standard_normal((10, 5, 8)).astype(np.float32)
    ref0 = lagged_cka_reference(X0, Y0, max_lag=3)
    vec0 = lagged_cka(X0, Y0, max_lag=3)
    assert ref0 == 0.0 and vec0 == 0.0, f"FAIL [degenerate] ref={ref0} vec={vec0}"
    print(f"  PASS [degenerate zero input]  ref={ref0}  vec={vec0}")

    if all_passed:
        print("\nAll equivalence tests PASSED.")
    else:
        print("\nSome tests FAILED — do not proceed with full run.")
        sys.exit(1)


if __name__ == "__main__":
    test_equivalence()
