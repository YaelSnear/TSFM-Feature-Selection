"""Unit tests for lagged_cka_window_centered.

Tests:
    1. Identical inputs (X = Y) → score > 0.99
    2. Uncorrelated random inputs → finite, < 1.0, != 1.0
    3. Real cached embeddings → old lagged_cka degenerate, new variant non-constant
    4. Edge cases: max_lag clipping, all-zero X, P=2

Also invokes scripts/test_lagged_cka_equiv.py to confirm the old lagged_cka
is still mathematically unchanged.

Usage:
    conda run --no-capture-output -n yael_env python scripts/test_lagged_cka_fixed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.scoring.tsfm_scorers import lagged_cka, lagged_cka_window_centered

CACHE_DIR = Path("outputs/EXP_tsfm_full_run_all206_20260530_172932/cache")

all_passed = True


def _check(condition: bool, label: str, detail: str = "") -> bool:
    global all_passed
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{': ' + detail if detail else ''}")
        all_passed = False
    return condition


# ---------------------------------------------------------------------------
# Test 1 — Identical inputs (X = Y)
# ---------------------------------------------------------------------------

print("\n[Test 1] Identical inputs (X = Y)")
rng = np.random.default_rng(0)
X1 = rng.standard_normal((50, 11, 64)).astype(np.float32)
score1, lag1 = lagged_cka_window_centered(X1, X1, max_lag=5)
_check(score1 > 0.99, f"score > 0.99  (got {score1:.6f})")
if lag1 != 0:
    print(f"    [warn] best_lag = {lag1} (expected 0; ties at other lags are possible)")
_check(isinstance(lag1, int), f"best_lag is int (got {type(lag1).__name__})")

# ---------------------------------------------------------------------------
# Test 2 — Uncorrelated random inputs
# ---------------------------------------------------------------------------

print("\n[Test 2] Uncorrelated random inputs")
X2 = np.random.default_rng(1).standard_normal((100, 11, 64)).astype(np.float32)
Y2 = np.random.default_rng(2).standard_normal((100, 11, 64)).astype(np.float32)
score2, lag2 = lagged_cka_window_centered(X2, Y2, max_lag=5)
score_id, _ = lagged_cka_window_centered(X2, X2, max_lag=5)
_check(np.isfinite(score2), f"score is finite  (got {score2})")
_check(score2 < score_id, f"score < identical-input score  ({score2:.6f} < {score_id:.6f})")
_check(score2 != 1.0, f"score != 1.0  (got {score2:.6f})")

# ---------------------------------------------------------------------------
# Test 3 — Non-degenerate on real cached embeddings
# ---------------------------------------------------------------------------

print("\n[Test 3] Real cached embeddings (anonymous mathematical check)")
if not CACHE_DIR.exists():
    print(f"  SKIP  cache dir not found: {CACHE_DIR}")
else:
    cache_files = sorted(CACHE_DIR.glob("*.npz"))[:20]
    if len(cache_files) < 5:
        print(f"  SKIP  fewer than 5 cache files found")
    else:
        Y3 = np.load(cache_files[0])["layer_8"]
        old_scores = []
        new_scores = []
        for f in cache_files[1:]:
            X3 = np.load(f)["layer_8"]
            old_scores.append(lagged_cka(X3, Y3, max_lag=5))
            s, _ = lagged_cka_window_centered(X3, Y3, max_lag=5)
            new_scores.append(s)
        old_arr = np.array(old_scores)
        new_arr = np.array(new_scores)
        _check(
            np.all(np.abs(old_arr - 1.0) < 1e-4),
            f"old lagged_cka all ≈ 1.0  (range [{old_arr.min():.6f}, {old_arr.max():.6f}])"
        )
        _check(
            np.std(new_arr) > 1e-6,
            f"new variant non-constant  (std={np.std(new_arr):.6f}, range [{new_arr.min():.6f}, {new_arr.max():.6f}])"
        )

# ---------------------------------------------------------------------------
# Test 4 — Edge cases
# ---------------------------------------------------------------------------

print("\n[Test 4] Edge cases")

# 4a: max_lag > P-1 — should be clipped
rng4 = np.random.default_rng(42)
X4a = rng4.standard_normal((20, 11, 32)).astype(np.float32)
Y4a = rng4.standard_normal((20, 11, 32)).astype(np.float32)
try:
    s4a, l4a = lagged_cka_window_centered(X4a, Y4a, max_lag=100)
    _check(
        isinstance(s4a, float) and isinstance(l4a, int) and np.isfinite(s4a),
        f"max_lag=100 clipped cleanly (score={s4a:.6f}, lag={l4a})"
    )
except Exception as e:
    _check(False, f"max_lag=100 clipped cleanly", str(e))

# 4b: all-zero X → score should be 0.0
X4b = np.zeros((10, 5, 8), dtype=np.float32)
Y4b = rng4.standard_normal((10, 5, 8)).astype(np.float32)
try:
    s4b, _ = lagged_cka_window_centered(X4b, Y4b, max_lag=3)
    _check(s4b == 0.0, f"all-zero X → score=0.0 (got {s4b})")
except Exception as e:
    _check(False, "all-zero X → score=0.0", str(e))

# 4c: P=2 — minimum valid
X4c = rng4.standard_normal((5, 2, 4)).astype(np.float32)
Y4c = rng4.standard_normal((5, 2, 4)).astype(np.float32)
try:
    s4c, l4c = lagged_cka_window_centered(X4c, Y4c, max_lag=10)
    _check(
        isinstance(s4c, float) and isinstance(l4c, int),
        f"P=2 no crash (score={s4c:.6f}, lag={l4c})"
    )
except Exception as e:
    _check(False, "P=2 no crash", str(e))

# ---------------------------------------------------------------------------
# Equivalence check: old lagged_cka is unchanged
# ---------------------------------------------------------------------------

print("\n[Equivalence check] Old lagged_cka still correct (via test_lagged_cka_equiv.py)")
import subprocess
result = subprocess.run(
    ["conda", "run", "--no-capture-output", "-n", "yael_env",
     "python", "scripts/test_lagged_cka_equiv.py"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("  PASS  Old equivalence test passed")
else:
    print("  FAIL  Old equivalence test failed:")
    print(result.stdout[-500:] if result.stdout else "")
    print(result.stderr[-300:] if result.stderr else "")
    all_passed = False

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if all_passed:
    print("All tests PASSED.")
    sys.exit(0)
else:
    print("Some tests FAILED. Do not proceed with diagnostic or rerun.")
    sys.exit(1)
