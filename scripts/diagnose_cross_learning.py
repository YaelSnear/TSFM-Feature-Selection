"""Minimal diagnostic to decide whether to use cross_learning=False or True.

This script is NOT part of the research experiment. Its only purpose is to
determine whether Chronos-2 past_covariates are used when cross_learning=False,
or whether cross_learning=True is required.

Run:
    conda activate yael_env
    python scripts/diagnose_cross_learning.py

Decision:
    - If cond2 (False + good cov) differs from cond1 (target_only): use cross_learning=False
    - If cond2 ≈ cond1 but cond3 (True + good cov) differs: use cross_learning=True
    - If all ≈ cond1: STOP — past_covariates may not be wired correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from chronos import BaseChronosPipeline

MODEL_ID       = "amazon/chronos-2"
CONTEXT_LENGTH = 144
HORIZON        = 12
SEED           = 0


def _to_median(preds: list[torch.Tensor], median_idx: int) -> np.ndarray:
    """Extract median forecast from predict() output. Shape: [len(preds), horizon]."""
    return np.stack([p[0, median_idx, :].cpu().numpy() for p in preds])


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    print(f"[diagnose_cross_learning] Loading {MODEL_ID} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device)
    pipeline.model.eval()
    print(f"[diagnose_cross_learning] Pipeline type : {type(pipeline).__name__}")
    print(f"[diagnose_cross_learning] Quantiles     : {pipeline.quantiles}")

    median_idx = pipeline.quantiles.index(0.5)

    # ------------------------------------------------------------------
    # Synthetic data (seed=0)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)

    # x_good: base noise + strong positive shock in last 12 context steps
    x_good = rng.standard_normal(CONTEXT_LENGTH).astype(np.float32)
    x_good[132:] += 5.0

    # x_bad: inverted shock (opposite direction)
    x_bad = x_good.copy()
    x_bad[132:] *= -1.0

    # y: delayed copy of x_good so y_future ≈ shock region of x_good
    # y[t] = x_good[t - HORIZON] + small noise  →  y[144..155] ≈ x_good[132..143]
    y = np.concatenate([
        rng.standard_normal(HORIZON).astype(np.float32),
        x_good[:-HORIZON],
    ])
    y += rng.standard_normal(CONTEXT_LENGTH).astype(np.float32) * 0.1

    # Ground-truth future = x_good[132:144] (the shock portion)
    y_true_future = x_good[132:].copy()  # shape (12,)

    print(f"\n[diagnose_cross_learning] Context length : {CONTEXT_LENGTH}")
    print(f"[diagnose_cross_learning] Horizon        : {HORIZON}")
    print(f"[diagnose_cross_learning] x_good shock magnitude (last 12 steps mean): "
          f"{x_good[132:].mean():.3f}")
    print(f"[diagnose_cross_learning] y_true_future mean: {y_true_future.mean():.3f}")

    # ------------------------------------------------------------------
    # 4 conditions
    # ------------------------------------------------------------------
    conditions = [
        ("cond1_target_only_False",      False, None),
        ("cond2_good_cov_False",         False, x_good),
        ("cond3_good_cov_True",          True,  x_good),
        ("cond4_bad_cov_True",           True,  x_bad),
    ]

    results: dict[str, np.ndarray] = {}

    for name, cl, cov in conditions:
        if cov is None:
            inputs = [{"target": y}]
        else:
            inputs = [{"target": y, "past_covariates": {"x": cov}}]

        with torch.no_grad():
            preds = pipeline.predict(inputs, prediction_length=HORIZON,
                                     batch_size=1, cross_learning=cl)

        median = _to_median(preds, median_idx)   # shape (1, 12)
        results[name] = median[0]                 # shape (12,)

        finite_ok = bool(np.all(np.isfinite(median)))
        print(f"\n  [{name}]")
        print(f"    output shape  : {preds[0].shape}")
        print(f"    finite        : {finite_ok}")
        print(f"    median[:4]    : {median[0, :4].round(3)}")

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------
    ref = results["cond1_target_only_False"]
    print("\n[diagnose_cross_learning] ── Comparison vs cond1 (target_only, False) ──")
    for name, arr in results.items():
        if name == "cond1_target_only_False":
            continue
        mad  = _mean_abs_diff(arr, ref)
        mae  = _mae(arr, y_true_future)
        mae1 = _mae(ref, y_true_future)
        impr = mae1 - mae
        print(f"  {name}")
        print(f"    mean_abs_diff vs target_only : {mad:.5f}")
        print(f"    MAE vs synthetic future      : {mae:.5f}  (target_only: {mae1:.5f})")
        print(f"    MAE improvement over target_only: {impr:+.5f}")

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    mad_cond2 = _mean_abs_diff(results["cond2_good_cov_False"], ref)
    mad_cond3 = _mean_abs_diff(results["cond3_good_cov_True"],  ref)

    THRESHOLD = 1e-6

    print("\n[diagnose_cross_learning] ── DECISION ──")
    if mad_cond2 > THRESHOLD:
        print(f"  cond2 (cross_learning=False + good cov) differs from target_only "
              f"(mean_abs_diff={mad_cond2:.6f}).")
        print("  VERDICT: use cross_learning=False  ← past_covariates are active with False")
        verdict = "False"
    elif mad_cond3 > THRESHOLD:
        print(f"  cond2 (False) ≈ target_only (mean_abs_diff={mad_cond2:.6f}).")
        print(f"  cond3 (cross_learning=True + good cov) differs "
              f"(mean_abs_diff={mad_cond3:.6f}).")
        print("  VERDICT: use cross_learning=True")
        verdict = "True"
    else:
        print(f"  cond2 (False) ≈ target_only (mad={mad_cond2:.6f})")
        print(f"  cond3 (True)  ≈ target_only (mad={mad_cond3:.6f})")
        print("  VERDICT: STOP — past_covariates may not be wired correctly.")
        print("  Do not proceed with the experiment until this is investigated.")
        verdict = "STOP"

    print(f"\n[diagnose_cross_learning] Set CROSS_LEARNING = {verdict} in run_experiment_tsfm.py")


if __name__ == "__main__":
    main()
