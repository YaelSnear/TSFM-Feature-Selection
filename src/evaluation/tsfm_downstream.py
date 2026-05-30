"""Frozen Chronos-2 downstream forecaster for TSFM feature selection evaluation.

The downstream model is Chronos-2 with frozen weights. Covariates are passed
as past_covariates in the native Chronos2Pipeline.predict() API.

cross_learning diagnostic result: use cross_learning=False.
past_covariates are active when cross_learning=False (mean_abs_diff=0.227 vs target_only).

Metrics are computed jointly over all windows × all horizon steps — no averaging
over the horizon dimension before metric computation.

Outputs per evaluate() call: RMSE, MAE, MAPE, mae_per_horizon (list), rmse_per_horizon (list)
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test_covariate_api(
    pipeline,
    context_length: int = 144,
    horizon: int = 12,
    cross_learning: bool = False,
) -> None:
    """Verify that Chronos-2 past_covariates produce finite, non-trivial predictions.

    Asserts:
        1. Output shape = (1, len(pipeline.quantiles), horizon) for all modes
        2. All values finite
        3. Predictions with covariates differ from target_only predictions

    Raises RuntimeError if any check fails.
    """
    rng = np.random.default_rng(0)

    target = rng.standard_normal(context_length).astype(np.float32)
    cov1   = rng.standard_normal(context_length).astype(np.float32)
    cov5   = {f"c{i}": rng.standard_normal(context_length).astype(np.float32) for i in range(5)}

    n_q            = len(pipeline.quantiles)
    median_idx     = pipeline.quantiles.index(0.5)
    expected_shape = (1, n_q, horizon)

    def _predict(inputs):
        with torch.no_grad():
            return pipeline.predict(inputs, prediction_length=horizon,
                                    batch_size=1, cross_learning=cross_learning)

    preds_only  = _predict([{"target": target}])
    preds_one   = _predict([{"target": target, "past_covariates": {"c0": cov1}}])
    preds_five  = _predict([{"target": target, "past_covariates": cov5}])

    for label, preds in [("target_only", preds_only), ("one_cov", preds_one), ("five_cov", preds_five)]:
        shape = tuple(preds[0].shape)
        if shape != expected_shape:
            raise RuntimeError(f"[smoke_test] {label} shape {shape} ≠ expected {expected_shape}")
        if not torch.isfinite(preds[0]).all():
            raise RuntimeError(f"[smoke_test] {label} contains NaN/Inf")

    if torch.allclose(preds_only[0], preds_one[0]):
        raise RuntimeError(
            "[smoke_test] one-covariate prediction is IDENTICAL to target_only — "
            "past_covariates may not be wired correctly."
        )
    if torch.allclose(preds_only[0], preds_five[0]):
        raise RuntimeError(
            "[smoke_test] five-covariate prediction is IDENTICAL to target_only — "
            "past_covariates may not be wired correctly."
        )

    mid = median_idx
    print(f"[smoke_test] n_quantiles={n_q}  median_idx={mid}", flush=True)
    print(f"[smoke_test] target_only  median[:4] = {preds_only[0][0, mid, :4].cpu().numpy().round(4)}", flush=True)
    print(f"[smoke_test] one_cov      median[:4] = {preds_one[0][0, mid, :4].cpu().numpy().round(4)}", flush=True)
    print(f"[smoke_test] five_cov     median[:4] = {preds_five[0][0, mid, :4].cpu().numpy().round(4)}", flush=True)
    print("[smoke_test] PASSED", flush=True)


# ---------------------------------------------------------------------------
# TSFMForecaster
# ---------------------------------------------------------------------------

class TSFMForecaster:
    """Zero-shot covariate-informed forecasting with frozen Chronos-2.

    The model weights are never updated. The only trained component in the
    broader pipeline are the representation scorers (CKA, Pearson, etc.) which
    operate on train split only.

    cross_learning=False: each forecast window is processed independently.
    past_covariates are active with this setting (confirmed by diagnostic).
    """

    def __init__(
        self,
        pipeline,
        prediction_length: int = 12,
        cross_learning: bool = False,
    ) -> None:
        self.pipeline          = pipeline
        self.prediction_length = prediction_length
        self.cross_learning    = cross_learning
        self._median_idx: int | None = None

    def _get_median_idx(self) -> int:
        if self._median_idx is None:
            qs = self.pipeline.quantiles
            self._median_idx = qs.index(0.5) if 0.5 in qs else int(np.argmin([abs(q - 0.5) for q in qs]))
        return self._median_idx

    def evaluate(
        self,
        target_windows_test: np.ndarray,
        covariate_windows_test: dict[int, np.ndarray],
        y_test: np.ndarray,
        selected_cols: list[int],
        batch_size: int = 256,
    ) -> dict:
        """Evaluate frozen Chronos-2 forecasting on test windows.

        Args:
            target_windows_test    : [N_test, context_length] float32-compatible
            covariate_windows_test : {df_col: [N_test, context_length]}
            y_test                 : [N_test, horizon] ground-truth future values
            selected_cols          : df_col list of covariates to include; [] = target_only
            batch_size             : batch size for pipeline.predict()

        Returns:
            dict with RMSE, MAE, MAPE, mae_per_horizon (list[float]), rmse_per_horizon (list[float])
        """
        N_test = target_windows_test.shape[0]

        if len(selected_cols) > 100:
            print(f"  [warn] {len(selected_cols)} covariates per window — prediction may be slow", flush=True)

        # ------------------------------------------------------------------
        # Build input list for pipeline.predict()
        # ------------------------------------------------------------------
        inputs = []
        for i in range(N_test):
            entry: dict = {"target": target_windows_test[i].astype(np.float32)}
            if selected_cols:
                entry["past_covariates"] = {
                    f"c{col}": covariate_windows_test[col][i].astype(np.float32)
                    for col in selected_cols
                    if col in covariate_windows_test
                }
            inputs.append(entry)

        # ------------------------------------------------------------------
        # Predict (frozen, no gradient)
        # ------------------------------------------------------------------
        with torch.no_grad():
            predictions = self.pipeline.predict(
                inputs,
                prediction_length=self.prediction_length,
                batch_size=batch_size,
                cross_learning=self.cross_learning,
            )

        # ------------------------------------------------------------------
        # Extract median forecasts — shape (N_test, horizon)
        # ------------------------------------------------------------------
        mid = self._get_median_idx()
        # predictions[i].shape = (1, n_quantiles, prediction_length) for 1-d target
        y_pred = np.stack([
            pred[0, mid, :self.prediction_length].cpu().numpy()
            for pred in predictions
        ])  # [N_test, horizon]

        y_true = y_test[:N_test, :self.prediction_length]  # [N_test, horizon]

        # ------------------------------------------------------------------
        # Metrics — computed jointly over all windows × all horizon steps
        # ------------------------------------------------------------------
        diff    = y_pred - y_true                          # [N_test, horizon]
        abs_diff = np.abs(diff)

        mae  = float(abs_diff.mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

        mask = y_true != 0
        if mask.any():
            mape = float(np.mean(abs_diff[mask] / np.abs(y_true[mask])) * 100)
        else:
            mape = float("nan")

        # Per-horizon metrics (length = horizon)
        mae_per_horizon  = abs_diff.mean(axis=0).tolist()
        rmse_per_horizon = np.sqrt((diff ** 2).mean(axis=0)).tolist()

        return {
            "RMSE":             rmse,
            "MAE":              mae,
            "MAPE":             mape,
            "mae_per_horizon":  mae_per_horizon,
            "rmse_per_horizon": rmse_per_horizon,
        }
