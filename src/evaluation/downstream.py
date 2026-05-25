"""Downstream Ridge Regression evaluation for feature selection validation.

Design principle: latent scores are used only for sensor *selection*.
All forecasting is done on RAW traffic values so that any RMSE improvement
is attributable to the quality of the feature selection, not to the
representation itself.

Feature matrix (input to Ridge):
    [Y_context (144) | X1_context (144) | ... | Xk_context (144)]
    → shape [N, 144 * (1 + top_k)]

Baselines evaluated alongside scored methods:
    Univariate  — Y context only → [N, 144]
    Geographic  — Y + all 5 proxy_relevant sensors → [N, 144 * 6]
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.evaluation.labeled_ranking import mean_relevant_rank, precision_at_k


def _fit_and_evaluate(
    feat_matrix: np.ndarray,
    y_target: np.ndarray,
    test_frac: float,
    gap: int = 144,
) -> dict[str, float]:
    """Fit RidgeCV on train rows, evaluate on test rows separated by a gap.

    The gap prevents leakage from stride-1 windows that share 143 of 144
    context steps between the last train window and first test window.

    Temporal order is preserved — no shuffle.
    """
    N = feat_matrix.shape[0]
    n_train = int(N * (1.0 - test_frac))
    test_start = n_train + gap
    if test_start >= N:
        # Gap exceeds available data; fall back to no gap so evaluation can proceed.
        print(f"    [warn] gap={gap} too large for N={N}; using no gap (test_start={n_train})")
        test_start = n_train

    X_tr_raw = feat_matrix[:n_train]
    X_te_raw = feat_matrix[test_start:]
    y_tr = y_target[:n_train]
    y_te = y_target[test_start:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    model = MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mse  = float(mean_squared_error(y_te, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_te, y_pred))
    r2   = float(r2_score(y_te, y_pred))

    mask = y_te != 0
    mape = float(np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def _proxy_ranking_metrics(
    feat_scores: dict[int, float],
    all_sensor_df_cols: list[int],
    proxy_relevant_df_cols: list[int],
    top_k: int,
) -> dict[str, float]:
    """Compute Precision@K and MRR against proxy_relevant ground truth.

    Converts feat_scores (keyed by df column index) to a numpy array ordered
    by all_sensor_df_cols, then calls existing labeled_ranking functions.
    """
    scores_arr = np.array([feat_scores[col] for col in all_sensor_df_cols])
    proxy_positions = [all_sensor_df_cols.index(col) for col in proxy_relevant_df_cols]
    return {
        f"Precision_at_{top_k}": precision_at_k(scores_arr, proxy_positions, k=top_k),
        "MRR": mean_relevant_rank(scores_arr, proxy_positions),
    }


def build_feature_matrix(
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    selected_df_cols: list[int],
) -> np.ndarray:
    """Concatenate Y context and selected X_i context windows into a flat matrix.

    Args:
        Y_series         : [N, context_length] raw Y context windows
        raw_windows_X    : {df_col: [N, context_length]} raw X_i context windows
        selected_df_cols : ordered list of df column indices to include

    Returns:
        np.ndarray of shape [N, context_length * (1 + len(selected_df_cols))]
    """
    parts = [Y_series] + [raw_windows_X[col] for col in selected_df_cols]
    return np.hstack(parts)


def evaluate_method(
    feat_scores: dict[int, float],
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    proxy_relevant_df_cols: list[int],
    top_k: int,
    test_frac: float,
) -> dict[str, float]:
    """Full evaluation for one (method, condition, layer) combination.

    1. Rank features by score, select top-K.
    2. Build raw feature matrix [N, 144*(1+top_k)].
    3. Train/test RidgeCV with StandardScaler and 144-step gap; compute MSE/RMSE/MAE/MAPE/R2.
    4. Compute Precision@K and MRR against proxy_relevant ground truth.

    Returns a dict of all metrics.
    """
    top_k_cols = sorted(feat_scores, key=feat_scores.__getitem__, reverse=True)[:top_k]

    feat_matrix = build_feature_matrix(Y_series, raw_windows_X, top_k_cols)
    downstream = _fit_and_evaluate(feat_matrix, y_target, test_frac)
    proxy = _proxy_ranking_metrics(
        feat_scores, all_sensor_df_cols, proxy_relevant_df_cols, top_k
    )
    return {**downstream, **proxy}


def evaluate_univariate_baseline(
    Y_series: np.ndarray,
    y_target: np.ndarray,
    test_frac: float,
) -> dict[str, float]:
    """Univariate baseline: forecast using Y's own past values only."""
    metrics = _fit_and_evaluate(Y_series, y_target, test_frac)
    metrics.update({f"Precision_at_5": float("nan"), "MRR": float("nan")})
    return metrics


def evaluate_geographic_baseline(
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    proxy_relevant_df_cols: list[int],
    test_frac: float,
) -> dict[str, float]:
    """Geographic baseline: forecast using Y + all 5 proxy_relevant sensors.

    These sensors are the ground-truth neighbours in the road network,
    selected without any latent-space scoring.
    """
    feat_matrix = build_feature_matrix(Y_series, raw_windows_X, proxy_relevant_df_cols)
    metrics = _fit_and_evaluate(feat_matrix, y_target, test_frac)
    # Precision@5 = 1.0 by construction (we select exactly the proxy_relevant sensors).
    # MRR is NaN — geographic uses known labels, not a scored ranking.
    metrics.update({"Precision_at_5": 1.0, "MRR": float("nan")})
    return metrics
