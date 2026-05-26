"""Downstream LightGBM evaluation for feature selection validation.

Design principle: latent scores are used only for sensor *selection*.
All forecasting is done on RAW traffic values so that any RMSE improvement
is attributable to the quality of the feature selection, not to the
representation itself.

Feature matrix (input to LightGBM):
    [Y_context (144) | X1_context (144) | ... | Xk_context (144)]
    → shape [N, 144 * (1 + top_k)]

Baselines evaluated alongside scored methods:
    Univariate  — Y context only → [N, 144]
    Geographic  — Y + top-K sensors ranked by adjacency weight → [N, 144 * (1+K)]

All evaluate_* functions return (metrics_dict, selected_df_cols) so the
orchestrator can track which sensors each method chose at every K.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.labeled_ranking import mean_relevant_rank, precision_at_k


def _fit_and_evaluate(
    feat_matrix: np.ndarray,
    y_target: np.ndarray,
    test_frac: float,
    gap: int = 144,
) -> dict[str, float]:
    """Fit a single LightGBM on the mean-horizon target, evaluate on held-out rows.

    Using a single regressor on y_target.mean(axis=1) instead of
    MultiOutputRegressor × 12 gives a 12× speedup with negligible impact
    on sensor ranking quality.  All metrics are computed on the mean target,
    consistent with how Lasso and RF baselines aggregate y.

    The gap prevents leakage from stride-12 windows where consecutive windows
    share context steps.  Temporal order is preserved — no shuffle.
    """
    N = feat_matrix.shape[0]
    n_train = int(N * (1.0 - test_frac))
    test_start = n_train + gap
    if test_start >= N:
        print(f"    [warn] gap={gap} too large for N={N}; using no gap (test_start={n_train})")
        test_start = n_train

    y_mean   = y_target.mean(axis=1)   # [N] — mean over 12 horizons
    X_tr     = feat_matrix[:n_train]
    X_te     = feat_matrix[test_start:]
    y_tr     = y_mean[:n_train]
    y_te     = y_mean[test_start:]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
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

    The precision key is always "Precision_at_K" — the numeric K value is
    stored in the calling row's "k" column, avoiding dynamic column names.
    """
    scores_arr = np.array([feat_scores[col] for col in all_sensor_df_cols])
    proxy_positions = [all_sensor_df_cols.index(col) for col in proxy_relevant_df_cols]
    return {
        "Precision_at_K": precision_at_k(scores_arr, proxy_positions, k=top_k),
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


def compute_ipg_ground_truth(
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    test_frac: float,
    n_top: int = 5,
) -> list[int]:
    """Rank candidate sensors by Incremental Predictive Gain, train split only.

    IPG(c) = RMSE_univariate - RMSE(Y + X_c), evaluated on an internal
    temporal validation fold carved from the training window.  The test split
    is never touched, ensuring zero leakage.

    Internal split layout (all within rows 0..n_train):
        inner_train : rows [0, n_inner_train)   — 80 % of train
        inner_val   : rows [n_inner_train, n_train) — 20 % of train
        (temporal order preserved, no shuffle, no gap needed inside train)

    A single LGBMRegressor is fit per sensor on the mean of the 12 forecast
    horizons.  This is consistent with how Lasso/RF supervised baselines
    aggregate the target and is computationally tractable (51 fits total).

    Args:
        Y_series           : [N, context_length] target context windows
        raw_windows_X      : {col: [N, context_length]} candidate context windows
        y_target           : [N, 12] forecast targets
        all_sensor_df_cols : candidate column indices to rank
        test_frac          : fraction used for the main train/test split
        n_top              : number of top sensors to return (default 5)

    Returns:
        Ordered list of top n_top column indices by IPG, highest first.
    """
    N = Y_series.shape[0]
    n_train = int(N * (1.0 - test_frac))

    n_inner_val   = max(int(n_train * 0.20), 1)
    n_inner_train = n_train - n_inner_val

    y_mean     = y_target.mean(axis=1)           # [N] — mean over 12 horizons
    y_inner_tr = y_mean[:n_inner_train]
    y_inner_val = y_mean[n_inner_train:n_train]

    _lgbm = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    # Univariate RMSE — Y context only
    _lgbm.fit(Y_series[:n_inner_train], y_inner_tr)
    rmse_uni = float(np.sqrt(np.mean(
        (y_inner_val - _lgbm.predict(Y_series[n_inner_train:n_train])) ** 2
    )))

    # Per-sensor RMSE — Y context + candidate sensor context
    ipg_scores: dict[int, float] = {}
    for col in all_sensor_df_cols:
        X_tr_ipg  = np.hstack([Y_series[:n_inner_train],
                                raw_windows_X[col][:n_inner_train]])
        X_val_ipg = np.hstack([Y_series[n_inner_train:n_train],
                                raw_windows_X[col][n_inner_train:n_train]])
        _lgbm.fit(X_tr_ipg, y_inner_tr)
        rmse_with = float(np.sqrt(np.mean(
            (y_inner_val - _lgbm.predict(X_val_ipg)) ** 2
        )))
        ipg_scores[col] = rmse_uni - rmse_with   # positive → sensor reduces error

    return sorted(ipg_scores, key=ipg_scores.__getitem__, reverse=True)[:n_top]


def evaluate_method(
    feat_scores: dict[int, float],
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    proxy_relevant_df_cols: list[int],
    top_k: int,
    test_frac: float,
) -> tuple[dict[str, float], list[int]]:
    """Full evaluation for one (method, condition, layer, k) combination.

    1. Rank features by score, select top-K.
    2. Build raw feature matrix [N, 144*(1+top_k)].
    3. Train/test LightGBM with 144-step gap; compute MSE/RMSE/MAE/MAPE/R2.
    4. Compute Precision@K and MRR against proxy_relevant ground truth.

    Returns:
        (metrics_dict, selected_df_cols)
    """
    top_k_cols = sorted(feat_scores, key=feat_scores.__getitem__, reverse=True)[:top_k]

    feat_matrix = build_feature_matrix(Y_series, raw_windows_X, top_k_cols)
    downstream = _fit_and_evaluate(feat_matrix, y_target, test_frac)
    proxy = _proxy_ranking_metrics(
        feat_scores, all_sensor_df_cols, proxy_relevant_df_cols, top_k
    )
    return {**downstream, **proxy}, top_k_cols


def evaluate_univariate_baseline(
    Y_series: np.ndarray,
    y_target: np.ndarray,
    test_frac: float,
) -> tuple[dict[str, float], list[int]]:
    """Univariate baseline: forecast using Y's own past values only.

    Returns:
        (metrics_dict, []) — no sensors selected beyond Y itself.
    """
    metrics = _fit_and_evaluate(Y_series, y_target, test_frac)
    metrics.update({"Precision_at_K": float("nan"), "MRR": float("nan")})
    return metrics, []


def evaluate_geographic_baseline(
    Y_series: np.ndarray,
    raw_windows_X: dict[int, np.ndarray],
    y_target: np.ndarray,
    all_sensor_df_cols: list[int],
    proxy_relevant_df_cols: list[int],
    geo_adj_scores: dict[int, float],
    top_k: int,
    test_frac: float,
) -> tuple[dict[str, float], list[int]]:
    """Geographic baseline: rank ALL candidate sensors by adjacency weight, select top-K.

    For K <= n_relevant (e.g. K=5), this selects exactly the ground-truth
    proxy_relevant sensors (Precision@K = 1.0).  For K > n_relevant, extra
    sensors with lower (but non-zero) adjacency weight are included, and
    Precision@K decreases accordingly — correctly reflecting the harder task.

    Args:
        all_sensor_df_cols    : all candidate sensor df column indices
        proxy_relevant_df_cols: ground-truth neighbour indices for Precision@K
        geo_adj_scores        : {col_idx: adjacency_weight} for all candidates
        top_k                 : number of sensors to select

    Returns:
        (metrics_dict, selected_df_cols)
    """
    top_k_cols = sorted(geo_adj_scores, key=geo_adj_scores.__getitem__, reverse=True)[:top_k]
    feat_matrix = build_feature_matrix(Y_series, raw_windows_X, top_k_cols)
    downstream = _fit_and_evaluate(feat_matrix, y_target, test_frac)
    proxy = _proxy_ranking_metrics(
        geo_adj_scores, all_sensor_df_cols, proxy_relevant_df_cols, top_k
    )
    return {**downstream, **proxy}, top_k_cols
