"""Forecasting evaluation metrics: MAE, RMSE, sMAPE, MASE.

Array shapes
------------
All public functions accept:
  1-D (T,)   → single series, returns a Python float
  2-D (N, T) → N series, returns np.ndarray of shape (N,)

NaN policy
----------
Raises ValueError on any NaN in any input array.
Clean your data (drop or impute) before calling these functions.

sMAPE range
-----------
Returns values in [0, 2] (not [0%, 200%]).
Formula: mean( 2|y_true - y_pred| / (|y_true| + |y_pred| + ε) )

MASE denominator
----------------
Computed from the training series alone:
  scale = mean( |y_train[t] - y_train[t - s]| )  for t = s, ..., len(y_train) - 1
where s = seasonality.  y_train must have length > seasonality.
Returns np.nan if scale < epsilon (undefined when naive baseline has zero error).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_no_nan(*arrays: np.ndarray) -> None:
    for arr in arrays:
        if np.isnan(arr).any():
            raise ValueError(
                "Input contains NaN values. Clean data before computing metrics."
            )


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )


# ---------------------------------------------------------------------------
# 1-D core helpers (operate on a single series, return float)
# ---------------------------------------------------------------------------

def _mae_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _smape_1d(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> float:
    return float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    )


def _mase_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int,
    epsilon: float,
) -> float:
    if len(y_train) <= seasonality:
        raise ValueError(
            f"y_train length ({len(y_train)}) must be greater than seasonality ({seasonality})."
        )
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale < epsilon:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# ---------------------------------------------------------------------------
# Dispatch helper: 1-D → scalar, 2-D → array over rows
# ---------------------------------------------------------------------------

def _dispatch(fn, *arrays: np.ndarray) -> float | np.ndarray:
    ndim = arrays[0].ndim
    if ndim == 1:
        return fn(*arrays)
    if ndim == 2:
        return np.array([fn(*[a[i] for a in arrays]) for i in range(arrays[0].shape[0])])
    raise ValueError(f"Expected 1-D or 2-D arrays, got ndim={ndim}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Mean absolute error.

    Returns float for 1-D input, ndarray of shape (N,) for 2-D input.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    _check_no_nan(y_true, y_pred)
    return _dispatch(_mae_1d, y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Root mean squared error.

    Returns float for 1-D input, ndarray of shape (N,) for 2-D input.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    _check_no_nan(y_true, y_pred)
    return _dispatch(_rmse_1d, y_true, y_pred)


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float | np.ndarray:
    """Symmetric mean absolute percentage error (range [0, 2]).

    Formula: mean( 2|y_true - y_pred| / (|y_true| + |y_pred| + ε) )

    Returns float for 1-D input, ndarray of shape (N,) for 2-D input.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    _check_no_nan(y_true, y_pred)
    return _dispatch(lambda a, b: _smape_1d(a, b, epsilon), y_true, y_pred)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
    epsilon: float = 1e-8,
) -> float | np.ndarray:
    """Mean absolute scaled error.

    Scale = mean( |y_train[t] - y_train[t - seasonality]| ) for t ≥ seasonality.
    MASE  = MAE(y_true, y_pred) / scale

    Returns np.nan (not a large number) when scale < epsilon, because MASE is
    undefined when the naive seasonal baseline has near-zero error (e.g. constant
    training series). The caller is responsible for handling np.nan results.

    For 2-D input (N, T), y_train can be:
      - 1-D (T_train,)  : same training series used for all N test series
      - 2-D (N, T_train): per-series training data

    Returns float for 1-D input, ndarray of shape (N,) for 2-D input.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    _check_shapes(y_true, y_pred)
    _check_no_nan(y_true, y_pred, y_train)

    if y_true.ndim == 1:
        if y_train.ndim != 1:
            raise ValueError("For 1-D y_true, y_train must also be 1-D.")
        return _mase_1d(y_true, y_pred, y_train, seasonality, epsilon)

    if y_true.ndim == 2:
        N = y_true.shape[0]
        # Broadcast a shared 1-D training series across all N test series.
        if y_train.ndim == 1:
            y_train = np.stack([y_train] * N)
        if y_train.shape[0] != N:
            raise ValueError(
                f"y_train leading dimension ({y_train.shape[0]}) must match "
                f"y_true leading dimension ({N})."
            )
        return np.array([
            _mase_1d(y_true[i], y_pred[i], y_train[i], seasonality, epsilon)
            for i in range(N)
        ])

    raise ValueError(f"Expected 1-D or 2-D arrays, got ndim={y_true.ndim}.")
