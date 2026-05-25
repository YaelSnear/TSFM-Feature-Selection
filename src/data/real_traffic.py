"""METR-LA traffic data loader.

Reads directly from the zip using h5py (no pytables required).

HDF5 layout (pandas 0.15.2 legacy format):
  df/axis0        (207,)       sensor IDs as byte strings
  df/axis1        (34272,)     timestamps as int64 nanoseconds since epoch
  df/block0_values (34272, 207) float64 speed values

Adjacency pickle layout (list of 3):
  [0] list of 207 sensor ID strings
  [1] dict  sensor_id → column index
  [2] ndarray (207, 207) weighted adjacency matrix
"""

from __future__ import annotations

import pickle
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MetrLaData:
    df: pd.DataFrame          # (T, N_sensors), DatetimeIndex, columns = sensor ID strings
    sensor_ids: list[str]
    adj_matrix: np.ndarray    # (N_sensors, N_sensors)
    sensor_id_to_idx: dict[str, int]
    metadata: dict = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return self.df.shape[1]

    @property
    def n_timestamps(self) -> int:
        return self.df.shape[0]


def _find_files(names: list[str]) -> tuple[str | None, str | None]:
    """Return (data_file, adjacency_file) by scanning zip member names."""
    data_file = next((n for n in names if n.endswith(".h5")), None)
    adj_file  = next((n for n in names if "adj" in n.lower() and n.endswith(".pkl")), None)
    return data_file, adj_file


def _load_h5(h5_path: str | Path, n_rows: int | None) -> tuple[pd.DataFrame, list[str]]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        sensor_ids = [s.decode() for s in f["df/axis0"][:]]
        ts_raw     = f["df/axis1"][:]
        values     = f["df/block0_values"][:]   # (T, N)

    if n_rows is not None:
        values = values[:n_rows]
        ts_raw = ts_raw[:n_rows]

    timestamps = pd.to_datetime(ts_raw, unit="ns")
    df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
    df.index.name = "timestamp"
    return df, sensor_ids


def _load_adj(pkl_path: str | Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    with open(pkl_path, "rb") as f:
        adj = pickle.load(f, encoding="latin1")
    sensor_ids      = adj[0]   # list[str]
    sensor_id_to_idx = adj[1]  # dict[str, int]
    adj_matrix       = adj[2]  # ndarray (N, N)
    return sensor_ids, sensor_id_to_idx, adj_matrix


@dataclass
class ForecastWindows:
    X_context: np.ndarray       # (n_windows, context_length, n_sensors)
    y_target: np.ndarray        # (n_windows, horizon)
    timestamps: pd.DatetimeIndex  # first forecast timestamp per window, length n_windows
    target_sensor: str
    metadata: dict = field(default_factory=dict)


def make_forecast_windows(
    df: pd.DataFrame,
    target_sensor: str | None = None,
    context_length: int = 288,
    horizon: int = 12,
    stride: int = 12,
    max_windows: int | None = None,
) -> ForecastWindows:
    """Slide a forecasting window over a multivariate traffic DataFrame.

    Each window consists of:
      - context : df.iloc[start : start+context_length]  — all sensors
      - target  : df[target_sensor].iloc[start+context_length : start+context_length+horizon]

    Args:
        df             : (T, N_sensors) DataFrame with DatetimeIndex
        target_sensor  : column name; None → first column
        context_length : number of past steps fed as input
        horizon        : number of future steps to predict
        stride         : step between consecutive window starts
        max_windows    : cap on returned windows (None = all valid windows)

    Returns:
        ForecastWindows with X_context (n_windows, context_length, N),
        y_target (n_windows, horizon), and aligned timestamps.
    """
    if target_sensor is None:
        target_sensor = df.columns[0]
    if target_sensor not in df.columns:
        raise ValueError(f"target_sensor {target_sensor!r} not in DataFrame columns.")

    T = len(df)
    window_size = context_length + horizon
    if T < window_size:
        raise ValueError(
            f"DataFrame too short (T={T}) for context_length={context_length} + horizon={horizon}."
        )

    n_valid = (T - window_size) // stride + 1
    n_windows = n_valid if max_windows is None else min(n_valid, max_windows)

    n_sensors = df.shape[1]
    values    = df.values                          # (T, N)
    target_col = df.columns.get_loc(target_sensor)

    X_context = np.empty((n_windows, context_length, n_sensors), dtype=np.float64)
    y_target  = np.empty((n_windows, horizon),                   dtype=np.float64)
    ts_list   = []

    for i in range(n_windows):
        s = i * stride
        X_context[i] = values[s : s + context_length]
        y_target[i]  = values[s + context_length : s + window_size, target_col]
        ts_list.append(df.index[s + context_length])

    return ForecastWindows(
        X_context=X_context,
        y_target=y_target,
        timestamps=pd.DatetimeIndex(ts_list),
        target_sensor=target_sensor,
        metadata={
            "context_length": context_length,
            "horizon": horizon,
            "stride": stride,
            "n_valid_windows": n_valid,
            "n_windows_returned": n_windows,
        },
    )


def select_sensors(
    data: "MetrLaData",
    target_sensor: str,
    n_relevant: int = 5,
    n_distractor: int = 10,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Return df column indices for proxy-relevant and distractor sensors.

    proxy_relevant : top-n_relevant sensors by adjacency weight (excluding self).
    distractor     : n_distractor sensors randomly sampled from the zero-adjacency
                     pool (adj < 1e-10) using a fixed seed to avoid spatial bias.

    Returns a dict with keys "proxy_relevant" and "distractor", each a list of
    integer df column indices suitable for indexing into ForecastWindows.X_context.
    """
    if target_sensor not in data.sensor_id_to_idx:
        raise ValueError(f"target_sensor {target_sensor!r} not in adjacency index.")

    target_adj_idx = data.sensor_id_to_idx[target_sensor]

    # Build reverse map: adj_matrix_index → df column index (via sensor ID bridge).
    adj_to_df_col: dict[int, int] = {}
    for sid, adj_i in data.sensor_id_to_idx.items():
        if sid in data.df.columns:
            adj_to_df_col[adj_i] = data.df.columns.get_loc(sid)

    adj_row = data.adj_matrix[target_adj_idx].copy()
    adj_row[target_adj_idx] = -1.0  # exclude self from ranking

    # Proxy relevant: top n_relevant adj-matrix neighbours by weight.
    sorted_adj = np.argsort(-adj_row)
    proxy_relevant = [adj_to_df_col[i] for i in sorted_adj[:n_relevant] if i in adj_to_df_col]

    # Distractor: random sample from sensors with zero adjacency to target.
    zero_mask = data.adj_matrix[target_adj_idx] < 1e-10
    zero_mask[target_adj_idx] = False
    zero_adj_indices = np.where(zero_mask)[0]
    if len(zero_adj_indices) < n_distractor:
        raise ValueError(
            f"Only {len(zero_adj_indices)} zero-adjacency sensors available, "
            f"need {n_distractor}."
        )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(zero_adj_indices, size=n_distractor, replace=False)
    distractor = [adj_to_df_col[i] for i in sampled if i in adj_to_df_col]

    return {"proxy_relevant": proxy_relevant, "distractor": distractor}


def load_metr_la(zip_path: str | Path, n_rows: int | None = None) -> MetrLaData:
    """Load METR-LA data from zip.

    Args:
        zip_path : path to METR-LA.zip
        n_rows   : number of time steps to load (None = all 34272)

    Returns:
        MetrLaData with df, sensor_ids, adj_matrix, sensor_id_to_idx, metadata
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist()]
        data_file, adj_file = _find_files(names)

        if data_file is None:
            raise RuntimeError(f"No .h5 file found in {zip_path}. Contents: {names}")

        with tempfile.TemporaryDirectory() as tmp:
            z.extract(data_file, tmp)
            df, sensor_ids = _load_h5(Path(tmp) / data_file, n_rows)

            adj_matrix, sensor_id_to_idx = np.zeros((0, 0)), {}
            if adj_file:
                z.extract(adj_file, tmp)
                _, sensor_id_to_idx, adj_matrix = _load_adj(Path(tmp) / adj_file)

    return MetrLaData(
        df=df,
        sensor_ids=sensor_ids,
        adj_matrix=adj_matrix,
        sensor_id_to_idx=sensor_id_to_idx,
        metadata={
            "source": str(zip_path),
            "zip_contents": names,
            "data_file": data_file,
            "adj_file": adj_file,
            "n_rows_requested": n_rows,
        },
    )
