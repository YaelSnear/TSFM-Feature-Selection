"""Windowing utilities for embedding extraction. Stub — not connected to smoke test."""

import numpy as np


def make_windows(
    series: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    """Slide a window over a 1-D series and return an array of windows.

    Returns shape (n_windows, window_length).
    """
    T = len(series)
    starts = range(0, T - window_length + 1, stride)
    return np.stack([series[s : s + window_length] for s in starts])
