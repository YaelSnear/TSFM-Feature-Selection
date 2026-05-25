"""Synthetic dataset: relevant features are lagged/noisy copies of the target."""

import numpy as np
from types import SimpleNamespace

from src.data import InstanceData


def _ar1(length: int, rng: np.random.Generator, phi: float = 0.8) -> np.ndarray:
    y = np.zeros(length)
    for t in range(1, length):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


def generate_instance(
    n_features: int,
    n_relevant: int,
    series_length: int,
    max_lag: int,
    noise_scale: float,
    rng: np.random.Generator,
    instance_id: int = 0,
) -> InstanceData:
    y = _ar1(series_length, rng)

    lags = rng.choice(np.arange(1, max_lag + 1), size=n_relevant, replace=False)

    X_raw = np.zeros((series_length, n_features))
    for i in range(n_relevant):
        lag = int(lags[i])
        X_raw[lag:, i] = y[:-lag] + noise_scale * rng.standard_normal(series_length - lag)
        X_raw[:lag, i] = noise_scale * rng.standard_normal(lag)
    for i in range(n_relevant, n_features):
        X_raw[:, i] = rng.standard_normal(series_length)

    # Shuffle columns so relevant features are not always first.
    perm = rng.permutation(n_features)
    X = X_raw[:, perm]
    inv_perm = np.argsort(perm)
    relevant_indices = sorted(inv_perm[:n_relevant].tolist())

    feature_names = [f"feat_{i:03d}" for i in range(n_features)]

    return InstanceData(
        X=X,
        y=y,
        feature_names=feature_names,
        relevant_feature_indices=relevant_indices,
        metadata={
            "instance_id": instance_id,
            "lags": lags.tolist(),
            "perm": perm.tolist(),
        },
    )


def generate_dataset(cfg: SimpleNamespace) -> list[InstanceData]:
    rng = np.random.default_rng(cfg.seed)
    return [
        generate_instance(
            n_features=cfg.n_features,
            n_relevant=cfg.n_relevant,
            series_length=cfg.series_length,
            max_lag=cfg.max_lag,
            noise_scale=cfg.noise_scale,
            rng=rng,
            instance_id=i,
        )
        for i in range(cfg.n_instances)
    ]
