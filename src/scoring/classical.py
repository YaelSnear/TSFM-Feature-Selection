"""Classical feature scorers. Each function returns a score array (N_features,),
higher = more relevant."""

import numpy as np
from src.data import InstanceData


def random_scorer(instance: InstanceData, rng: np.random.Generator) -> np.ndarray:
    return rng.random(instance.n_features)


def pearson_scorer(instance: InstanceData) -> np.ndarray:
    y = instance.y
    scores = np.array([
        abs(np.corrcoef(instance.X[:, i], y)[0, 1])
        for i in range(instance.n_features)
    ])
    return np.nan_to_num(scores, nan=0.0)


def lagged_pearson_scorer(instance: InstanceData, max_lag: int = 15) -> np.ndarray:
    y = instance.y
    T = len(y)
    scores = np.zeros(instance.n_features)
    for i in range(instance.n_features):
        best = 0.0
        for lag in range(1, max_lag + 1):
            if lag >= T:
                break
            r = abs(np.corrcoef(instance.X[lag:, i], y[:-lag])[0, 1])
            if not np.isnan(r) and r > best:
                best = r
        scores[i] = best
    return scores


REGISTRY: dict[str, callable] = {
    "random": random_scorer,
    "pearson": pearson_scorer,
    "lagged_pearson": lagged_pearson_scorer,
}
