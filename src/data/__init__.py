from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class InstanceData:
    X: np.ndarray               # (T, N_features)
    y: np.ndarray               # (T,)
    feature_names: list[str]
    relevant_feature_indices: list[int] | None = None  # None for unlabeled experiments
    metadata: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def series_length(self) -> int:
        return self.X.shape[0]
