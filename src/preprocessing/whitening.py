"""Global ZCA whitening for patch embedding tensors.

Given an embedding array of shape [N_windows, P_patches, D_embed], flatten
all windows+patches to [N*P, D], fit ZCA once, apply, and reshape back.

ZCA formula:
    W = U @ diag(1 / sqrt(S + eps)) @ U.T
    H_white = (H - mean) @ W.T

Because the input covariance is symmetric (Σ = H.T H / n), its SVD satisfies
V = U, so U.T is correct (not Vt from a general SVD).
"""

from __future__ import annotations

import numpy as np


def whiten_embeddings(emb: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply global ZCA whitening to a patch embedding tensor.

    Args:
        emb : np.ndarray of shape [N, P, D]
        eps : regularisation added to singular values before inversion

    Returns:
        Whitened array of the same shape [N, P, D].
        The output has zero mean and approximately identity covariance
        across the N*P samples.
    """
    N, P, D = emb.shape
    flat = emb.reshape(-1, D)                         # [N*P, D]
    flat_c = flat - flat.mean(axis=0)                 # centre columns
    cov = flat_c.T @ flat_c / flat_c.shape[0]         # [D, D] covariance

    # SVD of symmetric covariance: V == U, so we use U.T (not Vt).
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T    # ZCA whitening matrix

    flat_white = flat_c @ W.T                         # [N*P, D]
    return flat_white.reshape(N, P, D)
