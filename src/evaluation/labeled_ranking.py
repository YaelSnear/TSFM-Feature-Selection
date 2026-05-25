"""Ranking metrics for labeled experiments (ground-truth relevant features known)."""

import numpy as np


def _ranked_indices(scores: np.ndarray) -> np.ndarray:
    """Feature indices sorted from highest to lowest score."""
    return np.argsort(-scores)


def precision_at_k(scores: np.ndarray, relevant: list[int], k: int) -> float:
    top_k = set(_ranked_indices(scores)[:k])
    return len(top_k & set(relevant)) / k


def recall_at_k(scores: np.ndarray, relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(_ranked_indices(scores)[:k])
    return len(top_k & set(relevant)) / len(relevant)


def average_precision(scores: np.ndarray, relevant: list[int]) -> float:
    if not relevant:
        return 0.0
    ranked = _ranked_indices(scores)
    rel_set = set(relevant)
    hits, cumulative = 0, 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in rel_set:
            hits += 1
            cumulative += hits / rank
    return cumulative / len(relevant)


def ndcg_at_k(scores: np.ndarray, relevant: list[int], k: int) -> float:
    rel_set = set(relevant)
    ranked = _ranked_indices(scores)[:k]
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, idx in enumerate(ranked, start=1)
        if idx in rel_set
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_relevant_rank(scores: np.ndarray, relevant: list[int]) -> float:
    if not relevant:
        return float("nan")
    ranked = _ranked_indices(scores).tolist()
    reciprocal_ranks = [1.0 / (ranked.index(r) + 1) for r in relevant if r in ranked]
    return float(np.mean(reciprocal_ranks))


def evaluate(
    scores: np.ndarray,
    relevant: list[int],
    k_values: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"P@{k}"] = precision_at_k(scores, relevant, k)
        metrics[f"R@{k}"] = recall_at_k(scores, relevant, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(scores, relevant, k)
    metrics["AP"] = average_precision(scores, relevant)
    metrics["MRR"] = mean_relevant_rank(scores, relevant)
    return metrics
