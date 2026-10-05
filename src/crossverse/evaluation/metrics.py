"""Ranking and beyond-accuracy metrics. All functions take a ranked list and a relevant set."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for i in ranked[:k] if i in relevant)
    return hits / min(len(relevant), k)


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return float(any(i in relevant for i in ranked[:k]))


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(ranked[:k]) if i in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for r, i in enumerate(ranked[:k]):
        if i in relevant:
            hits += 1
            total += hits / (r + 1)
    return total / min(len(relevant), k)


def intra_list_diversity(ranked: Sequence[int], embeddings: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of the list (embeddings assumed L2-normalised)."""
    if len(ranked) < 2:
        return 0.0
    E = embeddings[np.asarray(ranked)]
    S = E @ E.T
    n = len(ranked)
    return float(1.0 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def novelty(ranked: Sequence[int], popularity: np.ndarray, n_users: int) -> float:
    """Mean self-information -log2(p(item)) of recommended items."""
    if len(ranked) == 0:
        return 0.0
    p = (popularity[np.asarray(ranked)] + 1.0) / (n_users + 1.0)
    return float(np.mean(-np.log2(p)))


def popularity_percentile(ranked: Sequence[int], pop_rank_pct: np.ndarray) -> float:
    """Average popularity percentile (1.0 = most popular item); a popularity-bias diagnostic."""
    if len(ranked) == 0:
        return 0.0
    return float(np.mean(pop_rank_pct[np.asarray(ranked)]))
