"""Shared primitives: catalog indexing, history encoding and the retriever interface.

Every model in CrossVerse is a *history -> scores* function. That one contract serves offline
evaluation (fold-in of held-out users), anonymous API profiles ("I liked these 3 movies") and
cold-start, and it keeps the ranking stage model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from crossverse import DOMAINS


def rating_to_weight(rating: float) -> float:
    """Explicit rating -> signed preference weight (dislikes are negative evidence)."""
    if rating >= 5:
        return 1.0
    if rating >= 4:
        return 0.75
    if rating >= 3:
        return 0.0
    return -1.0


@dataclass
class Catalog:
    items: pd.DataFrame
    item_ids: np.ndarray = field(init=False)
    index: dict[str, int] = field(init=False)
    domain: np.ndarray = field(init=False)  # 0 = movie, 1 = game

    def __post_init__(self) -> None:
        self.items = self.items.reset_index(drop=True)
        self.item_ids = self.items["item_id"].to_numpy()
        self.index = {iid: i for i, iid in enumerate(self.item_ids)}
        self.domain = (self.items["domain"].to_numpy() == "game").astype(np.int8)

    def __len__(self) -> int:
        return len(self.item_ids)

    def mask(self, domain: str | None) -> np.ndarray:
        if domain in (None, "both", "all"):
            return np.ones(len(self), dtype=bool)
        return self.domain == DOMAINS.index(domain)

    def idx(self, item_ids) -> np.ndarray:
        return np.array([self.index[i] for i in item_ids if i in self.index], dtype=np.int64)


@dataclass
class History:
    """A user's (or anonymous profile's) interactions, oldest first."""

    idx: np.ndarray
    weight: np.ndarray  # signed preference weights

    @classmethod
    def empty(cls) -> History:
        return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))

    @classmethod
    def from_pairs(cls, catalog: Catalog, item_ids, ratings) -> History:
        idx, w = [], []
        for iid, r in zip(item_ids, ratings, strict=True):
            j = catalog.index.get(iid)
            if j is not None:
                idx.append(j)
                w.append(rating_to_weight(float(r)))
        return cls(np.asarray(idx, dtype=np.int64), np.asarray(w, dtype=np.float32))

    def positives(self) -> History:
        m = self.weight > 0
        return History(self.idx[m], self.weight[m])

    def restrict(self, catalog: Catalog, domain: str | None) -> History:
        if domain in (None, "both", "all"):
            return self
        m = catalog.domain[self.idx] == DOMAINS.index(domain)
        return History(self.idx[m], self.weight[m])

    def __len__(self) -> int:
        return len(self.idx)


@dataclass
class TrainData:
    """Training interactions indexed against the catalog."""

    catalog: Catalog
    interactions: pd.DataFrame  # user_id, item_id, domain, rating, timestamp (+ item_idx, user_idx, weight)
    user_ids: np.ndarray
    positive: sp.csr_matrix  # users x items, 1.0 for positive interactions
    signed: sp.csr_matrix  # users x items, signed preference weights (ordered input for encoders)

    @classmethod
    def build(cls, catalog: Catalog, interactions: pd.DataFrame, positive_threshold: float = 4.0) -> TrainData:
        df = interactions[interactions["item_id"].isin(catalog.index)].copy()
        df = df.sort_values(["user_id", "timestamp"])
        df["item_idx"] = df["item_id"].map(catalog.index).astype(np.int64)
        codes, user_ids = pd.factorize(df["user_id"], sort=False)
        df["user_idx"] = codes
        df["weight"] = df["rating"].map(rating_to_weight).astype(np.float32)
        shape = (len(user_ids), len(catalog))
        pos = df[df["rating"] >= positive_threshold]
        positive = sp.csr_matrix(
            (np.ones(len(pos), dtype=np.float32), (pos["user_idx"], pos["item_idx"])), shape=shape
        )
        signed = sp.csr_matrix((df["weight"].to_numpy(), (df["user_idx"], df["item_idx"])), shape=shape)
        return cls(catalog, df, np.asarray(user_ids), positive, signed)

    def item_popularity(self) -> np.ndarray:
        return np.asarray(self.positive.sum(axis=0)).ravel()

    def user_histories(self) -> dict[int, History]:
        out = {}
        for u, g in self.interactions.groupby("user_idx", sort=False):
            out[int(u)] = History(g["item_idx"].to_numpy(), g["weight"].to_numpy())
        return out


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: TrainData) -> Retriever: ...

    @abstractmethod
    def score(self, history: History) -> np.ndarray:
        """Return a score for every catalog item (higher = better)."""


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None, exclude: np.ndarray | None = None) -> np.ndarray:
    s = scores.astype(np.float64, copy=True)
    if mask is not None:
        s[~mask] = -np.inf
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    part = np.argpartition(-s, k - 1)[:k]
    return part[np.argsort(-s[part], kind="stable")]


def normalize_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)
