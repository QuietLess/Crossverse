"""Stage A: cheap multi-source candidate retrieval (union of top-N per retriever)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crossverse.features.themes import THEMES, extract_themes
from crossverse.retrieval.base import Catalog, History, Retriever, top_k
from crossverse.retrieval.content import ContentRetriever


@dataclass
class CandidateSet:
    idx: np.ndarray  # candidate item indices
    scores: dict[str, np.ndarray]  # retriever -> score for each candidate
    ranks: dict[str, np.ndarray]  # retriever -> rank (0 = best) among target-domain items, capped
    sources: list[list[str]]  # retrievers that nominated each candidate


class CandidateGenerator:
    def __init__(self, catalog: Catalog, retrievers: dict[str, Retriever], per_source: int = 100,
                 rank_cap: int = 1000):
        self.catalog = catalog
        self.retrievers = retrievers
        self.per_source = per_source
        self.rank_cap = rank_cap

    def all_scores(self, history: History, preferences: list[str] | None = None,
                   free_text: str = "") -> dict[str, np.ndarray]:
        out = {}
        has_history = len(history) > 0
        for name, r in self.retrievers.items():
            if name == "content" and not has_history and (preferences or free_text):
                assert isinstance(r, ContentRetriever)
                out[name] = r.score_query(free_text, preferences)
            elif not has_history and name != "popularity":
                out[name] = np.zeros(len(self.catalog), dtype=np.float32)
            else:
                out[name] = r.score(history)
        if not has_history and (preferences or free_text):
            # Cold start: popular items *within* the requested themes, so the popularity source
            # nominates candidates that respect the user's stated intent.
            match = self.theme_match(preferences, free_text)
            out["popularity"] = out["popularity"] + 0.5 * match
        return out

    def theme_match(self, preferences: list[str] | None, free_text: str = "") -> np.ndarray:
        """Number of requested themes each catalog item carries."""
        content = self.retrievers["content"]
        assert isinstance(content, ContentRetriever)
        wanted = set(preferences or []) | set(extract_themes(free_text))
        cols = [i for i, t in enumerate(THEMES) if t in wanted]
        if not cols:
            return np.zeros(len(self.catalog), dtype=np.float32)
        return (content.theme_matrix_[:, cols] > 0).sum(axis=1).astype(np.float32)

    def generate(self, history: History, target_domain: str | None, exclude: np.ndarray | None = None,
                 preferences: list[str] | None = None, free_text: str = "") -> CandidateSet:
        mask = self.catalog.mask(target_domain)
        full = self.all_scores(history, preferences, free_text)
        nominated: dict[int, list[str]] = {}
        for name, s in full.items():
            if not np.any(s[mask]):
                continue  # retriever has no opinion (e.g. copref for a same-domain query)
            n = self.per_source if name != "popularity" else self.per_source // 2
            for i in top_k(s, n, mask=mask, exclude=exclude):
                nominated.setdefault(int(i), []).append(name)
        if not nominated:  # nothing at all (empty history, no preferences): popularity fallback
            for i in top_k(full["popularity"], self.per_source, mask=mask, exclude=exclude):
                nominated.setdefault(int(i), []).append("popularity")
        idx = np.fromiter(nominated.keys(), dtype=np.int64)
        scores, ranks = {}, {}
        for name, s in full.items():
            scores[name] = s[idx].astype(np.float32)
            # Rank of each candidate within the target domain(s), capped: only the top `rank_cap`
            # need exact ranks, so partition instead of sorting the whole catalog.
            lookup = np.full(len(self.catalog), self.rank_cap, dtype=np.int64)
            top = top_k(s, self.rank_cap, mask=mask)
            lookup[top] = np.arange(len(top))
            ranks[name] = lookup[idx]
        return CandidateSet(idx, scores, ranks, [nominated[int(i)] for i in idx])
