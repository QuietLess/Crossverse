"""Ranker features (blueprint 5.5). Identical code path offline (training) and online (serving)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from crossverse.ranking.candidates import CandidateSet
from crossverse.retrieval.base import Catalog, History
from crossverse.retrieval.baselines import CoPreferenceRetriever
from crossverse.retrieval.content import ContentRetriever


class FeatureBuilder:
    def __init__(self, catalog: Catalog, content: ContentRetriever, copref: CoPreferenceRetriever,
                 popularity: np.ndarray, retriever_names: list[str]):
        self.catalog = catalog
        self.content = content
        self.copref = copref
        self.retriever_names = retriever_names
        items = catalog.items
        self.log_pop = np.log1p(popularity).astype(np.float32)
        prior_n, prior_mean = 10.0, float(np.nanmean(items["rating_mean"])) if "rating_mean" in items else 4.0
        rm = items.get("rating_mean", pd.Series(prior_mean, index=items.index)).fillna(prior_mean).to_numpy()
        rc = items.get("rating_count", pd.Series(0, index=items.index)).fillna(0).to_numpy()
        self.bayes_rating = ((rm * rc + prior_mean * prior_n) / (rc + prior_n)).astype(np.float32)
        self.log_count = np.log1p(rc).astype(np.float32)
        self.year = items["year"].to_numpy(dtype=np.float32)
        self.year[self.year <= 0] = np.nan
        self.themes = items["themes"].tolist()
        self.creator = items.get("creator", pd.Series("", index=items.index)).fillna("").astype(str).to_numpy()

    @property
    def feature_names(self) -> list[str]:
        cols = []
        for r in self.retriever_names:
            cols += [f"{r}_score", f"{r}_rank", f"{r}_z"]
        return cols + [
            "n_sources", "is_game", "cross_domain_frac", "n_hist", "n_hist_same", "n_hist_other", "n_dislikes",
            "log_pop", "bayes_rating", "log_rating_count", "theme_affinity", "theme_jaccard",
            "disliked_theme_overlap", "max_sim_pos", "mean_sim_pos", "max_sim_neg", "max_colikes",
            "log_sum_colikes", "year_gap", "creator_match", "cold_start",
        ]

    def build(self, history: History, cands: CandidateSet, preferences: list[str] | None = None) -> pd.DataFrame:
        idx = cands.idx
        n = len(idx)
        f: dict[str, np.ndarray] = {}
        for r in self.retriever_names:
            s = cands.scores[r]
            f[f"{r}_score"] = s
            f[f"{r}_rank"] = np.log1p(cands.ranks[r]).astype(np.float32)
            sd = s.std()
            f[f"{r}_z"] = (s - s.mean()) / sd if sd > 0 else np.zeros(n, dtype=np.float32)
        f["n_sources"] = np.array([len(s) for s in cands.sources], dtype=np.float32)
        cand_dom = self.catalog.domain[idx]
        f["is_game"] = cand_dom.astype(np.float32)

        pos = history.positives()
        neg_idx = history.idx[history.weight < 0]
        hist_dom = self.catalog.domain[pos.idx]
        n_pos = max(len(pos), 1)
        same = np.array([(hist_dom == d).sum() for d in (0, 1)], dtype=np.float32)
        f["n_hist"] = np.full(n, len(history), dtype=np.float32)
        f["n_hist_same"] = same[cand_dom]
        f["n_hist_other"] = same[1 - cand_dom]
        f["cross_domain_frac"] = f["n_hist_other"] / n_pos
        f["n_dislikes"] = np.full(n, len(neg_idx), dtype=np.float32)
        f["log_pop"] = self.log_pop[idx]
        f["bayes_rating"] = self.bayes_rating[idx]
        f["log_rating_count"] = self.log_count[idx]

        # Theme profile of the user (or explicit cold-start preferences).
        prof: Counter[str] = Counter()
        for i in pos.idx:
            prof.update(self.themes[int(i)][:4])
        for t in preferences or []:
            prof[t] += 3
        total = sum(prof.values()) or 1
        neg_themes: set[str] = set()
        for i in neg_idx:
            neg_themes.update(self.themes[int(i)][:3])
        top = {t for t, _ in prof.most_common(6)}
        aff, jac, dis = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
        for row, i in enumerate(idx):
            ts = set(self.themes[int(i)])
            if ts:
                aff[row] = sum(prof[t] for t in ts) / total
                jac[row] = len(ts & top) / len(ts | top) if top else 0.0
                dis[row] = len(ts & neg_themes) / len(ts)
        f["theme_affinity"], f["theme_jaccard"], f["disliked_theme_overlap"] = aff, jac, dis

        E = self.content.embeddings_
        if len(pos):
            S = E[idx] @ E[pos.idx].T
            f["max_sim_pos"] = S.max(axis=1)
            f["mean_sim_pos"] = S.mean(axis=1)
            co = np.asarray(self.copref.counts_[pos.idx][:, idx].todense())
            f["max_colikes"] = co.max(axis=0)
            f["log_sum_colikes"] = np.log1p(co.sum(axis=0))
            hy = self.year[pos.idx]
            med = np.nanmedian(hy) if np.isfinite(hy).any() else np.nan
            f["year_gap"] = np.abs(self.year[idx] - med)
            creators = {c for c in self.creator[pos.idx] if c}
            f["creator_match"] = np.array([c in creators for c in self.creator[idx]], dtype=np.float32)
        else:
            for k in ("max_sim_pos", "mean_sim_pos", "max_colikes", "log_sum_colikes", "creator_match"):
                f[k] = np.zeros(n, np.float32)
            f["year_gap"] = np.full(n, np.nan, np.float32)
        f["max_sim_neg"] = (E[idx] @ E[neg_idx].T).max(axis=1) if len(neg_idx) else np.zeros(n, np.float32)
        f["cold_start"] = np.full(n, float(len(history) == 0), dtype=np.float32)
        return pd.DataFrame(f, columns=self.feature_names).astype(np.float32)
