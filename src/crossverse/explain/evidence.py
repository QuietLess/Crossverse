"""Deterministic explanation evidence (blueprint section 7).

The ML system chooses the items; this module only *reads* signals that already drove the ranking
(theme overlap, co-like counts among bridge users, embedding similarity, nominating retrievers)
and packages them as an evidence object. Any natural-language layer (template here, optionally an
LLM) may verbalise this evidence but never invent it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from crossverse.retrieval.base import Catalog, History
from crossverse.retrieval.baselines import CoPreferenceRetriever
from crossverse.retrieval.content import ContentRetriever

SOURCE_LABELS = {
    "two_tower": "shared taste embedding",
    "als": "joint collaborative filtering",
    "als_movie": "movie collaborative filtering",
    "als_game": "game collaborative filtering",
    "item_knn": "people who liked the same titles",
    "copref": "cross-domain co-preference",
    "content": "similar themes and description",
    "semantic": "similar story and themes",
    "popularity": "widely liked",
}
# Fallback reasons in plain language (no retriever names in user-facing text).
SOURCE_REASONS = {
    "semantic": "its story and themes are close to what you picked",
    "content": "its description and themes are close to what you picked",
    "two_tower": "fans with a taste profile like yours tend to love it",
    "als": "people with tastes like yours liked it",
    "als_movie": "people with tastes like yours liked it",
    "als_game": "people with tastes like yours liked it",
    "item_knn": "people who liked your titles also liked it",
    "copref": "fans of your titles also liked it",
    "popularity": "it is widely loved",
}


class Explainer:
    SEMANTIC_CLAIM = 0.35  # min. embedding cosine before the text claims "resembles X"
    GENERIC_DF = 0.2  # themes on more than this share of the catalog are too generic to be a reason

    def __init__(self, catalog: Catalog, content: ContentRetriever, copref: CoPreferenceRetriever):
        self.catalog = catalog
        self.content = content
        self.copref = copref
        self.titles = catalog.items["title"].to_numpy()
        self.themes = catalog.items["themes"].tolist()

    @property
    def theme_df(self) -> dict[str, float]:
        """Share of catalog items carrying each theme (lazy, so older pickled bundles still load)."""
        if "_theme_df" not in self.__dict__:
            df: Counter[str] = Counter(t for ts in self.themes for t in set(ts))
            n = max(len(self.themes), 1)
            self.__dict__["_theme_df"] = {t: c / n for t, c in df.items()}
        return self.__dict__["_theme_df"]

    def explain(self, history: History, item: int, sources: list[str], preferences: list[str] | None = None,
                ranker_score: float | None = None) -> dict[str, Any]:
        pos = history.positives()
        neg = history.idx[history.weight < 0]
        profile: Counter[str] = Counter()
        for i in pos.idx:
            profile.update(self.themes[int(i)][:4])
        for t in preferences or []:
            profile[t] += 3
        cand_themes = self.themes[item]
        # Rarest shared themes first: "cyberpunk" says more than "adventure".
        shared = sorted((t for t in cand_themes if t in profile), key=lambda t: self.theme_df.get(t, 0.0))
        top_profile = {t for t, _ in profile.most_common(6)}
        cset = set(cand_themes)
        genre_overlap = len(cset & top_profile) / len(cset | top_profile) if cset and top_profile else 0.0

        anchors: list[dict[str, Any]] = []
        semantic = 0.0
        colikes_best = 0
        if len(pos):
            sims = self.content.similarity(item, pos.idx)
            co = self.copref.co_likes(pos.idx, item)
            semantic = float(sims.max())
            colikes_best = int(co.max()) if len(co) else 0
            # Anchor = history item that best "explains" the candidate: behavioural co-likes first,
            # semantic similarity as tie-breaker.
            strength = np.log1p(co) + 2.0 * sims
            for j in np.argsort(-strength)[:3]:
                anchors.append(
                    {
                        "item_id": str(self.catalog.item_ids[pos.idx[j]]),
                        "title": str(self.titles[pos.idx[j]]),
                        "users_who_liked_both": int(co[j]),
                        "semantic_similarity": round(float(sims[j]), 3),
                    }
                )
        disliked_overlap = []
        if len(neg):
            nt: set[str] = set()
            for i in neg:
                nt.update(self.themes[int(i)][:3])
            disliked_overlap = sorted(cset & nt)

        if (colikes_best >= 20 or semantic >= 0.6) and shared:
            confidence = "high"
        elif colikes_best >= 5 or semantic >= 0.4 or shared:
            confidence = "medium"
        else:
            confidence = "low"
        evidence = {
            "shared_themes": shared[:5],
            "genre_overlap": round(genre_overlap, 3),
            "users_who_liked_both": colikes_best,
            "semantic_similarity": round(semantic, 3),
            "anchors": anchors,
            "candidate_sources": sources,
            "disliked_theme_overlap": disliked_overlap,
            "reason_confidence": confidence,
        }
        if ranker_score is not None:
            evidence["ranker_score"] = round(float(ranker_score), 4)
        evidence["summary"] = self.verbalize(item, evidence, preferences)
        return evidence

    def verbalize(self, item: int, ev: dict[str, Any], preferences: list[str] | None = None) -> str:
        """Template verbalisation of the evidence (no free generation)."""
        title = self.titles[item]
        parts = []
        if ev["anchors"]:
            # Only claim what clears a threshold; weak evidence is not dressed up as a reason.
            behavioural = max(ev["anchors"], key=lambda a: a["users_who_liked_both"])
            semantic = max(ev["anchors"], key=lambda a: a["semantic_similarity"])
            if behavioural["users_who_liked_both"] >= 3:
                parts.append(f"{behavioural['users_who_liked_both']} people who liked {behavioural['title']} also liked {title}")
            if semantic["semantic_similarity"] >= self.SEMANTIC_CLAIM and (semantic is not behavioural or not parts):
                parts.append(f"its story and setting resemble {semantic['title']}")
        # Near-universal themes (on > GENERIC_DF of the catalog) are kept in the evidence object but
        # not offered as a reason on their own.
        specific = [t for t in ev["shared_themes"] if self.theme_df.get(t, 0.0) <= self.GENERIC_DF]
        if specific:
            parts.append("it shares your taste for " + ", ".join(specific[:3]))
        elif preferences:
            parts.append("it matches the preferences you picked")
        if not parts:  # no single strong signal: say in plain words which kind of evidence found it
            reasons = list(dict.fromkeys(SOURCE_REASONS[s] for s in ev["candidate_sources"] if s in SOURCE_REASONS))
            parts.append(" and ".join(reasons[:2]) if reasons else "it fits the overall mix of what you picked")
        text = "Recommended because " + "; ".join(parts) + "."
        if ev["disliked_theme_overlap"]:
            text += " Note: it overlaps with themes you disliked (" + ", ".join(ev["disliked_theme_overlap"]) + ")."
        return text
