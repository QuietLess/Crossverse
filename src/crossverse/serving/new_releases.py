"""New releases: titles the catalog doesn't have (data/new_releases.py), served by story match only.

Nobody in the review data has rated them, so the trained ranker never sees them. The engine carries a
small side index instead:

* sentence embeddings from the same model as the semantic retriever, so a profile of catalog items
  can be compared with them ("✨ New releases" row, by story and setting);
* franchise links to catalog items and to each other (serving/universe.py), for the same-universe row
  (Super Mario Odyssey -> The Super Mario Bros. Movie, 2023);
* stand-ins: a new title picked as a favourite is replaced, for the ranked list, by its closest catalog
  titles of the same medium (Marvel Rivals -> Overwatch), preferring its own franchise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from crossverse.retrieval.semantic import item_text
from crossverse.serving import universe

STAND_INS = 2  # catalog titles standing in for one picked new title
STAND_IN_RATING = 4.0  # a little below a real favourite (5)
KIDS_GENRES = {"Kids", "Family"}  # TMDB genres; kept only for profiles with family titles
KIDS_PENALTY = 0.08  # cosine similarity


@dataclass
class NewReleaseIndex:
    items: pd.DataFrame  # data/new_releases.COLUMNS
    embeddings: np.ndarray  # (n, dim), unit rows, the engine's semantic model
    links: dict[int, list[int]] = field(default_factory=dict)  # combined index (catalog first) -> combined
    stand_ins: dict[int, list[int]] = field(default_factory=dict)  # new row -> catalog indices
    fetched_at: str = ""

    def __post_init__(self) -> None:
        self.index = {str(i): n for n, i in enumerate(self.items["item_id"])}
        self._norm = [universe.norm(t) for t in self.items["title"]]

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self.index

    def item_dict(self, n: int) -> dict[str, Any]:
        r = self.items.iloc[n]
        return {"item_id": str(r["item_id"]), "title": str(r["title"]), "domain": str(r["domain"]),
                "themes": list(r["themes"]), "year": int(r["year"]) or None, "popularity": 0,
                "image": str(r["image"]) or None, "kind": str(r["kind"]), "new": True}

    def search(self, query: str, domain: str | None = None, limit: int = 10) -> list[tuple[bool, int]]:
        """(exact title, row) for titles containing every query word as a word prefix."""
        q = universe.norm(query).split()
        if not q:
            return []
        dom = self.items["domain"].to_numpy()
        hits = []
        for n, title in enumerate(self._norm):
            words = title.split()
            if (domain is None or dom[n] == domain) and all(any(w.startswith(t) for w in words) for t in q):
                hits.append((title == " ".join(q) or title.removeprefix("the ") == " ".join(q), n))
        hits.sort(key=lambda h: (not h[0], -int(self.items["votes"].iloc[h[1]])))
        return hits[:limit]


def build_index(engine: Any, new: pd.DataFrame, matches: pd.DataFrame | None,
                embed: Any = None) -> NewReleaseIndex:
    """Embed the new titles with the engine's semantic model and link them to the catalog.
    `embed` (texts -> unit vectors) defaults to the semantic retriever's own encoder."""
    sem = engine.retrievers.get("semantic")
    if sem is None:
        raise ValueError("new releases need an engine with the semantic retriever (v11+)")
    new = new.reset_index(drop=True)
    texts = [item_text(str(t), list(th), str(x)) for t, th, x in zip(new["title"], new["themes"], new["text"], strict=True)]
    vecs = np.asarray((embed or sem._embed)(texts), dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    # Franchise links over catalog + new titles in one index space (catalog first).
    cat_titles, cat_kinds, cat_coll, cat_kw = universe.catalog_metadata(engine, matches)
    n_cat = len(cat_titles)
    pop = np.concatenate([engine.popularity, np.zeros(len(new))])
    blocked = np.union1d(engine.ineligible(), engine.compilations())
    links = universe.link_items(cat_titles + new["title"].tolist(), cat_kinds + new["kind"].tolist(),
                                cat_coll + new["collection"].fillna("").tolist(),
                                cat_kw + [list(k) for k in new["keywords"]], pop, blocked)
    links = {i: js for i, js in links.items() if i >= n_cat or any(j >= n_cat for j in js)}

    # Stand-ins: closest well-liked catalog titles of the same medium; the same franchise or title root first
    # (Hollow Knight: Silksong -> Hollow Knight). Franchise links only cross media, so match those here.
    quality = engine.quality()
    quality[blocked] = False
    domain = engine.catalog.domain  # 0 movie, 1 game
    family_of: dict[str, set[int]] = {}
    for j, (title, coll) in enumerate(zip(cat_titles, cat_coll, strict=True)):
        for key in {universe.title_root(title), *universe.franchise_names(coll)} - {""}:
            family_of.setdefault(key, set()).add(j)
    allowed = np.isin(np.arange(n_cat), blocked, invert=True)
    stand_ins = {}
    for n in range(len(new)):
        same = quality & (domain == (1 if new["domain"].iloc[n] == "game" else 0))
        sims = np.where(same, sem.embeddings_ @ vecs[n], -np.inf)
        keys = {universe.title_root(str(new["title"].iloc[n])), *universe.franchise_names(str(new["collection"].iloc[n]))}
        # (the family needn't clear the quality floor: Hollow Knight is the right stand-in however few rated it)
        kin = allowed & (domain == (1 if new["domain"].iloc[n] == "game" else 0))
        family = sorted({j for k in keys - {""} for j in family_of.get(k, ()) if kin[j]},
                        key=lambda j: -float(sem.embeddings_[j] @ vecs[n]))
        order = family + [int(j) for j in np.argsort(-sims)[: STAND_INS * 4] if j not in family]
        stand_ins[n] = [int(j) for j in order if j in family or np.isfinite(sims[j])][:STAND_INS]
    return NewReleaseIndex(new, vecs, links, stand_ins, str(new.attrs.get("fetched_at", "")))


def attach(engine: Any, path: Any, metadata_path: Any) -> int:
    """Build the index from new_releases.parquet and attach it; returns its size (0 when there is no file
    or the engine has no sentence model)."""
    from pathlib import Path

    from crossverse.data.external import load_matches

    if not Path(path).exists() or "semantic" not in engine.retrievers:
        return 0
    new = pd.read_parquet(path)
    new["themes"] = new["themes"].map(list)
    engine.attach_new_releases(build_index(engine, new, load_matches(metadata_path)))
    return len(new)
