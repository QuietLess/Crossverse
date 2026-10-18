"""New releases: titles the catalog doesn't have (data/new_releases.py), served by story match only.

Nobody in the review data has rated them, so the trained ranker never sees them. The engine carries a
small side index instead:

* sentence embeddings from the same model as the semantic retriever, so a profile of catalog items
  can be compared with them ("✨ New releases" row, by story and setting);
* franchise links to catalog items and to each other (serving/universe.py), for the same-universe row
  (Super Mario Odyssey -> The Super Mario Bros. Movie, 2023);
* stand-ins: a new title picked as a favourite is replaced, for the ranked list, by its closest catalog
  titles of the same medium (Marvel Rivals -> Overwatch), preferring its own franchise.

The index can also live outside the model: `refresh` (run on a schedule, `crossverse refresh-new-releases`)
fetches the latest titles and `publish`es a new version to a folder; the service reloads it
(CROSSVERSE_NEW_RELEASES_DIR), so new releases stay current without retraining or redeploying. A version is
plain data (Parquet, .npy, JSON), never a pickle, and `CURRENT` names the live one.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crossverse.retrieval.semantic import item_text
from crossverse.serving import universe

log = logging.getLogger(__name__)

STAND_INS = 2  # catalog titles standing in for one picked new title
STAND_IN_RATING = 4.0  # a little below a real favourite (5)
KIDS_GENRES = {"Kids", "Family"}  # TMDB genres; kept only for profiles with family titles
KIDS_PENALTY = 0.08  # cosine similarity
KEEP_VERSIONS = 3  # published index versions kept (the live one and two to roll back to)
UNFINISHED_AFTER = 2 * 3600  # seconds: a version folder without graph.json this old is a dead run's (job timeout 1 h)
MAX_SHRINK = 0.5  # refuse to publish an index under half the size of the live one (an outage, an expired key)
_VERSION = re.compile(r"^\d{8}T\d{6}-\d{2}$")


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
        full = universe.norm(query)
        q = list(dict.fromkeys(full.split()))[:12]  # bounded work per query, as the catalog search
        if not q:
            return []
        dom, votes = self.items["domain"].to_numpy(), self.items["votes"].to_numpy()
        hits = []
        for n, title in enumerate(self._norm):
            if domain is not None and dom[n] != domain:
                continue
            words = title.split()
            if all(any(w.startswith(t) for w in words) for t in q):
                hits.append((title == full or title.removeprefix("the ") == full, n))
        hits.sort(key=lambda h: (not h[0], -int(votes[h[1]])))
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


# ---------------------------------------------------------------------------------------------
# Published index: a folder of versions, refreshed on a schedule, reloaded by the service
# ---------------------------------------------------------------------------------------------
def current_version(root: Path) -> str | None:
    """The live version named by root/CURRENT, if it is a well-formed name of an existing version."""
    try:
        version = (Path(root) / "CURRENT").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return version if _VERSION.match(version) and (Path(root) / version).is_dir() else None


def _fits(graph: dict[str, Any], n_new: int, dim: int, engine: Any) -> str:
    """Why an index can't serve with `engine` ('' when it can). Links and stand-ins are row numbers in the
    catalog it was built on, so it must come from the same model, and every number must be in range."""
    model = graph.get("model") or {}
    n_cat = len(engine.catalog)
    if model.get("version") != engine.version or model.get("catalog") != n_cat:
        return f"built for model {model.get('version')} ({model.get('catalog')} items), serving {engine.version}"
    sem = getattr(engine, "retrievers", {}).get("semantic")
    if sem is not None and getattr(sem, "embeddings_", None) is not None and sem.embeddings_.shape[1] != dim:
        return f"embedding width {dim}, model {sem.embeddings_.shape[1]}"
    total = n_cat + n_new
    links_ok = all(0 <= int(k) < total and all(0 <= j < total for j in v) for k, v in graph["links"].items())
    stand_ok = all(0 <= int(k) < n_new and all(0 <= j < n_cat for j in v) for k, v in graph["stand_ins"].items())
    return "" if links_ok and stand_ok else "row numbers out of range"


def load_published(root: Path, engine: Any = None) -> tuple[str | None, NewReleaseIndex | None]:
    """(version, index) of the live version, or (None, None) when nothing is published. With `engine`,
    also (None, None) when the index wasn't built for that model."""
    version = current_version(root)
    if version is None:
        return None, None
    folder = Path(root) / version
    items = pd.read_parquet(folder / "items.parquet")
    for col in ("genres", "themes", "keywords"):
        items[col] = items[col].map(list)
    embeddings = np.load(folder / "embeddings.npy", allow_pickle=False)
    graph = json.loads((folder / "graph.json").read_text(encoding="utf-8"))
    if len(embeddings) != len(items) or embeddings.ndim != 2:
        raise ValueError(f"new releases {version}: {len(items)} titles but embeddings {embeddings.shape}")
    if engine is not None:
        reason = _fits(graph, len(items), int(embeddings.shape[1]), engine)
        if reason:
            log.warning("new releases %s not used: %s", version, reason)
            return None, None
    return version, NewReleaseIndex(items, embeddings,
                                    {int(k): [int(j) for j in v] for k, v in graph["links"].items()},
                                    {int(k): [int(j) for j in v] for k, v in graph["stand_ins"].items()},
                                    str(graph.get("fetched_at", "")))


def publish(index: NewReleaseIndex, root: Path, force: bool = False, engine: Any = None) -> str:
    """Write `index` as a new version under root and make it live; returns the version. The files are
    written before CURRENT points at them, so a reader never sees half a version. `engine` (the model the
    index was built with) is recorded; the service only loads an index built for its own model."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        _, live = load_published(root)
    except Exception as e:  # a broken live version must not block its replacement
        log.warning("live new-release version unreadable (%s); publishing over it", e)
        live = None
    if live is not None and not force and len(index) < MAX_SHRINK * len(live):
        raise ValueError(f"new releases shrank from {len(live)} to {len(index)} titles: not published "
                         "(check the API keys and sources, or pass force=True)")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    n = 0
    while (root / f"{stamp}-{n:02d}").exists():
        n += 1
    version = f"{stamp}-{n:02d}"
    folder = root / version
    folder.mkdir()
    index.items.to_parquet(folder / "items.parquet", index=False)
    np.save(folder / "embeddings.npy", np.ascontiguousarray(index.embeddings, dtype=np.float32), allow_pickle=False)
    graph: dict[str, Any] = {"links": {str(k): v for k, v in index.links.items()},
                             "stand_ins": {str(k): v for k, v in index.stand_ins.items()},
                             "fetched_at": index.fetched_at}
    if engine is not None:
        graph["model"] = {"version": engine.version, "catalog": len(engine.catalog),
                          "dim": int(index.embeddings.shape[1])}
    (folder / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    pointer = root / "CURRENT.tmp"
    pointer.write_text(version, encoding="utf-8")
    pointer.replace(root / "CURRENT")
    # Keep the newest complete versions (graph.json is written last); drop older ones, and unfinished ones
    # left by a dead run (not one another execution is still writing).
    versions = sorted(p for p in root.iterdir() if p.is_dir() and _VERSION.match(p.name) and p.name != version)
    complete = [p for p in versions if (p / "graph.json").exists()]
    dead = [p for p in versions if p not in complete and time.time() - p.stat().st_mtime > UNFINISHED_AFTER]
    for old in dead + complete[:max(len(complete) - KEEP_VERSIONS + 1, 0)]:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    return version


def refresh(engine: Any, clients: Any, root: Path, since: str | None = None, until: str | None = None,
            matches: Any = None, embed: Any = None, force: bool = False) -> tuple[str, dict[str, int]]:
    """Fetch the latest titles (data/new_releases.fetch_rows), drop what the catalog has, embed them with
    the engine's sentence model and publish. Returns (version, titles per kind)."""
    from crossverse.data import new_releases as data

    rows = data.fetch_rows(clients, since or data.SINCE, until)
    items = engine.catalog.items
    if "ext_id" in items:
        ext = items["ext_id"].fillna(-1).astype(int)
        known, unmatched = items.loc[ext >= 0, ["ext_source", "ext_id"]], items[ext < 0]
    else:
        known, unmatched = pd.DataFrame({"ext_source": [], "ext_id": []}), items
    titles = {str(d): {universe.norm(t) for t in g["title"]} for d, g in unmatched.groupby("domain")}
    new = data.build(rows, known, titles)
    version = publish(build_index(engine, new, matches, embed=embed), root, force=force, engine=engine)
    return version, {str(k): int(v) for k, v in new["kind"].value_counts().items()}
