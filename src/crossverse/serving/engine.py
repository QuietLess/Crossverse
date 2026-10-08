"""CrossVerseEngine: the full online recommendation path, shared by API, UI and offline eval.

history -> Stage A candidates (all retrievers) -> features -> LightGBM ranker
        -> diversity / business rules -> evidence-based explanations
"""

from __future__ import annotations

import pickle
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from crossverse.evaluation.protocol import EvalCase
from crossverse.explain.evidence import Explainer
from crossverse.ranking.candidates import CandidateGenerator
from crossverse.ranking.features import FeatureBuilder
from crossverse.ranking.ranker import LightGBMRanker, RoutedRanker, diversify
from crossverse.retrieval.base import Catalog, History, Retriever, top_k
from crossverse.retrieval.baselines import CoPreferenceRetriever, ItemKNNRetriever
from crossverse.retrieval.content import ContentRetriever


@dataclass
class Recommendation:
    recommendation_id: str
    item_id: str
    title: str
    domain: str
    rank: int
    score: float
    sources: list[str]
    themes: list[str]
    year: int | None
    evidence: dict[str, Any]
    image: str | None = None


@dataclass
class EngineResult:
    items: list[Recommendation]
    model_version: str
    latency_ms: float
    candidate_count: int
    source_mix: dict[str, int] = field(default_factory=dict)
    cold_start: bool = False


_NOT_RECOMMENDABLE = re.compile(
    r"\b(membership|subscription|gift\s*card|points?\s*card|cash\s*card|live\s*gold|psn\s*card|eshop\s*card"
    r"|season\s*pass|expansion\s*pass|booster\s*course\s*pass|dlc|add-?on|downloadable\s*content|virtual\s*currency"
    r"|v-?bucks|coins?\s*pack|points?\s*pack"
    # accessories Amazon files under "Games"
    r"|collectible\s*case|case\s+for|controller|headset|charg(?:er|ing)|amiibo|figurine|console\s*bundle"
    r"|fight\s*stick|arcade\s*stick|starship\s*pack|weapon\s*pack"
    # currency with an amount ("Xbox LIVE 1600 Microsoft Points", "500 Halo Credits", "1,200 Robux")
    r"|\d[\d,]*\s+(?:microsoft\s+)?points|\d[\d,]*\s+[\w ]{0,20}?credits|robux|gift\s*code"
    # hardware: consoles by storage size ("60GB System", "Slim 120GB"), systems, accessories
    r"|\d+\s*gb|video\s*game\s*system|audio\s*system|lens\s*cleaner|connector|memory\s*unit|kinect\s*sensor"
    r"|(?:wii|ps\d|base|horizontal|vertical)\s+stand|compatible\s+with|screen\s*protector"
    # toys-to-life figures ("Starter Pack" is the game itself and stays)
    r"|(?:single|triple|mini|battle|trap|vehicle)\s+(?:character\s+|trap\s+)?pack|character\s+pack|character\s*$"
    r"|character\s*\(|power\s*disc|toy\s*figure"
    # prepaid time, vouchers, hardware bundles, add-ons that need the base game, non-game software
    r"|time\s*card|pre-?paid|voucher|gold\s*card|key\s*card|download\s*card|hardware\s*bundle|starter\s*bundle"
    r"|dualshock|guitar\s*bundle|wi-?fi\s*bundle|bundle\s*-\s*(?:black|white|blue|red|electric|neon)"
    r"|expansion|upgrade\s*pack|challenger\s*pack|mcafee|norton|antivirus|total\s*protection"
    r"|stuff(?:\s*pack)?\s*$|game\s*pack\s*$)\b",  # Sims add-ons: "The Sims 4 - Movie Hangout Stuff"
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


_TRAILING_YEAR = re.compile(r"\s*[(\[]\s*((?:18|19|20)\d\d)\s*[)\]]\s*$")
_DOMAIN_PREFIX = re.compile(r"^\s*(movie|film|tv|show|game)\s*:\s*", re.I)
_PREFIX_DOMAIN = {"movie": "movie", "film": "movie", "tv": "movie", "show": "movie", "game": "game"}


_BOOK_REVIEW = re.compile(
    r"\b(?:publishers\s+weekly|booklist|kirkus(?:\s+reviews)?|school\s+library\s+journal|library\s+journal)\b", re.I)
_GUIDE_TITLE = re.compile(r"\b(?:strategy|official)\s+(?:nintendo\s+)?guide\b|\bguide\s+final\s+fantasy\b", re.I)

TASTE_MIN_FANS = 20  # taste mode only promotes titles at least this many training users liked

_COMPILATION = re.compile(
    r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten)[\s-]*(?:film|movie|feature)s?\b|\btrilogy\b|\bquadrilogy\b"
    r"|\banthology\b|\b(?:double|triple)\s+feature\b|\bfavorites\b|\bbox\s*set\b|\bmovie\s+collection\b"
    r"|\bcomplete\s+(?:saga|collection)\b|\bcollection\s*(?:\(|$)|\s/\s.+\s/\s",
    re.I,
)


_TV_SET = re.compile(r"\bseasons?\b|\bseries\b", re.I)


def _percentile(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1] (1 = best), so scores on different scales can be blended."""
    if len(x) < 2:
        return np.ones(len(x))
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / (len(x) - 1)


def _image(row: Any) -> str | None:
    """Cover image URL; catalogs built before images were collected have no column."""
    url = row.get("image") if hasattr(row, "get") else None
    return str(url) if isinstance(url, str) and url else None


def _match_key(title: str) -> str:
    return re.sub(r"^(?:the|a|an) ", "", _norm(title))


def parse_query(text: str) -> tuple[str, str | None, int | None]:
    """Split 'game: Batman Begins (2005)' into (title, domain, year). Domain/year are optional hints."""
    domain = None
    m = _DOMAIN_PREFIX.match(text)
    if m:
        domain = _PREFIX_DOMAIN[m.group(1).lower()]
        text = text[m.end():]
    year = None
    m = _TRAILING_YEAR.search(text)
    if m:
        year = int(m.group(1))
        text = text[: m.start()]
    return text.strip(), domain, year


class CrossVerseEngine:
    def __init__(self, catalog: Catalog, retrievers: dict[str, Retriever], ranker: LightGBMRanker | RoutedRanker | None,
                 per_source: int = 100, diversity_lambda: float = 0.15, version: str = "dev",
                 metadata: dict[str, Any] | None = None):
        self.catalog = catalog
        self.retrievers = retrievers
        self.ranker = ranker
        self.version = version
        self.metadata = metadata or {}
        self.diversity_lambda = diversity_lambda
        content = retrievers["content"]
        copref = retrievers["copref"]
        assert isinstance(content, ContentRetriever) and isinstance(copref, CoPreferenceRetriever)
        self.content = content
        self.copref = copref
        self.generator = CandidateGenerator(catalog, retrievers, per_source=per_source)
        pop = retrievers["popularity"].scores_  # type: ignore[attr-defined]
        self.popularity = np.expm1(pop)
        self.features = FeatureBuilder(catalog, content, copref, self.popularity, list(retrievers))
        self.explainer = Explainer(catalog, content, copref)
        items = catalog.items
        self._titles = items["title"].to_numpy()
        self._norm_titles = np.array([_norm(t) for t in self._titles])
        self._creators = items.get("creator", items["title"].map(lambda _: "")).fillna("").astype(str).to_numpy()

    # ------------------------------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------------------------------
    def ineligible(self) -> np.ndarray:
        """Catalog items that are never recommended: memberships, gift/points cards and add-on
        content that is useless without a base game (DLC, season passes). Lazy for old pickles."""
        if "_ineligible" not in self.__dict__:
            # "Fallout 4 Game + Season Pass" is the game plus an extra: keep it.
            hit = np.array([bool(_NOT_RECOMMENDABLE.search(t)) and " + " not in t for t in self._titles])
            # Books Amazon filed under games ("Still Foolin' 'Em", "Wonder"): no IGDB match and an editorial
            # review from a book trade journal, or a strategy guide.
            items = self.catalog.items
            if "ext_id" in items and "text" in items:
                unmatched = items["ext_id"].to_numpy() < 0
                bookish = items["text"].fillna("").str.contains(_BOOK_REVIEW).to_numpy()
                guide = items["title"].str.contains(_GUIDE_TITLE).to_numpy()
                hit |= unmatched & (bookish | guide)
            self.__dict__["_ineligible"] = np.flatnonzero(hit & (self.catalog.domain == 1))
        return self.__dict__["_ineligible"]

    def compilations(self) -> np.ndarray:
        """Movie box sets and multi-film packs ("Complete 8-Film Collection", "Trilogy", "A / B / C").
        Not recommended: they bundle works the user may already know and crowd out single titles."""
        if "_compilations" not in self.__dict__:
            # A TV show's season set *is* the show (seasons are merged into one item): keep it.
            hit = np.array([bool(_COMPILATION.search(t)) and not _TV_SET.search(t) for t in self._titles])
            self.__dict__["_compilations"] = np.flatnonzero(hit & (self.catalog.domain == 0))
        return self.__dict__["_compilations"]

    def quality(self) -> np.ndarray:
        """Items liked by at least TASTE_MIN_FANS users: the pool taste mode may promote."""
        return self.popularity >= TASTE_MIN_FANS

    def rank(self, history: History, target_domain: str | None, k: int, exclude: np.ndarray | None = None,
             preferences: list[str] | None = None, free_text: str = "", use_ranker: bool = True,
             diversify_results: bool = True, taste: float = 0.0, drop_compilations: bool = True):
        """taste in [0, 1]: 0 = the trained ranker ("what similar fans liked"), 1 = description similarity
        to the history among well-liked titles ("similar story & setting"). In between, a blend of the
        two percentile ranks. Only applies to non-empty histories."""
        blocked = self.ineligible()
        if drop_compilations:
            blocked = np.union1d(blocked, self.compilations())
        exclude = blocked if exclude is None else np.union1d(exclude, blocked)
        taste = float(np.clip(taste, 0.0, 1.0)) if len(history) else 0.0
        quality = self.quality() if taste > 0 else None
        cands = self.generator.generate(history, target_domain, exclude, preferences, free_text, quality)
        if self.ranker is not None and use_ranker and len(cands.idx):
            X = self.features.build(history, cands, preferences)
            scores = self.ranker.predict(X)
        else:  # fallback: reciprocal-rank fusion of retrievers
            scores = np.zeros(len(cands.idx))
            for r in cands.ranks.values():
                scores += 1.0 / (60.0 + r)
        if taste > 0 and quality is not None and len(cands.idx):
            # Below the quality floor an item keeps only its behavioural share: similarity alone
            # must not lift a title almost nobody liked.
            similar = cands.scores[self.generator.similarity_source()]
            sim = np.where(quality[cands.idx], _percentile(similar), 0.0)
            scores = (1 - taste) * _percentile(scores) + taste * sim
        if len(history) == 0 and (preferences or free_text) and len(cands.idx):
            # Explicit-intent rule: a cold-start user who asked for "cyberpunk" sees cyberpunk
            # titles first whenever at least k candidates match. (Offline this costs a little
            # NDCG versus pure popularity; see reports/benchmark.md.)
            n_match = self.generator.theme_match(preferences, free_text)[cands.idx]
            if (n_match > 0).sum() >= k:
                # Tiers: items matching more of the requested themes first, ranker order within a tier.
                scores = scores + 1e3 * n_match
        if not diversify_results:
            return cands, scores, np.argsort(-scores)[:k]
        if target_domain in (None, "both", "all"):
            order = self._domain_quota_order(history, cands.idx, scores, k)
        else:
            order = diversify(cands.idx, scores, self.content.embeddings_, k, self.diversity_lambda, self._creators)
        return cands, scores, order

    def _domain_quota_order(self, history: History, idx: np.ndarray, scores: np.ndarray, k: int,
                            min_share: float = 0.3) -> np.ndarray:
        """Business rule for unified lists: each domain gets a quota from the profile's domain mix
        (at least `min_share`), so one domain's score scale cannot crowd out the other."""
        pos = history.positives()
        game_share = float(self.catalog.domain[pos.idx].mean()) if len(pos) else 0.5
        game_share = min(max(game_share, min_share), 1 - min_share)
        quota = {1: int(round(k * game_share))}
        quota[0] = k - quota[1]
        parts = []
        for d in (0, 1):
            sel = np.flatnonzero(self.catalog.domain[idx] == d)
            if len(sel) == 0:
                continue
            o = diversify(idx[sel], scores[sel], self.content.embeddings_, quota[d], self.diversity_lambda,
                          self._creators)
            parts.append(sel[o])
        merged = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        if len(merged) < k:  # one domain ran short: back-fill from the best remaining candidates
            rest = [j for j in np.argsort(-scores) if j not in set(merged.tolist())]
            merged = np.concatenate([merged, np.asarray(rest[: k - len(merged)], dtype=np.int64)])
        return merged[np.argsort(-scores[merged], kind="stable")]

    def recommend_case(self, case: EvalCase, target_domain: str | None, k: int, **kw) -> np.ndarray:
        """Offline-evaluation adapter (returns catalog indices)."""
        cands, scores, order = self.rank(case.history, target_domain, k, case.exclude, case.preferences, **kw)
        return cands.idx[order]

    # ------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------
    def build_history(self, liked: list[tuple[str, float]], disliked: list[str] | None = None) -> History:
        ids = [i for i, _ in liked] + list(disliked or [])
        ratings = [r for _, r in liked] + [1.0] * len(disliked or [])
        return History.from_pairs(self.catalog, ids, ratings)

    def recommend(self, liked: list[tuple[str, float]], disliked: list[str] | None = None,
                  target_domain: str | None = None, k: int = 10, preferences: list[str] | None = None,
                  free_text: str = "", exclude_ids: list[str] | None = None, explain: bool = True,
                  diversify_results: bool = True, taste: float = 0.0) -> EngineResult:
        t0 = time.perf_counter()
        history = self.build_history(liked, disliked)
        exclude = np.unique(np.concatenate([history.idx, self.catalog.idx(exclude_ids or [])]))
        cold = len(history) == 0
        cands, scores, order = self.rank(history, target_domain, k, exclude, preferences, free_text,
                                         diversify_results=diversify_results, taste=taste)
        out = []
        mix: dict[str, int] = {}
        for rank, j in enumerate(order):
            i = int(cands.idx[j])
            row = self.catalog.items.iloc[i]
            srcs = cands.sources[j]
            for s in srcs:
                mix[s] = mix.get(s, 0) + 1
            ev = self.explainer.explain(history, i, srcs, preferences, float(scores[j])) if explain else {}
            year = int(row["year"]) if row["year"] and row["year"] > 0 else None
            out.append(
                Recommendation(
                    recommendation_id=uuid.uuid4().hex[:16], item_id=str(row["item_id"]), title=str(row["title"]),
                    domain=str(row["domain"]), rank=rank + 1, score=float(scores[j]), sources=srcs,
                    themes=list(row["themes"]), year=year, evidence=ev, image=_image(row),
                )
            )
        return EngineResult(out, self.version, (time.perf_counter() - t0) * 1000, len(cands.idx), mix, cold)

    def similar(self, item_id: str, k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Same-domain and cross-domain neighbours of one item (behaviour + content blend)."""
        i = self.catalog.index[item_id]
        h = History(np.array([i]), np.array([1.0], dtype=np.float32))
        sem = self.content.score(h)
        beh = np.zeros_like(sem)
        knn = self.retrievers.get("item_knn")
        if isinstance(knn, ItemKNNRetriever):
            beh = knn.score(h)
        cross = self.copref.score(h)

        def z(x: np.ndarray, m: np.ndarray) -> np.ndarray:
            v = x[m]
            sd = v.std()
            out = np.zeros_like(x)
            out[m] = (v - v.mean()) / sd if sd > 0 else 0.0
            return out

        dom = self.catalog.domain[i]
        same_m = self.catalog.domain == dom
        other_m = ~same_m
        same_score = z(sem, same_m) + z(beh, same_m)
        other_score = z(sem, other_m) + z(cross, other_m) + 0.5 * z(beh, other_m)
        excl = np.array([i])

        def pack(idx: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
            res = []
            for j in idx:
                ev = self.explainer.explain(h, int(j), ["similar"])
                res.append({**self.item_dict(int(j)), "score": float(score[j]), "evidence": ev})
            return res

        return {
            "item": [self.item_dict(i)],
            "same_domain": pack(top_k(same_score, k, same_m, excl), same_score),
            "cross_domain": pack(top_k(other_score, k, other_m, excl), other_score),
        }

    def item_dict(self, i: int) -> dict[str, Any]:
        row = self.catalog.items.iloc[i]
        return {
            "item_id": str(row["item_id"]), "title": str(row["title"]), "domain": str(row["domain"]),
            "themes": list(row["themes"]), "year": int(row["year"]) if row["year"] and row["year"] > 0 else None,
            "popularity": int(self.popularity[i]), "image": _image(row),
        }

    @property
    def _match_titles(self) -> np.ndarray:
        """Normalised titles without a trailing '(YYYY)' or leading article, so 'Batman Begins (2005)'
        matches 'Batman Begins' and 'martian' is a prefix of 'The Martian'.
        Built lazily: engines pickled before this existed don't carry it."""
        if "_match" not in self.__dict__:
            self.__dict__["_match"] = np.array([_match_key(_TRAILING_YEAR.sub("", t)) for t in self._titles])
        return self.__dict__["_match"]

    def search(self, query: str, domain: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        title, hinted_domain, year = parse_query(query)
        domain = domain or hinted_domain
        q = _norm(title)
        if not q:
            return []
        index = self._prefix_index()
        sets = [index.get(t[:12], set()) for t in q.split()]
        if any(len(t) > 12 for t in q.split()):  # long tokens: verify the full token on the shortlist
            cand = set.intersection(*sets) if sets else set()
            sets = [{i for i in cand if all(any(w.startswith(t) for w in self._norm_titles[i].split())
                                               for t in q.split())}]
        hits = set.intersection(*sets) if sets else set()
        mask = self.catalog.mask(domain)
        idx = np.array(sorted(i for i in hits if mask[i]), dtype=np.int64)
        if not len(idx):
            return []
        match, key_q = self._match_titles, _match_key(title)
        exact = np.array([match[i] == key_q for i in idx])
        prefix = np.array([match[i].startswith(key_q) for i in idx])
        years = self.catalog.items["year"].to_numpy()
        same_year = (years[idx] == year) if year else np.zeros(len(idx), dtype=bool)
        key = exact * 1e10 + same_year * 1e9 + prefix * 1e6 + self.popularity[idx]
        return [self.item_dict(int(i)) for i in idx[np.argsort(-key)][:limit]]

    def _prefix_index(self) -> dict[str, set[int]]:
        """Title-token prefix -> item indices (type-ahead search). Built lazily on first use."""
        if "_prefix" not in self.__dict__:
            index: dict[str, set[int]] = {}
            for i, title in enumerate(self._norm_titles):
                for word in title.split():
                    for n in range(1, min(len(word), 12) + 1):
                        index.setdefault(word[:n], set()).add(i)
            self.__dict__["_prefix"] = index
        return self.__dict__["_prefix"]

    def suggest(self, query: str, domain: str | None = None, n: int = 3) -> list[dict[str, Any]]:
        """'Did you mean' for unresolved titles: candidates sharing a token prefix, ranked by similarity."""
        from difflib import SequenceMatcher

        q = _norm(query)
        index = self._prefix_index()
        cand: set[int] = set()
        for tok in q.split():
            if len(tok) >= 3:
                cand |= index.get(tok[:4] if len(tok) >= 4 else tok, set())
        mask = self.catalog.mask(domain)
        scored = [(SequenceMatcher(None, q, self._norm_titles[i]).ratio(), self.popularity[i], i)
                  for i in cand if mask[i]]
        scored.sort(reverse=True)
        return [self.item_dict(i) for s, _, i in scored[:n] if s >= 0.5]

    def resolve(self, name_or_id: str, domain: str | None = None) -> str | None:
        return self.resolve_detail(name_or_id, domain)[0]

    def resolve_detail(self, name_or_id: str, domain: str | None = None) -> tuple[str | None, list[dict[str, Any]]]:
        """Resolve a title or id. Also returns exact-title matches in the *other* domain, so callers can
        tell the user 'Batman Begins' could also mean the game (pass 'game: Batman Begins' to pick it)."""
        if name_or_id in self.catalog.index:
            return name_or_id, []
        hits = self.search(name_or_id, domain, 10)
        if not hits:
            return None, []
        best = hits[0]
        title, hinted_domain, _ = parse_query(name_or_id)
        if domain or hinted_domain:
            return best["item_id"], []
        q, match = _match_key(title), self._match_titles
        alternatives = [h for h in hits[1:] if h["domain"] != best["domain"]
                        and match[self.catalog.index[h["item_id"]]] == q]
        return best["item_id"], alternatives[:3]

    # ------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def refresh_display_titles(self) -> None:
        """Re-clean titles with the current display_title(), so bundles trained before a cleaner
        improvement show today's titles. Display and search only: ids, features and scores are unchanged."""
        from crossverse.data.canonical import display_titles

        items = self.catalog.items
        items["title"] = display_titles(items)
        self._titles = items["title"].to_numpy()
        self._norm_titles = np.array([_norm(t) for t in self._titles])
        self.explainer.titles = self._titles
        for lazy in ("_prefix", "_match", "_ineligible", "_compilations"):  # rebuilt with current rules
            self.__dict__.pop(lazy, None)

    @staticmethod
    def export_portable(src: Path, dst: Path) -> None:
        """Re-save an engine so it loads on any OS: a pickled pathlib.WindowsPath (or PosixPath) cannot be
        unpickled on the other platform, which crashed the Linux container on Cloud Run."""
        engine = CrossVerseEngine.load(src)  # current __getstate__ hooks drop machine-specific state
        stray = _find_paths(engine)
        if stray:
            raise ValueError(f"engine still holds filesystem paths: {stray}")
        engine.save(dst)

    @staticmethod
    def load(path: Path) -> CrossVerseEngine:
        with open(path, "rb") as fh:
            engine: CrossVerseEngine = pickle.load(fh)
        engine.refresh_display_titles()
        return engine


def _find_paths(obj: Any, where: str = "engine", depth: int = 0, seen: set[int] | None = None) -> list[str]:
    """Attribute paths under `obj` that hold a pathlib path (they make a pickle OS-specific)."""
    import pathlib

    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 8:
        return []
    seen.add(id(obj))
    if isinstance(obj, pathlib.PurePath):
        return [where]
    if isinstance(obj, dict):
        items: Any = obj.items()
    elif isinstance(obj, list | tuple):
        items = enumerate(obj)
    elif type(obj).__module__.startswith("crossverse") and hasattr(obj, "__dict__"):
        state = obj.__getstate__() if hasattr(type(obj), "__getstate__") and type(obj).__getstate__ is not object.__getstate__ else vars(obj)
        items = state.items()
    else:
        return []
    return [p for k, v in items for p in _find_paths(v, f"{where}.{k}", depth + 1, seen)]
