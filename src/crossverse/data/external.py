"""External metadata: match catalog items to TMDB (movies, TV) and IGDB (games).

Amazon descriptions are product copy ("Includes bonus disc"), genres are store shelves and movie
years are mostly DVD release dates. TMDB/IGDB give the work itself: overview, genres, keywords,
release year and a stable id. Matching is title-based with guards:

* titles are compared after `normalize_title` (format/edition noise removed); a candidate must reach
  `MIN_SIMILARITY`;
* a year in the Amazon title ("Carrie (2013)") must agree within a year; a movie can't premiere after
  its earliest Amazon product (the DVD), and a game's release year must be within 2 years;
* ties go to the more popular candidate (TMDB vote count, IGDB rating count).

Every HTTP response is cached in data/external/*.jsonl, so a run is resumable and repeatable without
the network. Keys come from .env (TMDB_TOKEN, IGDB_CLIENT_ID, IGDB_CLIENT_SECRET); never logged.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from crossverse.data.canonical import _TOKENS, display_title, normalize_title

log = logging.getLogger(__name__)

MIN_SIMILARITY = 0.88
TITLE_YEAR = re.compile(r"\((\d{4})\)\s*$")


# ---------------------------------------------------------------------------------------------
# HTTP with an on-disk cache and a shared rate limit
# ---------------------------------------------------------------------------------------------
class ResponseCache:
    """Append-only JSON-lines cache: key -> response body. Thread-safe."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # a line cut off by an interrupted run
                        continue
                    self.data[row["k"]] = row["v"]
        path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.interval
        if delay > 0:
            time.sleep(delay)


def _request(method: str, url: str, limiter: RateLimiter, attempts: int = 6, **kw: Any) -> Any:
    for attempt in range(attempts):
        limiter.wait()
        try:
            r = requests.request(method, url, timeout=30, **kw)
        except requests.RequestException as e:
            log.warning("%s: %s (attempt %d)", url.split("?")[0], type(e).__name__, attempt + 1)
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 2**attempt)))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{url.split('?')[0]}: no answer after {attempts} attempts")


class TMDB:
    BASE = "https://api.themoviedb.org/3"

    def __init__(self, token: str, cache_dir: Path, per_second: float = 30):
        self.headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
        self.cache = ResponseCache(cache_dir / "tmdb.jsonl")
        self.limiter = RateLimiter(per_second)

    def get(self, path: str, **params: Any) -> Any:
        key = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        if key in self.cache:
            return self.cache.get(key)
        body = _request("GET", f"{self.BASE}/{path}", self.limiter, headers=self.headers, params=params)
        self.cache.put(key, body)
        return body

    def search(self, kind: str, query: str) -> list[dict[str, Any]]:
        body = self.get(f"search/{kind}", query=query, include_adult="false")
        return (body or {}).get("results", [])[:8]

    def details(self, kind: str, tmdb_id: int) -> dict[str, Any] | None:
        return self.get(f"{kind}/{tmdb_id}", append_to_response="keywords")


class IGDB:
    BASE = "https://api.igdb.com/v4"
    FIELDS = ("name, first_release_date, total_rating_count, genres.name, themes.name, keywords.name, "
              "summary, storyline, franchises.name, collection.name, player_perspectives.name")

    def __init__(self, client_id: str, client_secret: str, cache_dir: Path, per_second: float = 3.5):
        self.client_id, self.client_secret = client_id, client_secret
        self.cache = ResponseCache(cache_dir / "igdb.jsonl")
        self.limiter = RateLimiter(per_second)
        self._token: str | None = None

    def _auth(self) -> dict[str, str]:
        if self._token is None:
            body = _request("POST", "https://id.twitch.tv/oauth2/token", self.limiter, params={
                "client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "client_credentials"})
            self._token = body["access_token"]
        return {"Client-ID": self.client_id, "Authorization": f"Bearer {self._token}"}

    def search(self, query: str) -> list[dict[str, Any]]:
        """One title search. IGDB allows ~4 requests/second; `search` does not work inside multiquery
        (it returns empty results), so titles are searched one by one, in parallel up to the limit.
        No server-side type filter: IGDB renamed `category` to `game_type`, and DLC / bundles carry
        different names, so the title-similarity guard already rejects them."""
        key = f"search:{query}"
        if key in self.cache:
            return self.cache.get(key) or []
        body = f'search "{_igdb_escape(query)}"; fields {self.FIELDS}; limit 20;'
        res = _request("POST", f"{self.BASE}/games", self.limiter, headers=self._auth(), data=body) or []
        self.cache.put(key, res)
        return res

    def by_name(self, query: str) -> list[dict[str, Any]]:
        """Games whose name equals the query (case-insensitive), e.g. every "Tomb Raider" release."""
        key = f"name:{query}"
        if key in self.cache:
            return self.cache.get(key) or []
        body = f'fields {self.FIELDS}; where name ~ "{_igdb_escape(query)}"; limit 20;'
        res = _request("POST", f"{self.BASE}/games", self.limiter, headers=self._auth(), data=body) or []
        self.cache.put(key, res)
        return res


def _igdb_escape(q: str) -> str:
    return q.replace("\\", " ").replace('"', " ")


# ---------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------
@dataclass
class Match:
    item_id: str
    domain: str
    source: str  # "tmdb" | "igdb"
    kind: str  # "movie" | "tv" | "game"
    ext_id: int
    ext_title: str
    ext_year: int | None
    similarity: float
    genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    overview: str = ""
    collection: str = ""


STUDIO_PREFIX = re.compile(r"^(?:marvel|disney|pixar|dreamworks|walt\s+disney)'?s\s+", re.I)
BRACKETS = re.compile(r"\s*[(\[][^)\]]*[)\]]?")
YEAR_BONUS = 0.15  # candidate released within a year of the catalog's year: strong evidence
FORMAT_BONUS = 0.2  # TMDB kind (movie / tv) agrees with what the Amazon products look like


def similarity(a: str, b: str, domain: str) -> float:
    """Title similarity after normalisation; a candidate's subtitle also counts ("Skyrim VR" vs
    "The Elder Scrolls V: Skyrim VR"), slightly below a full-title match."""
    na = normalize_title(STUDIO_PREFIX.sub("", a), domain)
    best = 0.0
    for full, cap in ((b, 1.0), (b.rsplit(":", 1)[-1] if ":" in b else "", 0.95)):
        nb = normalize_title(STUDIO_PREFIX.sub("", full), domain) if full else ""
        if na and nb:
            best = max(best, cap if na == nb else cap * SequenceMatcher(None, na, nb).ratio())
    return best


def _only_noise(text: str, domain: str) -> bool:
    """True when nothing but format/platform/edition words remain (normalize_title would fall back to
    the original text in that case, to protect films literally named "Unrated")."""
    return not re.sub(r"[^a-z0-9]+", "", _TOKENS[domain].sub(" ", text.lower()))


def search_title(title: str, domain: str = "movie") -> tuple[str, int | None]:
    """Catalog title -> (search query, year stated in the title). Re-cleans the title (catalogs built
    before a display_title improvement still carry noise), drops bracketed text, studio prefixes
    ("Marvel's") and a " - <platform/edition>" tail that normalises to nothing."""
    title = display_title(title, domain)
    m = TITLE_YEAR.search(title)
    year = int(m.group(1)) if m else None
    q = STUDIO_PREFIX.sub("", BRACKETS.sub(" ", title))
    if " - " in q:
        head, tail = q.split(" - ", 1)
        if head.strip() and _only_noise(tail, domain):  # "Halo 4 - Xbox 360": the tail is noise
            q = head
    return " ".join(q.split()), year


def _year(date: str | int | None) -> int | None:
    if isinstance(date, int):  # IGDB: unix seconds
        return time.gmtime(date).tm_year
    return int(date[:4]) if date and len(date) >= 4 and date[:4].isdigit() else None


def _key(sim: float, votes: int | None, year: int | None, catalog_year: int | None) -> float:
    near_year = YEAR_BONUS if year and catalog_year and abs(year - catalog_year) <= 1 else 0.0
    return sim + near_year + 0.03 * math.log1p(votes or 0)


def pick_movie(title: str, catalog_year: int | None, candidates: list[tuple[str, dict[str, Any]]],
               tv_hint: bool | None = None) -> tuple[str, dict, float] | None:
    """Best (kind, candidate, similarity) among TMDB movie/tv results, or None. Amazon movie years are
    unreliable (DVD dates, sometimes junk), so they only add a bonus on an exact match; a year written
    in the title is a hard constraint."""
    query, title_year = search_title(title)
    best, best_key = None, -math.inf
    for kind, c in candidates:
        names = [n for n in (c.get("title") or c.get("name"), c.get("original_title") or c.get("original_name")) if n]
        sim = max((similarity(query, n, "movie") for n in names), default=0.0)
        if sim < MIN_SIMILARITY:
            continue
        year = _year(c.get("release_date") or c.get("first_air_date"))
        if title_year and year and abs(year - title_year) > 1:
            continue
        key = _key(sim, c.get("vote_count"), year, catalog_year)
        if tv_hint is not None:
            key += FORMAT_BONUS if (kind == "tv") == tv_hint else 0.0
        if key > best_key:
            best, best_key = (kind, c, sim), key
    return best


def pick_game(title: str, catalog_year: int | None, candidates: list[dict[str, Any]]) -> tuple[dict, float] | None:
    query, title_year = search_title(title, "game")
    best, best_key = None, -math.inf
    for c in candidates:
        sim = similarity(query, c.get("name", ""), "game")
        if sim < MIN_SIMILARITY:
            continue
        cy = _year(c.get("first_release_date"))
        if title_year and cy and abs(cy - title_year) > 1:
            continue
        key = _key(sim, c.get("total_rating_count"), cy, catalog_year)
        if key > best_key:
            best, best_key = (c, sim), key
    return best


def movie_match(tmdb: TMDB, item: dict[str, Any]) -> Match | None:
    query, _ = search_title(item["title"])
    if not query:
        return None
    cands = [("movie", c) for c in tmdb.search("movie", query)] + [("tv", c) for c in tmdb.search("tv", query)]
    catalog_year = item["year"] if item.get("year") and item["year"] > 0 else None
    picked = pick_movie(item["title"], catalog_year, cands, item.get("tv_hint"))
    if picked is None:
        return None
    kind, c, sim = picked
    d = tmdb.details(kind, c["id"]) or {}
    kw = d.get("keywords", {})
    return Match(item["item_id"], "movie", "tmdb", kind, int(c["id"]), c.get("title") or c.get("name") or "",
                 _year(c.get("release_date") or c.get("first_air_date")), round(sim, 3),
                 [g["name"] for g in d.get("genres", [])],
                 [k["name"] for k in (kw.get("keywords") or kw.get("results") or [])][:30],
                 (d.get("overview") or c.get("overview") or "")[:2000],
                 (d.get("belongs_to_collection") or {}).get("name", "") if kind == "movie" else "")


def game_match(igdb: IGDB, item: dict[str, Any]) -> Match | None:
    query, _ = search_title(item["title"], "game")
    if not query:
        return None
    catalog_year = item["year"] if item.get("year") and item["year"] > 0 else None
    picked = pick_game(item["title"], catalog_year, igdb.search(query))
    if picked is None:  # IGDB's relevance ranking can bury the main game under DLC: try the exact name
        picked = pick_game(item["title"], catalog_year, igdb.by_name(query))
    if picked is None:
        return None
    c, sim = picked
    names = lambda key: [x["name"] for x in c.get(key) or [] if "name" in x]  # noqa: E731
    overview = " ".join(x for x in (c.get("summary"), c.get("storyline")) if x)[:2000]
    collection = (c.get("collection") or {}).get("name", "") or ", ".join(names("franchises"))
    return Match(item["item_id"], "game", "igdb", "game", int(c["id"]), c.get("name", ""),
                 _year(c.get("first_release_date")), round(sim, 3), names("genres") + names("themes"),
                 (names("keywords") + names("player_perspectives"))[:30], overview, collection)


def as_row(m: Match) -> dict[str, Any]:
    return asdict(m)


# ---------------------------------------------------------------------------------------------
# Using the matches in the dataset build
# ---------------------------------------------------------------------------------------------
def load_matches(path: Path) -> Any:
    """metadata.parquet from pipelines/enrich.py, or None when it hasn't been run."""
    import pandas as pd

    return pd.read_parquet(path) if path.exists() else None


def merge_map(matches: Any, popularity: dict[str, int], min_similarity: float = 0.95) -> dict[str, str]:
    """Items that matched the same TMDB/IGDB work (same domain, kind and id) -> the most-liked one.
    Only confident matches merge ("Mad Max (2013)" + "WB Games Mad Max"); a near-miss title match
    may enrich an item but never fuses two items."""
    sure = matches[matches["similarity"] >= min_similarity]
    out: dict[str, str] = {}
    for _, g in sure.groupby(["domain", "source", "kind", "ext_id"]):
        if len(g) < 2:
            continue
        ids = sorted(g["item_id"], key=lambda i: (-popularity.get(i, 0), i))
        out.update({i: ids[0] for i in ids[1:]})
    return out


def enrich_catalog(catalog: Any, matches: Any) -> Any:
    """Matched items get TMDB/IGDB genres (instead of Amazon store shelves), the real release year, and
    a text that starts with the overview, genres and keywords (the sentence model reads ~512 tokens)."""
    from crossverse.features.themes import item_themes

    m = matches.drop_duplicates("item_id").set_index("item_id")
    catalog = catalog.copy()
    for col in ("ext_source", "ext_kind"):
        catalog[col] = ""
    catalog["ext_id"] = -1
    rows = []
    for row in catalog.itertuples(index=False):
        r = row._asdict()
        if r["item_id"] in m.index:
            x = m.loc[r["item_id"]]
            genres = [str(g) for g in x["genres"]]
            keywords = [str(k) for k in x["keywords"]]
            body = r["text"][len(r["title"]):] if r["text"].startswith(r["title"]) else r["text"]
            text = " ".join(p for p in (
                f"{r['title']}.", str(x["overview"]), f"Genres: {' '.join(genres)}." if genres else "",
                f"Keywords: {', '.join(keywords)}." if keywords else "",
                f"Series: {x['collection']}." if x["collection"] else "", body) if p)
            r.update(text=text[:4000], genres=genres or list(r["genres"]),
                     themes=item_themes(genres or list(r["genres"]), text),
                     year=int(x["ext_year"]) if x["ext_year"] and x["ext_year"] == x["ext_year"] else r["year"],
                     ext_source=x["source"], ext_kind=x["kind"], ext_id=int(x["ext_id"]))
        rows.append(r)
    return type(catalog)(rows)
