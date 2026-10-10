"""Same-universe links: a game and the films / TV series of the same franchise.

"The Last of Us" (game) -> "The Last of Us" (HBO series); "Uncharted 4" -> "Uncharted" (2022 film);
"Super Mario Bros." (1993 film) -> the Mario games. Not a recommendation: these come from TMDB/IGDB
metadata, not taste, and are returned next to the ranked list, so offline metrics are untouched.

A film/series and a game are linked when

* their franchise names are equal: TMDB collection ("Resident Evil Collection") vs IGDB collection or
  franchise ("Resident Evil"); or
* one side is tagged as an adaptation (TMDB "based on video game"; IGDB "based on - movie",
  "tie-in", ...) and a franchise name or title root of one appears as whole words in the other's
  title ("Super Mario Bros." contains "mario").

Title equality alone is not enough: "Prey" (2017 game) and "Prey" (2022 Predator film) are unrelated.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

SCREEN_ADAPTATION = {"based on video game"}  # TMDB keywords ("based on game" also covers board games)
GAME_TIE_IN = {"based on - movie", "based on - tv show", "based on - anime", "based on - cartoons", "tie-in",
               "adapted to - movie", "adapted to - tv show"}  # IGDB keywords
# Franchise names too broad to mean "the same story": publishers, studios, sports leagues.
TOO_BROAD = {
    "disney", "pixar", "marvel", "dc", "dc comics", "dc universe", "lego", "nickelodeon", "dreamworks", "warner bros",
    "tom clancy s", "nba", "nhl", "nfl", "mlb", "fifa", "pga tour", "madden nfl", "ncaa", "nascar", "wwe", "ufc",
    "olympic games", "sports", "cartoon network", "nintendo", "sega", "walt disney", "disney s", "warner", "star", "game", "games", "movie", "the movie", "collection", "anime", "jump",
}
MIN_KEY_LEN = 4
MIN_SUBTITLED_KEY = 6  # "<franchise>: <subtitle>" titles link by franchise name from this length
MAX_LINKS = 24  # per item, most popular first
MAX_FRANCHISES = 4  # more than this is a compilation ("Super NES Classic"), not one universe

_ROMAN = r"(?:ii|iii|iv|v|vi|vii|viii|ix|x)"
_TAIL = re.compile(rf"(?:\s+(?:\d+|{_ROMAN}))+$")


def norm(text: str) -> str:
    """Lower-case ASCII words: "Pokémon" -> "pokemon", "Tom Clancy's" -> "tom clancy s"."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def collection_key(name: str) -> str:
    """TMDB "Resident Evil (Animated) Collection" -> "resident evil"."""
    name = re.sub(r"\((?:animated|live action)\)", " ", name, flags=re.I)
    name = re.sub(r"\b(?:collection|saga|series|franchise)\s*$", "", name.strip(), flags=re.I)
    return _clean_key(norm(name))


def title_root(title: str) -> str:
    """The franchise part of a title: before a subtitle, without a year or sequel number.
    "Uncharted 4: A Thief's End" -> "uncharted"; "Halo (2011)" -> "halo"; "The Last of Us Part II" -> "last of us"."""
    t = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", title)
    t = re.split(r":| - | – ", t, maxsplit=1)[0]
    t = re.sub(r"\bpart\s+(?:\d+|" + _ROMAN + r")\b", " ", norm(t))
    return _clean_key(_TAIL.sub("", " ".join(t.split())))


def _clean_key(key: str) -> str:
    key = re.sub(r"^(?:the|a|an) ", "", key).strip()
    return "" if len(key) < MIN_KEY_LEN or key in TOO_BROAD else key


def franchise_names(raw: str) -> list[str]:
    """IGDB gives "collection" or a comma-joined franchise list ("Mario Bros., Mario")."""
    raw = re.sub(r",\s*(?=(?:inc|ltd|llc|co)\b\.?)", " ", raw or "", flags=re.I)  # "Monsters, Inc." is one name
    return [k for k in (collection_key(p) for p in re.split(r",\s+", raw)) if k]


def link_items(titles: Iterable[str], kinds: Iterable[str], collections: Iterable[str],
               keywords: Iterable[Iterable[str]], popularity: np.ndarray,
               blocked: Iterable[int] = ()) -> dict[int, list[int]]:
    """Per catalog index: linked indices of the other medium, most popular first.

    `kinds` are "game" / "movie" / "tv"; `collections` the TMDB/IGDB franchise text ("" if none);
    `keywords` the TMDB/IGDB keywords. `blocked` items (DLC, box sets) are never linked to."""
    titles, kinds, collections = list(titles), list(kinds), list(collections)
    kw = [{str(k).lower() for k in ks} for ks in keywords]
    blocked = set(int(b) for b in blocked)
    norm_titles = [norm(t) for t in titles]
    roots = [title_root(t) for t in titles]
    has_subtitle = [bool(re.search(r"\S\s*(?::| - | – )\s*\S", t)) for t in titles]
    names = [franchise_names(c) for c in collections]
    flagged = [bool(kw[i] & (GAME_TIE_IN if kinds[i] == "game" else SCREEN_ADAPTATION)) for i in range(len(titles))]

    blocked |= {i for i in range(len(titles)) if len(names[i]) > MAX_FRANCHISES}

    # Index both sides by franchise name and title root, so only plausible pairs are compared.
    by_name: dict[str, set[int]] = defaultdict(set)
    by_root: dict[str, set[int]] = defaultdict(set)
    for i in range(len(titles)):
        if i in blocked:
            continue
        for k in names[i]:
            by_name[k].add(i)
        if roots[i]:
            by_root[roots[i]].add(i)

    links: dict[int, set[int]] = defaultdict(set)

    def link(i: int, js: Iterable[int]) -> None:
        for j in js:
            if (kinds[j] == "game") != (kinds[i] == "game"):
                links[i].add(j)
                links[j].add(i)

    for i in range(len(titles)):
        if i in blocked:
            continue
        # 1. the same franchise name on both sides ("Resident Evil Collection" / "Resident Evil")
        for k in names[i]:
            link(i, by_name.get(k, ()))
        # ... or "<franchise>: <subtitle>" ("Cyberpunk: Edgerunners", "The Witcher: Blood Origin"): a title that
        # opens with the other side's franchise name. Short names are left out: "The Rage: Carrie 2" isn't Rage.
        if len(roots[i]) >= MIN_SUBTITLED_KEY and has_subtitle[i]:
            link(i, by_name.get(roots[i], ()))
        if not flagged[i]:
            continue
        # 2. an adaptation or tie-in: its franchise or title root is the other side's franchise or root
        franchise_hits = set().union(*(by_name.get(k, set()) for k in names[i]), by_name.get(roots[i], set()))
        link(i, franchise_hits | set().union(*(by_root.get(k, set()) for k in names[i]), by_root.get(roots[i], set())))
        # 3. ... and, without a franchise-name hit, the other side's franchise name anywhere in its title ("Super
        # Mario Bros." holds "Mario"). Only a franchise name, not a mere title root: "Ice Age: Dawn of the
        # Dinosaurs" must not reach the sitcom "Dinosaurs". "Pokémon: Destiny Deoxys" already hit the Pokémon
        # franchise, so it never gets here and stays away from the game "Destiny". A title-root hit doesn't stop
        # it: "The Super Mario Collection" equals the root of "Super Mario 64", not the Mario franchise.
        if any((kinds[j] == "game") != (kinds[i] == "game") for j in franchise_hits):
            continue
        words = norm_titles[i].split()
        for a in range(len(words)):
            for b in range(a + 1, len(words) + 1):
                link(i, by_name.get(" ".join(words[a:b]), ()))
    return {i: sorted(js, key=lambda j: (-popularity[j], j))[:MAX_LINKS] for i, js in links.items()}


def catalog_metadata(engine: Any, matches: Any) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    """Per catalog item: title, kind, franchise text and keywords. Matches (metadata.parquet) are joined on
    the external id, so items merged by the dataset build (several products, one work) still find theirs."""
    items = engine.catalog.items
    meta = {} if matches is None else {(r.source, r.kind, int(r.ext_id)): (r.collection or "", list(r.keywords))
                                       for r in matches.itertuples(index=False)}
    src = items["ext_source"].fillna("").to_numpy() if "ext_source" in items else np.full(len(items), "")
    ext_kind = items["ext_kind"].fillna("").to_numpy() if "ext_kind" in items else np.full(len(items), "")
    ext_id = items["ext_id"].fillna(-1).astype(int).to_numpy() if "ext_id" in items else np.full(len(items), -1)
    found = [meta.get((src[i], ext_kind[i], ext_id[i]), ("", [])) for i in range(len(items))]
    return items["title"].tolist(), engine.kinds().tolist(), [f[0] for f in found], [f[1] for f in found]


def from_engine(engine: Any, matches: Any) -> dict[int, list[int]]:
    """Links for an engine's catalog, from the TMDB/IGDB matches (metadata.parquet)."""
    titles, kinds, collections, keywords = catalog_metadata(engine, matches)
    blocked = np.union1d(engine.ineligible(), engine.compilations())
    return link_items(titles, kinds, collections, keywords, engine.popularity, blocked)


def attach(engine: Any, metadata_path: Any) -> int:
    """Compute and store links on the engine from metadata.parquet; returns how many items got links
    (0 when the TMDB/IGDB enrichment hasn't been run)."""
    from crossverse.data.external import load_matches

    matches = load_matches(metadata_path)
    if matches is None:
        return 0
    links = from_engine(engine, matches)
    engine.attach_universe(links)
    return len(links)
