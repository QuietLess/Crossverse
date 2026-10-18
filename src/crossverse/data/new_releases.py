"""Titles the catalog doesn't have: recent films, series and games from TMDB and IGDB, anime from AniList.

The catalog is the Amazon Reviews 2023 snapshot (reviews end September 2023), so later releases
(Marvel Rivals, Baldur's Gate 3) and works Amazon barely sold (Cyberpunk: Edgerunners) are missing.
Nobody in the data has rated them, so they can't enter the trained model. They get the metadata the
story-match side needs (overview, genres, keywords, franchise, poster) and are served next to the
ranked list (serving/new_releases.py).

Selection: released from SINCE onwards, with enough votes to be well known, minus works already in
the catalog (same TMDB/IGDB id). Every response goes through the same on-disk cache as enrichment.

Anime is not a medium of its own: an anime film is a film, an anime series a series, tagged with the
"Anime" genre (so its first theme is "anime"). TMDB's Japanese animation gets the tag; AniList adds the
anime TMDB doesn't know well enough, and a title both sources have is kept once, as TMDB's.
"""

from __future__ import annotations

import html
import re
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from crossverse.data.external import IGDB, TMDB, AniList, _year
from crossverse.features.themes import item_themes

SINCE = "2021-01-01"
# TMDB vote count / IGDB rating count / AniList popularity (members with the title on their list)
MIN_VOTES = {"movie": 300, "tv": 150, "game": 20, "anime": 15_000}
MAX_PAGES = 100  # TMDB discover: 20 results a page
# Not stories: documentaries, reality, talk, news, soaps.
SKIP_TMDB_GENRES = {99, 10763, 10764, 10766, 10767}
IGDB_MAIN_TYPES = (0, 4, 8, 9, 10)  # main game, standalone expansion, remake, remaster, expanded game
TMDB_POSTER = "https://image.tmdb.org/t/p/w342{}"
TMDB_ANIMATION, ANIME_GENRE = 16, "Anime"
# AniList formats that are works of their own; specials, OVAs and music videos are left out.
ANILIST_KIND = {"TV": "tv", "TV_SHORT": "tv", "ONA": "tv", "MOVIE": "movie"}
ANILIST_PAGES = 10  # 50 titles a page: the 500 most popular of the window at most (no bulk collection)
ANILIST_QUERY = """
query ($page: Int, $since: FuzzyDateInt, $minPopularity: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: ANIME, isAdult: false, startDate_greater: $since, popularity_greater: $minPopularity,
          format_in: [TV, TV_SHORT, ONA, MOVIE], sort: POPULARITY_DESC) {
      id title { english romaji } synonyms format startDate { year } description(asHtml: false)
      genres tags { name rank isMediaSpoiler } popularity coverImage { large }
      relations { edges { relationType node { type format } } }
    }
  }
}"""
_TAGS = re.compile(r"<[^>]+>")
IGDB_COVER = "https://images.igdb.com/igdb/image/upload/t_cover_big/{}.jpg"

COLUMNS = ["item_id", "domain", "kind", "title", "year", "text", "genres", "themes", "image", "collection",
           "keywords", "ext_source", "ext_id", "votes"]


def _text(title: str, overview: str, genres: list[str], keywords: list[str], collection: str) -> str:
    """Same layout as enriched catalog items (external.enrich_catalog), so embeddings are comparable."""
    return " ".join(p for p in (
        f"{title}.", overview, f"Genres: {' '.join(genres)}." if genres else "",
        f"Keywords: {', '.join(keywords)}." if keywords else "",
        f"Series: {collection}." if collection else "") if p)[:4000]


def tmdb_titles(tmdb: TMDB, kind: str, since: str = SINCE, until: str | None = None,
                min_votes: int | None = None, max_pages: int = MAX_PAGES) -> list[dict[str, Any]]:
    """Well-known films ("movie") or series ("tv") released in [since, until], with details."""
    until = until or datetime.now(UTC).date().isoformat()
    date = "primary_release_date" if kind == "movie" else "first_air_date"
    found: dict[int, dict[str, Any]] = {}
    for page in range(1, max_pages + 1):
        body = tmdb.get(f"discover/{kind}", **{f"{date}.gte": since, f"{date}.lte": until,
                                               "vote_count.gte": min_votes or MIN_VOTES[kind],
                                               "sort_by": "vote_count.desc", "include_adult": "false", "page": page})
        results = (body or {}).get("results", [])
        for c in results:
            if not SKIP_TMDB_GENRES & set(c.get("genre_ids") or []):
                found.setdefault(int(c["id"]), c)
        if page >= (body or {}).get("total_pages", 0) or not results:
            break
    rows = []
    for tid, c in found.items():
        d = tmdb.details(kind, tid) or {}
        kw = d.get("keywords", {})
        keywords = [k["name"] for k in (kw.get("keywords") or kw.get("results") or [])][:30]
        genres = [g["name"] for g in d.get("genres", [])]
        if TMDB_ANIMATION in (c.get("genre_ids") or []) and c.get("original_language") == "ja":
            genres.insert(0, ANIME_GENRE)  # first label -> first theme, the one cards show first
        title = c.get("title") or c.get("name") or ""
        collection = (d.get("belongs_to_collection") or {}).get("name", "") if kind == "movie" else ""
        overview = (d.get("overview") or c.get("overview") or "")[:2000]
        rows.append({"item_id": f"n_{kind[0]}{tid}", "domain": "movie", "kind": kind, "title": title,
                     "year": _year(c.get("release_date") or c.get("first_air_date")),
                     "text": _text(title, overview, genres, keywords, collection), "genres": genres,
                     "image": TMDB_POSTER.format(c["poster_path"]) if c.get("poster_path") else "",
                     "collection": collection, "keywords": keywords, "ext_source": "tmdb", "ext_id": tid,
                     "votes": int(c.get("vote_count") or 0)})
    return rows


def igdb_titles(igdb: IGDB, since: str = SINCE, min_votes: int | None = None, pages: int = 10) -> list[dict[str, Any]]:
    """Well-known main games (no DLC, ports or bundles) first released from `since`."""
    ts = int(datetime.fromisoformat(since).replace(tzinfo=UTC).timestamp())
    types = ",".join(map(str, IGDB_MAIN_TYPES))
    rows = []
    for page in range(pages):
        body = (f"fields {igdb.FIELDS}, cover.image_id; where first_release_date >= {ts} & "
                f"total_rating_count >= {min_votes or MIN_VOTES['game']} & game_type = ({types}) & "
                f"version_parent = null; sort total_rating_count desc; limit 500; offset {page * 500};")
        res = igdb.query(body)
        for c in res:
            names = lambda key, c=c: [x["name"] for x in c.get(key) or [] if "name" in x]  # noqa: E731
            genres = names("genres") + names("themes")
            keywords = (names("keywords") + names("player_perspectives"))[:30]
            collection = (c.get("collection") or {}).get("name", "") or ", ".join(names("franchises"))
            overview = " ".join(x for x in (c.get("summary"), c.get("storyline")) if x)[:2000]
            cover = (c.get("cover") or {}).get("image_id")
            rows.append({"item_id": f"n_g{c['id']}", "domain": "game", "kind": "game", "title": c.get("name", ""),
                         "year": _year(c.get("first_release_date")),
                         "text": _text(c.get("name", ""), overview, genres, keywords, collection), "genres": genres,
                         "image": IGDB_COVER.format(cover) if cover else "", "collection": collection,
                         "keywords": keywords, "ext_source": "igdb", "ext_id": int(c["id"]),
                         "votes": int(c.get("total_rating_count") or 0)})
        if len(res) < 500:
            break
    return rows


def anilist_titles(anilist: AniList, since: str = SINCE, min_popularity: int | None = None,
                   pages: int = ANILIST_PAGES) -> list[dict[str, Any]]:
    """Well-known anime films and series that started from `since`. A later season of a series is the
    same work (TMDB lists a series once), so titles continuing a TV series are skipped."""
    variables = {"since": int(since.replace("-", "")), "minPopularity": min_popularity or MIN_VOTES["anime"]}
    rows = []
    for page in range(1, pages + 1):
        body = anilist.query(ANILIST_QUERY, {**variables, "page": page}) or {}
        if body.get("errors") and not body.get("data"):
            raise RuntimeError(f"AniList: {body['errors'][0].get('message', 'error')}")
        found = ((body.get("data") or {}).get("Page") or {})
        for m in found.get("media") or []:
            kind = ANILIST_KIND.get(m.get("format") or "")
            edges = (m.get("relations") or {}).get("edges") or []
            sequel = any(e.get("relationType") == "PREQUEL" and (e.get("node") or {}).get("type") == "ANIME"
                         and ANILIST_KIND.get((e.get("node") or {}).get("format") or "") == "tv" for e in edges)
            names = m.get("title") or {}
            title = names.get("english") or names.get("romaji") or ""
            if kind is None or not title or (kind == "tv" and sequel):
                continue
            overview = re.sub(r"\s+", " ", html.unescape(_TAGS.sub(" ", m.get("description") or ""))).strip()[:2000]
            genres = [ANIME_GENRE, *(m.get("genres") or [])]
            keywords = [t["name"] for t in sorted(m.get("tags") or [], key=lambda t: -(t.get("rank") or 0))
                        if not t.get("isMediaSpoiler")][:30]
            rows.append({"item_id": f"n_a{m['id']}", "domain": "movie", "kind": kind, "title": title,
                         "year": (m.get("startDate") or {}).get("year"),
                         "text": _text(title, overview, genres, keywords, ""), "genres": genres,
                         "image": (m.get("coverImage") or {}).get("large") or "", "collection": "",
                         "keywords": keywords, "ext_source": "anilist", "ext_id": int(m["id"]),
                         "votes": int(m.get("popularity") or 0),
                         "_names": [n for n in (names.get("english"), names.get("romaji"), *(m.get("synonyms") or []))
                                    if n]})
        if not (found.get("pageInfo") or {}).get("hasNextPage"):
            break
    return rows


def merge_anime(rows: list[dict[str, Any]], anime: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add AniList rows to TMDB/IGDB rows. An anime TMDB also has (same kind, a matching English, romaji
    or alternative title, a year within one) stays TMDB's and gets the anime tag."""
    from crossverse.serving.universe import norm

    def key(title: str) -> str:  # "Journey’s" and "Journey's" (Unicode folding drops one, splits the other)
        return norm(re.sub(r"['’‘`]", "", title))

    by_title: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r["domain"] == "movie":
            by_title.setdefault((key(r["title"]), r["kind"]), []).append(r)
    out = list(rows)
    for a in anime:
        same = [r for name in a["_names"] for r in by_title.get((key(name), a["kind"]), [])
                if r["year"] is None or a["year"] is None or abs(r["year"] - a["year"]) <= 1]
        if same:
            if ANIME_GENRE not in same[0]["genres"]:
                same[0]["genres"] = [ANIME_GENRE, *same[0]["genres"]]
        else:
            out.append({k: v for k, v in a.items() if k != "_names"})
    return out


def fetch_rows(clients: Any, since: str = SINCE, until: str | None = None) -> list[dict[str, Any]]:
    """Films and series (TMDB), games (IGDB) and anime (AniList, when `clients.anilist` is set)."""
    rows = [*tmdb_titles(clients.tmdb, "movie", since, until), *tmdb_titles(clients.tmdb, "tv", since, until),
            *igdb_titles(clients.igdb, since)]
    if getattr(clients, "anilist", None) is not None:
        rows = merge_anime(rows, anilist_titles(clients.anilist, since))
    return rows


def build(rows: list[dict[str, Any]], known: pd.DataFrame, catalog_titles: dict[str, set[str]]) -> pd.DataFrame:
    """Drop works the catalog already has: the same TMDB/IGDB id (`known`: ext_source, ext_id), or the
    same normalised title as an *unmatched* catalog item in the same domain (`catalog_titles`). The title
    rule only applies to works out by LATEST_YEAR (an unmatched "Dune" may be the 2021 film): the catalog
    can't hold anything later, so "Nosferatu" (2024) stays next to the unmatched 1922 film."""
    from crossverse.data.external import LATEST_YEAR
    from crossverse.serving.universe import norm

    have = set(zip(known["ext_source"], known["ext_id"].astype(int), strict=True))

    def in_catalog(r: dict[str, Any]) -> bool:
        if (r["ext_source"], r["ext_id"]) in have:
            return True
        old = (r["year"] or 0) <= LATEST_YEAR
        return old and norm(r["title"]) in catalog_titles.get(r["domain"], set())

    out = [r for r in rows if r["title"] and not in_catalog(r)]
    df = pd.DataFrame(out, columns=COLUMNS).drop_duplicates("item_id")
    df["themes"] = [item_themes(g, t) for g, t in zip(df["genres"], df["text"], strict=True)]
    df["year"] = df["year"].fillna(0).astype(int)
    df.attrs["fetched_at"] = time.strftime("%Y-%m-%d")
    return df.sort_values("votes", ascending=False).reset_index(drop=True)
