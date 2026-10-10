"""Titles the catalog doesn't have: recent films, series and games from TMDB and IGDB.

The catalog is the Amazon Reviews 2023 snapshot (reviews end September 2023), so later releases
(Marvel Rivals, Baldur's Gate 3) and works Amazon barely sold (Cyberpunk: Edgerunners) are missing.
Nobody in the data has rated them, so they can't enter the trained model. They get the metadata the
story-match side needs (overview, genres, keywords, franchise, poster) and are served next to the
ranked list (serving/new_releases.py).

Selection: released from SINCE onwards, with enough votes to be well known, minus works already in
the catalog (same TMDB/IGDB id). Every response goes through the same on-disk cache as enrichment.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from crossverse.data.external import IGDB, TMDB, _year
from crossverse.features.themes import item_themes

SINCE = "2021-01-01"
MIN_VOTES = {"movie": 300, "tv": 150, "game": 20}  # TMDB vote count / IGDB rating count
MAX_PAGES = 100  # TMDB discover: 20 results a page
# Not stories: documentaries, reality, talk, news, soaps.
SKIP_TMDB_GENRES = {99, 10763, 10764, 10766, 10767}
IGDB_MAIN_TYPES = (0, 4, 8, 9, 10)  # main game, standalone expansion, remake, remaster, expanded game
TMDB_POSTER = "https://image.tmdb.org/t/p/w342{}"
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
