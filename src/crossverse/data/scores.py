"""Public scores shown on cards: TMDB audience rating (films, series) and IGDB critic / player ratings (games).

Films and series reuse the TMDB detail responses enrichment already cached; games take one batched IGDB
request per 500 ids. Output: one row per (source, kind, ext_id), joined to catalog items and new releases
by their external id, like the rest of the TMDB/IGDB metadata.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crossverse.data.external import IGDB, TMDB

IGDB_RATING_FIELDS = "aggregated_rating, aggregated_rating_count, total_rating, total_rating_count"
COLUMNS = ["source", "kind", "ext_id", "critics", "critics_count", "audience", "audience_count"]


def tmdb_scores(tmdb: TMDB, ids: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """(kind, tmdb id) -> audience score 0-10 from the cached detail responses (fetched if missing)."""
    rows = []
    for kind, tid in ids:
        d = tmdb.details(kind, tid) or {}
        if d.get("vote_count"):
            rows.append({"source": "tmdb", "kind": kind, "ext_id": int(tid), "critics": None, "critics_count": 0,
                         "audience": round(float(d.get("vote_average") or 0), 1),
                         "audience_count": int(d["vote_count"])})
    return rows


def igdb_scores(igdb: IGDB, ids: list[int], batch: int = 500) -> list[dict[str, Any]]:
    """IGDB id -> critic score (aggregated_rating, 0-100) and overall score (total_rating, 0-100)."""
    rows = []
    ids = sorted(set(int(i) for i in ids))
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        body = f"fields {IGDB_RATING_FIELDS}; where id = ({','.join(map(str, chunk))}); limit {batch};"
        for c in igdb.query(body):
            if c.get("aggregated_rating_count") or c.get("total_rating_count"):
                rows.append({"source": "igdb", "kind": "game", "ext_id": int(c["id"]),
                             "critics": round(c["aggregated_rating"]) if c.get("aggregated_rating") else None,
                             "critics_count": int(c.get("aggregated_rating_count") or 0),
                             "audience": round(c["total_rating"]) if c.get("total_rating") else None,
                             "audience_count": int(c.get("total_rating_count") or 0)})
    return rows


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(["source", "kind", "ext_id"])
