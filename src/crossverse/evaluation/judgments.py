"""Hand-judged evaluation: does a recommendation fit someone who liked the seed?

The offline benchmark asks "will this Amazon user buy it?", which rewards bestsellers. This set asks
a person instead. Queries live in docs/judgments/queries.json; ratings in ratings.csv
(2 = good fit, 1 = okay, 0 = bad). Candidates are pooled from several systems and shown blind, so a
rating says nothing about which system proposed the item. A new system only needs ratings for the
items it adds to the pool.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crossverse.config import PROJECT_ROOT

JUDGMENTS_DIR = PROJECT_ROOT / "docs" / "judgments"
RATING_COLUMNS = ["query_id", "item_id", "title", "domain", "rating", "rated_at"]
LABELS = {2: "good fit", 1: "okay", 0: "bad"}


@dataclass(frozen=True)
class Query:
    id: str
    seed_item_id: str
    seed_title: str
    seed_domain: str
    target_domain: str


def load_queries(directory: Path = JUDGMENTS_DIR) -> list[Query]:
    data = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    return [Query(**q) for q in data["queries"]]


def load_ratings(directory: Path = JUDGMENTS_DIR) -> pd.DataFrame:
    """Latest rating per (query, item): re-rating an item replaces the earlier answer."""
    path = directory / "ratings.csv"
    if not path.exists():
        return pd.DataFrame(columns=RATING_COLUMNS)
    df = pd.read_csv(path, dtype={"rating": int})
    return df.drop_duplicates(["query_id", "item_id"], keep="last").reset_index(drop=True)


def add_rating(query_id: str, item: dict[str, Any], rating: int, directory: Path = JUDGMENTS_DIR) -> None:
    """Append one rating (append-only, so a crash or a second window never loses earlier answers)."""
    if rating not in LABELS:
        raise ValueError(f"rating must be one of {sorted(LABELS)}")
    path = directory / "ratings.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(RATING_COLUMNS)
        w.writerow([query_id, item["item_id"], item["title"], item["domain"], rating,
                    datetime.now(UTC).isoformat(timespec="seconds")])


def blind_order(query_id: str, item_ids: list[str]) -> list[str]:
    """Stable pseudo-random order, so the rater can't infer which system (or rank) proposed an item."""
    return sorted(item_ids, key=lambda i: hashlib.sha1(f"{query_id}|{i}".encode()).hexdigest())


def graded_ndcg(ranked: list[str], gains: dict[str, int], k: int = 5) -> float:
    """NDCG@k with graded gains (2^rating - 1); the ideal list uses every judged item of the query.
    Unjudged items count as 0, so check `judged_share` before trusting a low score."""
    dcg = sum((2 ** gains.get(i, 0) - 1) / np.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / np.log2(r + 2) for r, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def score_lists(lists: dict[str, list[str]], ratings: pd.DataFrame, k: int = 5) -> dict[str, float]:
    """Summary over queries for one system: `lists` maps query_id -> ranked item ids."""
    rows = []
    for qid, ranked in lists.items():
        gains = dict(zip(ratings.loc[ratings.query_id == qid, "item_id"],
                         ratings.loc[ratings.query_id == qid, "rating"], strict=True))
        top = ranked[:k]
        judged = [gains[i] for i in top if i in gains]
        rows.append({
            "ndcg": graded_ndcg(top, gains, k),
            "good": sum(r == 2 for r in judged) / max(len(top), 1),
            "bad": sum(r == 0 for r in judged) / max(len(top), 1),
            "judged": len(judged) / max(len(top), 1),
        })
    df = pd.DataFrame(rows)
    return {f"{c}@{k}": float(df[c].mean()) for c in ("ndcg", "good", "bad")} | {"judged_share": float(df.judged.mean())}
