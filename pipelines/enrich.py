"""Match catalog items to TMDB (movies/TV) and IGDB (games) -> data/external/metadata.parquet.

    python pipelines/enrich.py                  # all items (~45 min first time; cached afterwards)
    python pipelines/enrich.py --limit 200      # quick check on the most popular items

Needs TMDB_TOKEN, IGDB_CLIENT_ID and IGDB_CLIENT_SECRET in .env. Every response is cached under
data/external/, so an interrupted run resumes where it stopped. TMDB and IGDB data are for local,
non-commercial use; see the README's attribution section.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from crossverse.config import PROJECT_ROOT, get_settings, load_env
from crossverse.data import external as X
from crossverse.progress import Progress

OUT_DIR = PROJECT_ROOT / "data" / "external"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="only the N most popular items per domain")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    env = load_env()
    missing = [k for k in ("TMDB_TOKEN", "IGDB_CLIENT_ID", "IGDB_CLIENT_SECRET") if not env.get(k)]
    if missing:
        print(f"missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    items = pd.read_parquet(get_settings().data.processed_dir / "items.parquet")
    items = items.sort_values("popularity", ascending=False)
    # TV hint: does any Amazon product merged into the item look like a TV release?
    amap = pd.read_parquet(get_settings().data.processed_dir / "asin_map.parquet")
    tv = amap["title"].str.contains(r"\b(?:seasons?|series|episodes?|volume|vol\.)\b", case=False, regex=True)
    items["tv_hint"] = items["item_id"].isin(set(amap.loc[tv, "canonical_item_id"]))
    rows: list[dict] = []

    games = items[items["domain"] == "game"]
    movies = items[items["domain"] == "movie"]
    if args.limit:
        games, movies = games.head(args.limit), movies.head(args.limit)

    igdb = X.IGDB(env["IGDB_CLIENT_ID"], env["IGDB_CLIENT_SECRET"], OUT_DIR)
    tmdb = X.TMDB(env["TMDB_TOKEN"], OUT_DIR)

    def run(label: str, records: list[dict], match, threads: int) -> None:
        bar = Progress(len(records), "items", label)
        with ThreadPoolExecutor(threads) as pool:
            for m in pool.map(match, records):
                if m:
                    rows.append(X.as_row(m))
                bar.update()

    # Both sites have their own rate limit, so the two run side by side (IGDB ~4/s is the slow one).
    with ThreadPoolExecutor(2) as phases:
        jobs = [
            phases.submit(run, "IGDB games  ", games[["item_id", "title", "year"]].to_dict("records"),
                          lambda r: X.game_match(igdb, r), 4),
            phases.submit(run, "TMDB movies ", movies[["item_id", "title", "year", "tv_hint"]].to_dict("records"),
                          lambda r: X.movie_match(tmdb, r), args.threads),
        ]
        for job in jobs:
            job.result()

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "metadata.parquet", index=False)
    for dom in ("movie", "game"):
        n = len(games if dom == "game" else movies)
        hit = out[out["domain"] == dom] if len(out) else out
        print(f"{dom}: matched {len(hit)}/{n} ({len(hit) / max(n, 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
