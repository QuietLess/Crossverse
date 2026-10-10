"""Fetch well-known films, series and games the catalog doesn't have -> data/external/new_releases.parquet.

    python pipelines/new_releases.py                 # released since 2021 (a few minutes; cached afterwards)
    python pipelines/new_releases.py --since 2023-09-01

Needs TMDB_TOKEN, IGDB_CLIENT_ID and IGDB_CLIENT_SECRET in .env (as pipelines/enrich.py). Then attach them
to a model with `python scripts/build_new_releases.py`. Re-run both to refresh: discover results are cached
by query, so pass a new --until (default: today) to see later releases.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from crossverse.config import PROJECT_ROOT, get_settings, load_env
from crossverse.data import external as X
from crossverse.data import new_releases as N
from crossverse.serving.universe import norm

OUT_DIR = PROJECT_ROOT / "data" / "external"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default=N.SINCE)
    p.add_argument("--until", default=None, help="last release date (default: today)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    env = load_env()
    missing = [k for k in ("TMDB_TOKEN", "IGDB_CLIENT_ID", "IGDB_CLIENT_SECRET") if not env.get(k)]
    if missing:
        print(f"missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    tmdb = X.TMDB(env["TMDB_TOKEN"], OUT_DIR)
    igdb = X.IGDB(env["IGDB_CLIENT_ID"], env["IGDB_CLIENT_SECRET"], OUT_DIR)

    rows = []
    for kind in ("movie", "tv"):
        got = N.tmdb_titles(tmdb, kind, args.since, args.until)
        print(f"TMDB {kind}: {len(got)}", flush=True)
        rows += got
    got = N.igdb_titles(igdb, args.since)
    print(f"IGDB games: {len(got)}", flush=True)
    rows += got

    items = pd.read_parquet(get_settings().data.processed_dir / "items.parquet")
    known = items[items["ext_id"] >= 0][["ext_source", "ext_id"]]
    unmatched = items[items["ext_id"] < 0]
    titles = {d: {norm(t) for t in g["title"]} for d, g in unmatched.groupby("domain")}
    df = N.build(rows, known, titles)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "new_releases.parquet", index=False)
    print(f"\n{len(df)} titles not in the catalog ({df['kind'].value_counts().to_dict()}) "
          f"-> {OUT_DIR / 'new_releases.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
