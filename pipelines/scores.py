"""Collect public scores for catalog items and new releases -> data/external/scores.parquet.

    python pipelines/scores.py
    python scripts/build_scores.py          # attach them to the production model

TMDB audience ratings come from the enrichment cache (no new requests for matched items); IGDB critic
and player ratings take one request per 500 games. Needs the same .env keys as pipelines/enrich.py.
"""

from __future__ import annotations

import sys

import pandas as pd

from crossverse.config import PROJECT_ROOT, get_settings, load_env
from crossverse.data import external as X
from crossverse.data import scores as S

OUT_DIR = PROJECT_ROOT / "data" / "external"


def main() -> int:
    env = load_env()
    missing = [k for k in ("TMDB_TOKEN", "IGDB_CLIENT_ID", "IGDB_CLIENT_SECRET") if not env.get(k)]
    if missing:
        print(f"missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    items = pd.read_parquet(get_settings().data.processed_dir / "items.parquet")[["ext_source", "ext_kind", "ext_id"]]
    new_path = OUT_DIR / "new_releases.parquet"
    new = pd.read_parquet(new_path)[["ext_source", "kind", "ext_id"]] if new_path.exists() else None
    if new is not None:
        items = pd.concat([items, new.rename(columns={"kind": "ext_kind"})])
    items = items[items["ext_id"] >= 0].drop_duplicates()

    tmdb = X.TMDB(env["TMDB_TOKEN"], OUT_DIR)
    igdb = X.IGDB(env["IGDB_CLIENT_ID"], env["IGDB_CLIENT_SECRET"], OUT_DIR)
    screen = items[items["ext_source"] == "tmdb"]
    rows = S.tmdb_scores(tmdb, list(zip(screen["ext_kind"], screen["ext_id"].astype(int), strict=True)))
    print(f"TMDB: {len(rows)} films / series with a rating", flush=True)
    games = S.igdb_scores(igdb, items.loc[items["ext_source"] == "igdb", "ext_id"].astype(int).tolist())
    print(f"IGDB: {len(games)} games with a rating", flush=True)
    out = S.frame(rows + games)
    out.to_parquet(OUT_DIR / "scores.parquet", index=False)
    print(f"{len(out)} scores -> {OUT_DIR / 'scores.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
