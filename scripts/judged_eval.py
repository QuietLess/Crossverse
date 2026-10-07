"""Score model versions / taste settings against your hand ratings (docs/judgments/).

    python scripts/judged_eval.py                        # production model, taste 0 / 0.5 / 1.0
    python scripts/judged_eval.py --version v10 v11 --taste 0 0.5
    python scripts/judged_eval.py --rater you            # only your own ratings (default: combined)

Per setting: graded NDCG@5 (2 = good fit, 1 = okay, 0 = bad), the share of the top 5 rated good and
bad, and `judged`: how much of the top 5 you have rated. Unrated items count as bad in NDCG, so
when `judged` is below ~90%, rate the new suggestions in the UI's ⭐ Rate tab first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from crossverse.config import get_settings
from crossverse.evaluation import judgments as J
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving.engine import CrossVerseEngine


def top_k(engine: CrossVerseEngine, q: J.Query, taste: float, k: int) -> list[str]:
    seed = engine.resolve(q.seed_item_id) or engine.resolve(f"{q.seed_domain}: {q.seed_title}")
    if seed is None:
        return []
    res = engine.recommend([(seed, 5.0)], target_domain=q.target_domain, k=k, explain=False, taste=taste)
    return [r.item_id for r in res.items]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", nargs="+", default=["production"])
    p.add_argument("--taste", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--rater", default="combined", help="you | claude | combined (yours where given, else Claude's)")
    args = p.parse_args(argv)

    queries, every = J.load_queries(), J.load_ratings()
    ratings = J.combined(every) if args.rater == "combined" else J.load_ratings(rater=args.rater, known_only=True)
    if ratings.empty:
        print("no ratings yet: rate suggestions in the UI's ⭐ Rate tab first", file=sys.stderr)
        return 1
    registry = ModelRegistry(get_settings().serving.artifacts_dir)
    rows = []
    for version in args.version:
        engine = CrossVerseEngine.load(registry.engine_path(version))
        for taste in args.taste:
            lists = {q.id: top_k(engine, q, taste, args.k) for q in queries}
            rows.append({"version": engine.version, "taste": taste, **J.score_lists(lists, ratings, args.k)})
    pd.set_option("display.width", 200)
    print(f"{len(ratings)} ratings ({args.rater}) over {ratings.query_id.nunique()} of {len(queries)} queries")
    agree = J.agreement(every)
    if agree.get("n"):
        print(f"you vs claude on {agree['n']} shared items: {agree['exact']:.0%} identical, "
              f"{agree['within_one']:.0%} within one step, {agree['opposite']:.0%} opposite")
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
