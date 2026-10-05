"""Terminal client for the production model.

    crossverse recommend "Blade Runner 2049" "Ex Machina" --to game
    crossverse recommend "Witcher 3" --to movie --dislike "Twilight"
    crossverse similar "Mass Effect" --domain game
    crossverse search witcher
"""

from __future__ import annotations

import argparse
import json
import sys

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving.engine import CrossVerseEngine

ICON = {"movie": "[movie]", "game": "[game] "}


def _engine(version: str) -> CrossVerseEngine:
    s = get_settings()
    return CrossVerseEngine.load(ModelRegistry(s.serving.artifacts_dir).engine_path(version))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crossverse", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="production")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recommend")
    r.add_argument("liked", nargs="*")
    r.add_argument("--dislike", action="append", default=[])
    r.add_argument("--to", choices=["movie", "game", "both"], default="both")
    r.add_argument("--theme", action="append", default=[], help="cold-start theme preference")
    r.add_argument("-k", type=int, default=10)
    r.add_argument("--json", action="store_true")
    s = sub.add_parser("similar")
    s.add_argument("item")
    s.add_argument("--domain", choices=["movie", "game"], default=None)
    s.add_argument("-k", type=int, default=8)
    q = sub.add_parser("search")
    q.add_argument("query")
    q.add_argument("--domain", choices=["movie", "game"], default=None)
    args = p.parse_args(argv)

    eng = _engine(args.version)
    if args.cmd == "search":
        for h in eng.search(args.query, args.domain, 15):
            print(f"{ICON[h['domain']]} {h['title']}  ({h['item_id']}, liked by {h['popularity']})")
        return 0
    if args.cmd == "similar":
        iid = eng.resolve(args.item, args.domain)
        if not iid:
            print(f"not found: {args.item}", file=sys.stderr)
            return 1
        sim = eng.similar(iid, args.k)
        print(f"Similar to {sim['item'][0]['title']}:")
        for label in ("same_domain", "cross_domain"):
            print(f"\n  {label.replace('_', ' ')}:")
            for x in sim[label]:
                print(f"    {ICON[x['domain']]} {x['title']}  — {', '.join(x['themes'][:3])}")
        return 0

    liked, missing = [], []
    for name in args.liked:
        iid = eng.resolve(name)
        (liked.append((iid, 5.0)) if iid else missing.append(name))
    disliked = [i for i in (eng.resolve(n) for n in args.dislike) if i]
    target = None if args.to == "both" else args.to
    res = eng.recommend(liked, disliked, target, args.k, preferences=args.theme)
    if args.json:
        print(json.dumps([r.__dict__ for r in res.items], indent=2, default=str))
        return 0
    if missing:
        print(f"(not found: {', '.join(missing)})")
    titles = [eng.item_dict(eng.catalog.index[i])["title"] for i, _ in liked]
    print(f"Profile: {', '.join(titles) or 'cold start ' + str(args.theme)}  ->  {args.to}\n")
    for r_ in res.items:
        year = f" ({r_.year})" if r_.year and str(r_.year) not in r_.title else ""
        print(f"{r_.rank:>2}. {ICON[r_.domain]} {r_.title}{year}")
        print(f"      {r_.evidence.get('summary', '')}")
    print(f"\nmodel {res.model_version} · {res.candidate_count} candidates · {res.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
