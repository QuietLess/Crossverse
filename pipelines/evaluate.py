"""Benchmark a registered model version on the held-out test tasks and write reports/.

    python pipelines/evaluate.py [--version v...]   (default: latest trained version)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from crossverse.config import PROJECT_ROOT, get_settings
from crossverse.evaluation.benchmark import (
    error_analysis,
    headline_metrics,
    markdown_table,
    paired_bootstrap,
    run_benchmark,
    run_benchmark_parallel,
    segment_table,
    significance_markdown,
)
from crossverse.monitoring.registry import ModelRegistry, log_run
from crossverse.serving.engine import CrossVerseEngine


def main(argv: list[str] | None = None) -> dict[str, float]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=None)
    p.add_argument("--reports-dir", default=None, help="default: reports/<version>")
    p.add_argument("--error-cases", type=int, default=50)
    p.add_argument("--jobs", type=int, default=0, help="worker processes (0 = cores-2, 1 = sequential)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = get_settings()
    registry = ModelRegistry(s.serving.artifacts_dir)
    version = args.version or registry.versions()[-1]
    d = registry.model_dir(version)
    engine = CrossVerseEngine.load(d / "engine.pkl")
    with open(d / "split.pkl", "rb") as fh:
        split = pickle.load(fh)

    n_users = int(split.summary.get("train_users", 1))
    case_rows: list[dict] = []
    if args.jobs == 1:
        df = run_benchmark(engine, split.test_tasks, n_users=n_users, case_rows=case_rows)
    else:
        df = run_benchmark_parallel(engine, d / "engine.pkl", d / "split.pkl", split.test_tasks, n_users=n_users,
                                    case_rows=case_rows, n_jobs=args.jobs or None)
    head = headline_metrics(df)
    cases = pd.DataFrame(case_rows)
    final = "crossverse_ranker+diversity"
    sig = pd.concat([paired_bootstrap(cases, final, b) for b in ("popularity", "rrf_fusion")], ignore_index=True)

    out = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "reports" / version
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    cases.to_csv(out / "benchmark_cases.csv.gz", index=False)
    sig.to_csv(out / "significance.csv", index=False)
    md = [f"# CrossVerse offline benchmark — {version}\n",
          "Test tasks from the leakage-safe split (see docs/evaluation.md). Bold = best per column.",
          "`pop_pct@10` is a popularity-bias diagnostic (1.0 = only the most popular items), lower = less biased.\n",
          "## Split\n", "```json", json.dumps(split.summary, indent=2), "```",
          "\n## Is the lift real? Final pipeline vs baselines on the same users (paired bootstrap, NDCG@10)\n",
          significance_markdown(sig),
          "\n## Results (all users)", markdown_table(df),
          "\n## Full pipeline by user segment (NDCG@10)\n", segment_table(df),
          "\n## Ranker feature importance (gain, top 15)\n"]
    fi = engine.metadata.get("feature_importance", {})
    md += ["| feature | gain |", "|---|---:|"] + [f"| {k} | {v:,.0f} |" for k, v in list(fi.items())[:15]]
    (out / "benchmark.md").write_text("\n".join(md), encoding="utf-8")

    ea = ["# Qualitative error analysis\n",
          "Random held-out cross-domain users; ✅ marks a recommendation the user actually liked later.\n"]
    for t in ("movie_to_game", "game_to_movie"):
        if split.test_tasks[t].cases:
            ea.append(error_analysis(engine, split.test_tasks[t], n=args.error_cases))
    (out / "error_analysis.md").write_text("\n".join(ea), encoding="utf-8")

    registry.update_manifest(version, test_metrics=head)
    log_run(s.serving.artifacts_dir, version, {"model": vars(s.model), "split": vars(s.split)}, head,
            [out / "benchmark.md", out / "benchmark.csv"])
    print(json.dumps(head, indent=2))
    return head


if __name__ == "__main__":
    main()
