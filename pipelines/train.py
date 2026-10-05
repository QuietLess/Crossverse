"""Train all retrievers + ranker on the leakage-safe split and register a new model version.

    python pipelines/train.py [--version v1] [--max-eval-cases 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle

import pandas as pd

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry, dataset_fingerprint, new_version
from crossverse.training import train


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=None)
    p.add_argument("--max-eval-cases", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=0, help="processes for ranker data (0 = cores-2, 1 = sequential)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = get_settings()
    interactions = pd.read_parquet(s.data.processed_dir / "interactions.parquet")
    items = pd.read_parquet(s.data.processed_dir / "items.parquet")
    items["themes"] = items["themes"].map(list)
    items["genres"] = items["genres"].map(list)
    version = args.version or new_version()

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)
    s.serving.artifacts_dir.mkdir(parents=True, exist_ok=True)
    engine, split, report = train(interactions, items, s, version, args.max_eval_cases, jobs)

    registry = ModelRegistry(s.serving.artifacts_dir)
    d = registry.model_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    engine.save(d / "engine.pkl")
    with open(d / "split.pkl", "wb") as fh:
        pickle.dump(split, fh, protocol=pickle.HIGHEST_PROTOCOL)
    data_report_path = s.data.processed_dir / "data_report.json"
    data_report = json.loads(data_report_path.read_text()) if data_report_path.exists() else {}
    registry.write_manifest(version, {"dataset_id": dataset_fingerprint(s.data.processed_dir),
                                      "dataset_dir": str(s.data.processed_dir),
                                      "train_report": report, "data_report": data_report,
                                      "settings": {"model": vars(s.model), "split": vars(s.split)}})
    print(json.dumps({k: v for k, v in report.items() if k != "feature_importance"}, indent=2, default=str))
    print(f"registered {version} -> {d}")
    return version


if __name__ == "__main__":
    main()
