"""Attach public scores (TMDB audience, IGDB critics / players) to a trained model, in place.

    python pipelines/scores.py               # collect them first
    python scripts/build_scores.py           # the PRODUCTION version
    python scripts/build_scores.py --version v13

Run after scripts/build_new_releases.py, so new releases get theirs too. Rankings are unchanged.
"""

from __future__ import annotations

import argparse
import shutil

import pandas as pd

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving.engine import CrossVerseEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=None, help="model version (default: PRODUCTION)")
    args = p.parse_args(argv)
    s = get_settings()
    scores = s.data.external_metadata.parent / "scores.parquet"
    if not scores.exists():
        print("no scores: run pipelines/scores.py first")
        return 1
    path = ModelRegistry(s.serving.artifacts_dir).engine_path(args.version or s.serving.model_version)
    engine = CrossVerseEngine.load(path)
    n = engine.attach_scores(pd.read_parquet(scores))
    backup = path.with_suffix(".pkl.bak")
    if not backup.exists():
        shutil.copy(path, backup)
    engine.save(path)
    print(f"{n} items with a public score -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
