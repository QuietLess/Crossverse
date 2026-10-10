"""Attach the new-releases index (titles newer than the review data) to a trained model, in place.

    python pipelines/new_releases.py                       # fetch from TMDB / IGDB first
    python scripts/build_new_releases.py                   # the PRODUCTION version
    python scripts/build_new_releases.py --version v13

Embeds the titles with the model's sentence model (a few minutes on CPU) and links them to catalog
franchises. Rankings are unchanged: new titles are served next to the ranked list. Training attaches
them too when data/external/new_releases.parquet exists.
"""

from __future__ import annotations

import argparse
import shutil

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving import new_releases
from crossverse.serving.engine import CrossVerseEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=None, help="model version (default: PRODUCTION)")
    args = p.parse_args(argv)
    s = get_settings()
    path = ModelRegistry(s.serving.artifacts_dir).engine_path(args.version or s.serving.model_version)
    engine = CrossVerseEngine.load(path)
    n = new_releases.attach(engine, s.data.external_metadata.parent / "new_releases.parquet", s.data.external_metadata)
    if not n:
        print("no new releases: run pipelines/new_releases.py first")
        return 1
    backup = path.with_suffix(".pkl.bak")
    if not backup.exists():
        shutil.copy(path, backup)
    engine.save(path)
    print(f"{n} new releases -> {path} (previous file kept as {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
