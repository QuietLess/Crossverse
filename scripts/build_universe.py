"""Add same-universe links (adaptations, tie-ins) to an already trained model, in place.

    python scripts/build_universe.py            # the PRODUCTION version
    python scripts/build_universe.py --version v12

Links come from data/external/metadata.parquet (pipelines/enrich.py); scores and rankings are unchanged.
New models get them at training time (pipelines/train.py).
"""

from __future__ import annotations

import argparse
import shutil

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving import universe
from crossverse.serving.engine import CrossVerseEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=None, help="model version (default: PRODUCTION)")
    args = p.parse_args(argv)
    s = get_settings()
    path = ModelRegistry(s.serving.artifacts_dir).engine_path(args.version or s.serving.model_version)
    engine = CrossVerseEngine.load(path)
    n = universe.attach(engine, s.data.external_metadata)
    if not n:
        print(f"no links: {s.data.external_metadata} missing or empty (run pipelines/enrich.py)")
        return 1
    backup = path.with_suffix(".pkl.bak")
    if not backup.exists():
        shutil.copy(path, backup)
    engine.save(path)
    print(f"{n} items with same-universe links -> {path} (previous file kept as {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
