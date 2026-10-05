"""Promote an evaluated model version to production, subject to the cross-domain NDCG gate.

    python pipelines/publish.py [--version v...] [--tolerance 0.02] [--force]
"""

from __future__ import annotations

import argparse
import sys

from crossverse.config import get_settings
from crossverse.monitoring.registry import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=None)
    p.add_argument("--tolerance", type=float, default=0.02, help="max relative drop of cross-domain NDCG@10")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    registry = ModelRegistry(get_settings().serving.artifacts_dir)
    version = args.version or registry.versions()[-1]
    ok, msg = registry.promote(version, args.tolerance, args.force)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
