"""Raw Amazon Reviews 2023 (or a synthetic fixture) -> data/processed/*.parquet.

    python pipelines/build_dataset.py --download        # real data (~830MB download)
    python pipelines/build_dataset.py --synthetic        # offline fixture for CI / demos
"""

from __future__ import annotations

import argparse
import json
import logging

from crossverse.config import DataConfig
from crossverse.data import amazon, synthetic


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="download raw files first")
    p.add_argument("--synthetic", action="store_true", help="generate the synthetic fixture instead")
    p.add_argument("--synthetic-users", type=int, default=3000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = DataConfig()
    if args.synthetic:
        report = synthetic.write_fixture(cfg.processed_dir, n_users=args.synthetic_users, seed=cfg.seed)
    else:
        if args.download:
            amazon.download(cfg.raw_dir)
        report = amazon.build_dataset(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
