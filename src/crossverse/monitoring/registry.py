"""Local model registry + promotion gate, with optional MLflow experiment tracking.

artifacts/models/<version>/engine.pkl      model bundle
artifacts/models/<version>/manifest.json   data/split/metrics lineage
artifacts/models/PRODUCTION                version currently served
artifacts/runs/<version>.json              run record (always; MLflow additionally if installed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRIMARY_METRIC = "cross_domain_ndcg@10"
GUARDRAIL_METRICS = {f"{t}_ndcg@10" for t in ("within_movie", "within_game", "mixed", "cold_start",
                                               "movie_to_game", "game_to_movie")}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def dataset_fingerprint(processed_dir: Path) -> str:
    """Content hash of the processed tables: models are only comparable on the same dataset."""
    h = hashlib.sha1()
    for name in ("interactions.parquet", "items.parquet"):
        p = Path(processed_dir) / name
        if p.exists():
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:12]


def new_version() -> str:
    return datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.models = self.root / "models"
        self.runs = self.root / "runs"

    def model_dir(self, version: str) -> Path:
        return self.models / version

    def engine_path(self, version: str = "production") -> Path:
        if version == "production":
            version = self.production_version() or ""
            if not version:
                raise FileNotFoundError("no production model; run `make train publish`")
        return self.model_dir(version) / "engine.pkl"

    def write_manifest(self, version: str, manifest: dict[str, Any]) -> Path:
        d = self.model_dir(version)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {"version": version, "git_sha": git_sha(),
                    "created_at": datetime.now(UTC).isoformat(), **manifest}
        p = d / "manifest.json"
        p.write_text(json.dumps(manifest, indent=2, default=str))
        return p

    def manifest(self, version: str) -> dict[str, Any]:
        p = self.model_dir(version) / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def update_manifest(self, version: str, **updates: Any) -> None:
        m = self.manifest(version)
        m.update(updates)
        (self.model_dir(version) / "manifest.json").write_text(json.dumps(m, indent=2, default=str))

    def production_version(self) -> str | None:
        p = self.models / "PRODUCTION"
        return p.read_text().strip() if p.exists() else None

    def versions(self) -> list[str]:
        if not self.models.exists():
            return []
        return sorted(d.name for d in self.models.iterdir() if d.is_dir())

    def promote(self, candidate: str, tolerance: float = 0.02, force: bool = False,
                guardrail_tolerance: float = 0.10) -> tuple[bool, str]:
        """Model gate.

        * primary: cross-domain NDCG@10 may not drop more than `tolerance` (relative);
        * guardrails: no other task's NDCG@10 may drop more than `guardrail_tolerance` (relative),
          so a model cannot buy a cross-domain tie with a within-domain or mixed-profile regression.
        """
        cand = self.manifest(candidate).get("test_metrics", {})
        if PRIMARY_METRIC not in cand:
            return False, f"candidate {candidate} has no {PRIMARY_METRIC}; run evaluation first"
        current = self.production_version()
        if current and current != candidate and not force:
            ds_cand = self.manifest(candidate).get("dataset_id")
            ds_prod = self.manifest(current).get("dataset_id")
            if ds_cand and ds_prod and ds_cand != ds_prod:
                return False, (f"not comparable: candidate {candidate} was evaluated on dataset {ds_cand}, production "
                               f"{current} on {ds_prod}. Retrain the production recipe on the new dataset and "
                               "promote that first (or pass --force after reviewing both benchmarks).")
            prod = self.manifest(current).get("test_metrics", {})
            if PRIMARY_METRIC in prod:
                floor = prod[PRIMARY_METRIC] * (1 - tolerance)
                if cand[PRIMARY_METRIC] < floor:
                    return False, (f"gate failed: {PRIMARY_METRIC} {cand[PRIMARY_METRIC]:.4f} < "
                                   f"{floor:.4f} (production {current} {prod[PRIMARY_METRIC]:.4f}, tol {tolerance:.0%})")
            breaches = [
                f"{m} {cand[m]:.4f} vs {prod[m]:.4f} ({cand[m] / prod[m] - 1:+.1%})"
                for m in sorted(GUARDRAIL_METRICS)
                if m in cand and m in prod and prod[m] > 0 and cand[m] < prod[m] * (1 - guardrail_tolerance)
            ]
            if breaches:
                return False, (f"guardrail failed vs production {current} (tol {guardrail_tolerance:.0%}): "
                               + "; ".join(breaches))
        self.models.mkdir(parents=True, exist_ok=True)
        (self.models / "PRODUCTION").write_text(candidate)
        self.update_manifest(candidate, promoted_at=datetime.now(UTC).isoformat(), previous=current)
        return True, f"promoted {candidate} (previous: {current})"


def log_run(root: Path, version: str, params: dict[str, Any], metrics: dict[str, float],
            artifacts: list[Path] | None = None) -> None:
    runs = Path(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{version}.json").write_text(json.dumps({"version": version, "params": params, "metrics": metrics,
                                                      "git_sha": git_sha()}, indent=2, default=str))
    try:
        import mlflow  # optional
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{(Path(root) / 'mlruns').as_posix()}"))
    mlflow.set_experiment("crossverse")
    with mlflow.start_run(run_name=version):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in metrics.items()})
        for a in artifacts or []:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
