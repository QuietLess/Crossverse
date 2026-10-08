"""Deploy CrossVerse to Google Cloud Run.

    python scripts/deploy_cloudrun.py                     # production model, service "crossverse"
    python scripts/deploy_cloudrun.py --version v12 --region europe-west1

Needs the gcloud CLI, logged in (`gcloud auth login`) with a project set (`gcloud config set project ID`)
that has billing and the Run, Cloud Build and Artifact Registry APIs enabled.
New projects also need Cloud Build's service account to be allowed to build (once):

    gcloud projects add-iam-policy-binding PROJECT_ID         --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com         --role=roles/cloudbuild.builds.builder

The engine is re-saved without machine-specific state (a pickled WindowsPath cannot load on Linux).

The staged folder (code + production model) is built by Cloud Build into an image in the project's
private Artifact Registry; nothing is published except the running service. The model holds data
derived from Amazon Reviews 2023 and TMDB/IGDB, so it must never go to a public repository.

Limits keep the bill at (near) zero for testing: scale to zero when idle, at most 2 instances.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from crossverse.config import PROJECT_ROOT, get_settings
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving.engine import CrossVerseEngine

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


def gcloud() -> str:
    found = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd"
    if found:
        return found
    if local.exists():
        return str(local)
    sys.exit("gcloud not found: install the Google Cloud SDK")


def stage(dest: Path, version: str, model_dir: Path, portable: bool = True) -> None:
    shutil.copy(PROJECT_ROOT / "infra" / "cloudrun" / "Dockerfile", dest / "Dockerfile")
    for name in ("pyproject.toml", "README.md", "constraints.txt"):
        shutil.copy(PROJECT_ROOT / name, dest / name)
    shutil.copytree(PROJECT_ROOT / "src", dest / "src", ignore=IGNORE)
    (dest / "apps").mkdir()
    shutil.copy(PROJECT_ROOT / "apps" / "__init__.py", dest / "apps" / "__init__.py")
    for sub in ("api", "ui"):  # not apps/web: the Streamlit research UI isn't part of the public app
        shutil.copytree(PROJECT_ROOT / "apps" / sub, dest / "apps" / sub, ignore=IGNORE)
    models = dest / "artifacts" / "models"
    (models / version).mkdir(parents=True)
    shutil.copy(model_dir / "manifest.json", models / version / "manifest.json")
    if portable:  # re-saved without machine-specific paths: a Windows pickle must load on Linux
        CrossVerseEngine.export_portable(model_dir / "engine.pkl", models / version / "engine.pkl")
    else:
        shutil.copy(model_dir / "engine.pkl", models / version / "engine.pkl")
    (models / "PRODUCTION").write_text(version)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="production")
    p.add_argument("--service", default="crossverse")
    p.add_argument("--region", default="europe-west1")
    p.add_argument("--memory", default="2Gi")
    args = p.parse_args(argv)

    registry = ModelRegistry(get_settings().serving.artifacts_dir)
    version = registry.production_version() if args.version == "production" else args.version
    model_dir = registry.model_dir(version)
    if not (model_dir / "engine.pkl").exists():
        print(f"no engine for {version} under {model_dir}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp), version, model_dir)
        print(f"deploying model {version} as Cloud Run service '{args.service}' in {args.region} ...", flush=True)
        cmd = [gcloud(), "run", "deploy", args.service, "--source", tmp, "--region", args.region,
               "--memory", args.memory, "--cpu", "1", "--cpu-boost", "--min-instances", "0", "--max-instances", "2",
               "--timeout", "60", "--allow-unauthenticated", "--quiet"]
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
