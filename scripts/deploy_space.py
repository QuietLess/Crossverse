"""Deploy CrossVerse to Hugging Face Spaces (public app, private model).

    python scripts/deploy_space.py                      # production model -> <you>/crossverse
    python scripts/deploy_space.py --version v12 --space crossverse-test

Needs HF_TOKEN (a Write token) in .env. What it does:
1. uploads the model (engine.pkl, manifest, PRODUCTION) to the PRIVATE model repo <you>/crossverse-model:
   it holds data derived from Amazon Reviews 2023 and TMDB/IGDB, which must not be redistributed;
2. creates the PUBLIC Docker Space <you>/<space>, sets its secret HF_TOKEN and variable MODEL_REPO
   (the Space downloads the model at startup, see infra/space/start.py);
3. uploads the code the app needs (src, apps/api, apps/ui, infra/space/*). Re-running updates both.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from crossverse.config import PROJECT_ROOT, get_settings, load_env
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving.engine import CrossVerseEngine

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


def stage_space(dest: Path) -> None:
    for name in ("Dockerfile", "README.md", "start.py"):
        shutil.copy(PROJECT_ROOT / "infra" / "space" / name, dest / name)
    for name in ("pyproject.toml", "constraints.txt"):
        shutil.copy(PROJECT_ROOT / name, dest / name)
    shutil.copytree(PROJECT_ROOT / "src", dest / "src", ignore=IGNORE)
    (dest / "apps").mkdir()
    shutil.copy(PROJECT_ROOT / "apps" / "__init__.py", dest / "apps" / "__init__.py")
    for sub in ("api", "ui"):  # not apps/web: the Streamlit research UI isn't part of the public app
        shutil.copytree(PROJECT_ROOT / "apps" / sub, dest / "apps" / sub, ignore=IGNORE)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="production", help="model version to deploy (default: production)")
    p.add_argument("--space", default="crossverse", help="Space name")
    p.add_argument("--model-repo", default="crossverse-model", help="private model repo name")
    args = p.parse_args(argv)

    token = load_env().get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in .env (Hugging Face > Settings > Access Tokens, type Write)", file=sys.stderr)
        return 1
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    registry = ModelRegistry(get_settings().serving.artifacts_dir)
    version = registry.production_version() if args.version == "production" else args.version
    model_dir = registry.model_dir(version)
    if not (model_dir / "engine.pkl").exists():
        print(f"no engine for {version} under {model_dir}", file=sys.stderr)
        return 1
    model_repo, space = f"{user}/{args.model_repo}", f"{user}/{args.space}"

    print(f"1/3 model {version} -> {model_repo} (private)")
    api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        portable = Path(tmp) / "engine.pkl"  # re-saved without machine-specific paths (loads on Linux)
        CrossVerseEngine.export_portable(model_dir / "engine.pkl", portable)
        for name, src in (("engine.pkl", portable), ("manifest.json", model_dir / "manifest.json")):
            api.upload_file(path_or_fileobj=src, path_in_repo=f"{version}/{name}", repo_id=model_repo,
                            repo_type="model", commit_message=f"CrossVerse model {version}: {name}")
    api.upload_file(path_or_fileobj=version.encode(), path_in_repo="PRODUCTION", repo_id=model_repo,
                    repo_type="model", commit_message=f"Serve {version}")

    print(f"2/3 Space {space} (public, Docker)")
    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space, "HF_TOKEN", token, description="read access to the private model repo")
    api.add_space_variable(space, "MODEL_REPO", model_repo)

    print("3/3 code")
    with tempfile.TemporaryDirectory() as tmp:
        stage_space(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message=f"Deploy CrossVerse ({version})", delete_patterns=["src/**", "apps/**"])

    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"\nSpace:  https://huggingface.co/spaces/{space}  (build log, settings)")
    print(f"App:    https://{host}.hf.space/ui/   (ready after the build, ~5-10 min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
