"""Space entrypoint: fetch the model from the private model repo (once per container), then serve.

The model contains data derived from Amazon Reviews 2023 and TMDB/IGDB, so it lives in a private
repo; the Space secret HF_TOKEN (a read token is enough) and variable MODEL_REPO point to it.
"""

import os
import sys
from pathlib import Path

ARTIFACTS = Path(os.environ.get("CROSSVERSE_ARTIFACTS_DIR", "/app/artifacts"))
MODELS = ARTIFACTS / "models"


def fetch_model() -> None:
    if (MODELS / "PRODUCTION").exists():
        return
    repo, token = os.environ.get("MODEL_REPO"), os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("MODEL_REPO (variable) and HF_TOKEN (secret) must be set in the Space settings")
    from huggingface_hub import snapshot_download

    print(f"downloading model from {repo} ...", flush=True)
    snapshot_download(repo_id=repo, repo_type="model", token=token, local_dir=MODELS)
    print("model ready:", (MODELS / "PRODUCTION").read_text().strip(), flush=True)


if __name__ == "__main__":
    fetch_model()
    os.execvp("uvicorn", ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"])
