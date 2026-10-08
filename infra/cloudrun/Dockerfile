# CrossVerse on Google Cloud Run. Built by Cloud Build from the folder scripts/deploy_cloudrun.py
# stages (code + the production model); the image lives in the project's private Artifact Registry.
# Python 3.13 matches the environment the engine was pickled in.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed CROSSVERSE_ARTIFACTS_DIR=/app/artifacts CROSSVERSE_UI_DIR=/app/apps/ui
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
RUN pip install ".[semantic]" -c constraints.txt
# Bake the sentence model into the image: cold starts must not download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY apps ./apps
COPY artifacts ./artifacts
RUN useradd --uid 1000 app && chown -R app /app
USER app

# Cloud Run sets PORT (8080). One worker: the model is per-process; Cloud Run scales instances instead.
CMD exec uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
