# CrossVerse API / web image. Build from the repo root:
#   docker build -f infra/Dockerfile -t crossverse .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp1 (LightGBM) and curl (healthcheck) track the base image; exact Debian pins break on base updates.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
RUN pip install ".[web,storage]" -c constraints.txt

COPY apps ./apps
COPY pipelines ./pipelines

RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts /app/data && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS http://localhost:8000/health || exit 1
# One worker per container: the 250 MB model and the Prometheus registry are per-process.
# Scale horizontally with replicas behind a load balancer instead of in-process workers.
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
