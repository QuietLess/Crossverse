# CrossVerse on Hugging Face Spaces (Docker SDK). Assembled and uploaded by scripts/deploy_space.py;
# the model is not in this image: start.py downloads it from a private model repo at startup.
# Python 3.13 matches the environment the engine was pickled in.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000 and expect the app on app_port (README front matter).
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user pyproject.toml README.md constraints.txt ./
COPY --chown=user src ./src
RUN pip install ".[semantic]" -c constraints.txt

COPY --chown=user apps ./apps
COPY --chown=user start.py ./
RUN mkdir -p /app/artifacts/models && chown -R user /app

USER user
ENV HOME=/home/user CROSSVERSE_ARTIFACTS_DIR=/app/artifacts CROSSVERSE_UI_DIR=/app/apps/ui
EXPOSE 7860
CMD ["python", "start.py"]
