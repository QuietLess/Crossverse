"""ASGI entrypoint: `uvicorn apps.api.main:app --port 8000`."""

from crossverse.serving.api import create_app

app = create_app()
