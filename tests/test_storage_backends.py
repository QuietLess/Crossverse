"""Integration tests against real Postgres / Redis.

Skipped unless the servers are provided, e.g.

    CROSSVERSE_TEST_PG_URL=postgresql://crossverse@127.0.0.1:5432/crossverse
    CROSSVERSE_TEST_REDIS_URL=redis://127.0.0.1:6379/15
"""

from __future__ import annotations

import dataclasses
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from crossverse.serving.storage import RedisCache, SQLStore

PG_URL = os.environ.get("CROSSVERSE_TEST_PG_URL", "")
REDIS_URL = os.environ.get("CROSSVERSE_TEST_REDIS_URL", "")

needs_pg = pytest.mark.skipif(not PG_URL, reason="CROSSVERSE_TEST_PG_URL not set")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="CROSSVERSE_TEST_REDIS_URL not set")


@needs_pg
def test_postgres_store_roundtrip():
    store = SQLStore(PG_URL)
    assert store.backend == "postgres" and store.ping()
    rid = uuid.uuid4().hex[:16]
    store.log_recommendations([
        {"recommendation_id": rid, "item_id": "g_x", "domain": "game", "rank": 1, "model_version": "t",
         "request_mode": "movie_to_game", "evidence": {"shared_themes": ["sci-fi"], "reason_confidence": "high"}},
        {"recommendation_id": rid + "b", "item_id": "g_y", "domain": "game", "rank": 2, "model_version": "t",
         "request_mode": "movie_to_game", "evidence": {}},
    ])
    rec = store.get_recommendation(rid)
    assert rec and rec["item_id"] == "g_x" and rec["evidence"]["shared_themes"] == ["sci-fi"]
    before = store.feedback_counts().get("save", 0)
    store.add_feedback({"item_id": "g_x", "event": "save", "recommendation_id": rid, "context": {"k": 1}})
    assert store.feedback_counts()["save"] == before + 1


@needs_redis
def test_redis_cache_roundtrip():
    cache = RedisCache(REDIS_URL)
    assert cache.ping()
    key = f"test:{uuid.uuid4().hex}"
    cache.set(key, {"items": [1, 2], "x": "y"}, ttl=30)
    assert cache.get(key) == {"items": [1, 2], "x": "y"}
    assert cache.get("missing:" + key) is None


@needs_pg
@needs_redis
def test_api_on_postgres_and_redis(trained, small_settings):
    from crossverse.serving.api import create_app

    engine, _, _ = trained
    settings = dataclasses.replace(
        small_settings,
        serving=dataclasses.replace(small_settings.serving, database_url=PG_URL, redis_url=REDIS_URL),
    )
    with TestClient(create_app(settings, engine)) as c:
        health = c.get("/health").json()
        assert health["database"] == {"backend": "postgres", "ok": True}
        assert health["cache"] == {"backend": "redis", "ok": True}
        title = engine.catalog.items.sort_values("popularity", ascending=False)
        movie = title[title["domain"] == "movie"]["title"].iloc[0]
        body = {"liked": [{"item": movie}], "k": 5}
        first = c.post("/recommend/movie-to-game", json=body).json()
        second = c.post("/recommend/movie-to-game", json=body).json()  # served from Redis
        assert [i["item_id"] for i in first["items"]] == [i["item_id"] for i in second["items"]]
        rid = first["items"][0]["recommendation_id"]
        assert c.get(f"/explain/{rid}").status_code == 200  # read back from Postgres
        assert c.post("/feedback", json={"item_id": first["items"][0]["item_id"], "event": "like",
                                         "recommendation_id": rid}).status_code == 200
        assert c.get("/admin/stats").json()["feedback"]["like"] >= 1
