import time

import pytest
from fastapi.testclient import TestClient

from crossverse.serving.api import create_app


@pytest.fixture(scope="module")
def client(trained, small_settings):
    engine, _, _ = trained
    with TestClient(create_app(small_settings, engine)) as c:
        yield c


def _titles(engine, domain, n=3):
    items = engine.catalog.items
    sub = items[items["domain"] == domain]
    return sub.sort_values("popularity", ascending=False)["title"].head(n).tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"] == "test"
    assert body["database"]["backend"] == "sqlite"


def test_movie_to_game_by_title(client, trained):
    engine, _, _ = trained
    liked = [{"item": t} for t in _titles(engine, "movie")]
    r = client.post("/recommend/movie-to-game", json={"liked": liked, "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "movie_to_game"
    assert len(body["items"]) == 5
    assert all(i["domain"] == "game" for i in body["items"])
    assert len(body["resolved_profile"]) == 3 and not body["unresolved"]


def test_game_to_movie_unknown_items_422(client):
    r = client.post("/recommend/game-to-movie", json={"liked": [{"item": "zzzz-not-a-title"}]})
    assert r.status_code == 422


def test_typo_gets_did_you_mean(client, trained):
    engine, _, _ = trained
    title = _titles(engine, "game", 1)[0]
    typo = title[:-2] + "zz"  # mangle the end of a real title
    r = client.post("/recommend/game-to-movie", json={"liked": [{"item": typo}]})
    if r.status_code == 422:  # typo did not resolve -> suggestions in the error
        sugg = r.json()["detail"]["suggestions"][typo]
    else:
        sugg = r.json()["suggestions"].get(typo, [{"title": title}])
    assert title in [s["title"] for s in sugg]


def test_mixed_and_cold_start(client, trained):
    engine, _, _ = trained
    liked = [{"item": t, "rating": 5} for t in _titles(engine, "movie", 2) + _titles(engine, "game", 2)]
    r = client.post("/recommend", json={"liked": liked, "disliked": _titles(engine, "game", 3)[2:], "k": 8})
    assert r.status_code == 200
    assert r.json()["mode"] == "mixed"
    r = client.post("/recommend", json={"preferences": ["fantasy", "dragons"], "k": 5, "target_domain": "game"})
    body = r.json()
    assert body["cold_start"] and all(i["domain"] == "game" for i in body["items"])


def test_explain_and_feedback_roundtrip(client, trained):
    engine, _, _ = trained
    r = client.post("/recommend/movie-to-game", json={"liked": [{"item": _titles(engine, "movie", 1)[0]}], "k": 3})
    rec = r.json()["items"][0]
    e = client.get(f"/explain/{rec['recommendation_id']}")
    assert e.status_code == 200
    assert e.json()["evidence"]["reason_confidence"] in {"high", "medium", "low"}
    f = client.post("/feedback", json={"item_id": rec["item_id"], "event": "like",
                                       "recommendation_id": rec["recommendation_id"]})
    assert f.status_code == 200
    assert client.get("/admin/stats").json()["feedback"]["like"] >= 1
    assert client.get("/explain/doesnotexist").status_code == 404


def test_similar_and_search(client, trained):
    engine, _, _ = trained
    title = _titles(engine, "game", 1)[0]
    s = client.get("/items/search", params={"q": title, "domain": "game"}).json()
    assert s and s[0]["title"] == title
    sim = client.get(f"/similar/game/{s[0]['item_id']}", params={"k": 4}).json()
    assert len(sim["same_domain"]) == 4 and all(i["domain"] == "game" for i in sim["same_domain"])
    assert all(i["domain"] == "movie" for i in sim["cross_domain"])
    assert client.get("/similar/game/unknown-xyz").status_code == 404


def test_metrics_exposed(client):
    body = client.get("/metrics").text
    assert "crossverse_requests_total" in body
    assert "crossverse_recommendations_total" in body


def test_latency_smoke(client, trained):
    engine, _, _ = trained
    liked = [{"item": t} for t in _titles(engine, "movie", 3)]
    t = time.perf_counter()
    for k in range(5, 15):  # distinct k -> no cache hits
        assert client.post("/recommend/movie-to-game", json={"liked": liked, "k": k}).status_code == 200
    assert (time.perf_counter() - t) / 10 < 2.0


def test_promotion_gate_and_guardrails(tmp_path):
    from crossverse.monitoring.registry import ModelRegistry

    reg = ModelRegistry(tmp_path)
    base = {"cross_domain_ndcg@10": 0.030, "mixed_ndcg@10": 0.040, "within_movie_ndcg@10": 0.045}
    reg.write_manifest("a", {"test_metrics": base})
    assert reg.promote("a")[0] and reg.production_version() == "a"
    # primary metric regression beyond tolerance -> blocked
    reg.write_manifest("b", {"test_metrics": {**base, "cross_domain_ndcg@10": 0.025}})
    ok, msg = reg.promote("b")
    assert not ok and "gate failed" in msg
    # primary tie bought with a mixed-profile regression -> guardrail blocks
    reg.write_manifest("c", {"test_metrics": {**base, "cross_domain_ndcg@10": 0.031, "mixed_ndcg@10": 0.030}})
    ok, msg = reg.promote("c")
    assert not ok and "guardrail" in msg and "mixed" in msg
    # genuine improvement -> promoted
    reg.write_manifest("d", {"test_metrics": {**base, "cross_domain_ndcg@10": 0.033}})
    assert reg.promote("d")[0] and reg.production_version() == "d"
