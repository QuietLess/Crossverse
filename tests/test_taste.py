import numpy as np
import pytest

from crossverse.serving.engine import _COMPILATION, _percentile


def _history(engine, n=3):
    items = engine.catalog.items
    top = items[items["domain"] == "game"].sort_values("popularity", ascending=False)["item_id"].head(n)
    return engine.build_history([(i, 5.0) for i in top])


def test_percentile_is_rank_normalised():
    assert _percentile(np.array([10.0, -3.0, 0.5])).tolist() == [1.0, 0.0, 0.5]
    assert _percentile(np.array([7.0])).tolist() == [1.0]


@pytest.mark.parametrize("title,is_compilation", [
    ("Harry Potter: Complete 8-Film Collection", True),
    ("The Lord of the Rings Trilogy", True),
    ("Alien Anthology (Alien / Aliens / Alien 3)", True),
    ("4 Film Favorites: Slasher", True),
    ("Hostiles", False),
    ("The Two Towers", False),
    ("Five Easy Pieces", False),
    ("Looney Tunes: Golden Collection", False),
])
def test_compilation_titles(title, is_compilation):
    assert bool(_COMPILATION.search(title)) is is_compilation


def test_taste_zero_is_the_trained_ranker(trained):
    engine, _, _ = trained
    h = _history(engine)
    a = engine.rank(h, "movie", 10)[0:3:2]
    b = engine.rank(h, "movie", 10, taste=0.0)[0:3:2]
    assert np.array_equal(a[0].idx[a[1]], b[0].idx[b[1]])


def test_full_taste_follows_content_similarity_within_quality_pool(trained, monkeypatch):
    engine, _, _ = trained
    h = _history(engine)
    monkeypatch.setattr("crossverse.serving.engine.TASTE_MIN_FANS", 0)  # tiny fixture: everything qualifies
    cands, scores, order = engine.rank(h, "movie", 10, taste=1.0, diversify_results=False)
    content = cands.scores["content"]
    # with taste=1 the order is exactly content similarity among the candidates
    assert np.array_equal(cands.idx[order], cands.idx[np.argsort(-content)[:10]])


def test_quality_floor_blocks_unpopular_similar_items(trained, monkeypatch):
    engine, _, _ = trained
    h = _history(engine)
    monkeypatch.setattr("crossverse.serving.engine.TASTE_MIN_FANS", 10**9)  # nobody qualifies
    cands, scores, order = engine.rank(h, "movie", 10, taste=1.0, diversify_results=False)
    assert np.all(scores == 0)  # similarity can't lift anything; nothing gains from taste


def test_cold_start_ignores_taste(trained):
    engine, _, _ = trained
    from crossverse.retrieval.base import History

    a = engine.rank(History.empty(), "movie", 10, preferences=["sci-fi"])
    b = engine.rank(History.empty(), "movie", 10, preferences=["sci-fi"], taste=1.0)
    assert np.array_equal(a[0].idx[a[2]], b[0].idx[b[2]])


def test_api_accepts_taste(trained, small_settings):
    from fastapi.testclient import TestClient

    from crossverse.serving.api import create_app

    engine, _, _ = trained
    title = engine.catalog.items.sort_values("popularity").iloc[-1]["title"]
    with TestClient(create_app(small_settings, engine)) as client:
        for path in ("/recommend", "/recommend/movie-to-game", "/recommend/game-to-movie"):
            r = client.post(path, json={"liked": [{"item": title}], "k": 3, "taste": 0.7})
            assert r.status_code in (200, 422), r.text  # 422 only when the title is in the other domain
        assert client.post("/recommend", json={"liked": [{"item": title}], "taste": 1.5}).status_code == 422
