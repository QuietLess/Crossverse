import numpy as np
import pytest
from fastapi.testclient import TestClient

from crossverse.serving.api import create_app
from crossverse.serving.engine import _norm, parse_query


@pytest.mark.parametrize("text, expected", [
    ("Batman Begins", ("Batman Begins", None, None)),
    ("Batman Begins (2005)", ("Batman Begins", None, 2005)),
    ("game: Batman Begins", ("Batman Begins", "game", None)),
    ("Movie:Dune [2021]", ("Dune", "movie", 2021)),
    ("TV: Arcane", ("Arcane", "movie", None)),
    ("Blade Runner 2049", ("Blade Runner 2049", None, None)),  # a bare year in the title is not a hint
])
def test_parse_query(text, expected):
    assert parse_query(text) == expected


@pytest.fixture
def shared_title(trained, monkeypatch):
    """Rename the most popular movie to '<least popular game title> (1999)': one title in both domains,
    where the movie carries a year suffix (as Amazon movie titles often do) and is the more popular one."""
    engine, _, _ = trained
    items = engine.catalog.items
    games = items[items["domain"] == "game"]
    movies = items[items["domain"] == "movie"]
    game_i = engine.catalog.index[games.loc[games["popularity"].idxmin(), "item_id"]]
    movie_i = engine.catalog.index[movies.loc[movies["popularity"].idxmax(), "item_id"]]
    title = str(engine._titles[game_i])

    engine._prefix_index(), engine._match_titles  # build lazies so monkeypatch restores the originals
    titles = engine._titles.copy()
    titles[movie_i] = f"{title} (1999)"
    monkeypatch.setattr(engine, "_titles", titles)
    monkeypatch.setattr(engine, "_norm_titles", np.array([_norm(t) for t in titles]))
    monkeypatch.delitem(engine.__dict__, "_prefix")
    monkeypatch.delitem(engine.__dict__, "_match")
    return engine, title, items.iloc[movie_i]["item_id"], items.iloc[game_i]["item_id"]


def test_year_suffix_does_not_lose_to_exact_match(shared_title):
    engine, title, movie_id, game_id = shared_title
    item_id, alternatives = engine.resolve_detail(title)
    assert item_id == movie_id  # popularity decides between equally exact matches
    assert [a["item_id"] for a in alternatives] == [game_id]


def test_domain_hints_pick_the_other_item(shared_title):
    engine, title, movie_id, game_id = shared_title
    assert engine.resolve_detail(f"game: {title}") == (game_id, [])
    assert engine.resolve_detail(title, "game") == (game_id, [])
    assert engine.resolve(f"{title} (1999)") == movie_id


def test_api_reports_ambiguity_and_accepts_domain(shared_title, small_settings):
    engine, title, movie_id, game_id = shared_title
    with TestClient(create_app(small_settings, engine)) as client:
        body = client.post("/recommend", json={"liked": [{"item": title}], "k": 3}).json()
        assert body["resolved_profile"][0]["item_id"] == movie_id
        assert [a["item_id"] for a in body["ambiguous"][title]] == [game_id]

        body = client.post("/recommend", json={"liked": [{"item": title, "domain": "game"}], "k": 3}).json()
        assert body["resolved_profile"][0]["item_id"] == game_id
        assert body["ambiguous"] == {}


def test_search_ignores_leading_article(trained, monkeypatch):
    """'martian' should rank 'The Martian' as a prefix match, not below 'Martian Child'."""
    engine, _, _ = trained
    items = engine.catalog.items
    movies = items[items["domain"] == "movie"].sort_values("popularity")
    small, big = engine.catalog.index[movies["item_id"].iloc[0]], engine.catalog.index[movies["item_id"].iloc[-1]]
    engine._prefix_index(), engine._match_titles
    titles = engine._titles.copy()
    titles[big], titles[small] = "The Zqxmartian", "Zqxmartian Child"
    monkeypatch.setattr(engine, "_titles", titles)
    monkeypatch.setattr(engine, "_norm_titles", np.array([_norm(t) for t in titles]))
    monkeypatch.delitem(engine.__dict__, "_prefix")
    monkeypatch.delitem(engine.__dict__, "_match")
    expected = [items.iloc[big]["item_id"], items.iloc[small]["item_id"]]
    assert [h["item_id"] for h in engine.search("zqxmartian", limit=2)] == expected
    assert engine.resolve("Zqxmartian") == expected[0]
