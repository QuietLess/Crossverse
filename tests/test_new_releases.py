from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crossverse.data import new_releases as N
from crossverse.serving.api import create_app
from crossverse.serving.new_releases import NewReleaseIndex, build_index


def row(item_id, title, kind, year, source="tmdb", ext_id=1, collection="", keywords=(), votes=500):
    return {"item_id": item_id, "domain": "game" if kind == "game" else "movie", "kind": kind, "title": title,
            "year": year, "text": f"{title}. A story.", "genres": ["Action"], "image": "", "collection": collection,
            "keywords": list(keywords), "ext_source": source, "ext_id": ext_id, "votes": votes}


def test_build_drops_what_the_catalog_has():
    rows = [row("n_m1", "Nosferatu", "movie", 2024, ext_id=1), row("n_m2", "Dune", "movie", 2021, ext_id=2),
            row("n_g3", "Elden Ring", "game", 2022, "igdb", 3), row("n_g4", "Marvel Rivals", "game", 2024, "igdb", 4)]
    known = pd.DataFrame({"ext_source": ["igdb"], "ext_id": [3]})  # Elden Ring is matched in the catalog
    titles = {"movie": {"nosferatu", "dune"}}  # unmatched catalog titles
    got = N.build(rows, known, titles)
    # the unmatched "Dune" may well be the 2021 film; the 2024 Nosferatu can't be in 2023 data (it's the 1922 one)
    assert sorted(got["title"]) == ["Marvel Rivals", "Nosferatu"]
    assert all(isinstance(t, list) for t in got["themes"])


def fake_engine(trained, dim=8, seed=0):
    """The trained fixture engine has no sentence model; give the story-match side a fake one."""
    engine, _, _ = trained
    vecs = np.random.default_rng(seed).normal(size=(len(engine.catalog), dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return engine, SimpleNamespace(embeddings_=vecs)


@pytest.fixture()
def fresh(trained):
    """Two new releases: a game close to the most popular catalog game, and a film sequel of the most
    popular catalog movie (same TMDB collection). Restores the shared engine afterwards."""
    engine, sem = fake_engine(trained)
    items = engine.catalog.items
    game = int(np.flatnonzero(engine.catalog.domain == 1)[np.argmax(engine.popularity[engine.catalog.domain == 1])])
    movie = int(np.flatnonzero(engine.catalog.domain == 0)[np.argmax(engine.popularity[engine.catalog.domain == 0])])
    new = pd.DataFrame([row("n_g900", "Brand New Game", "game", 2025, "igdb", 900),
                        row("n_m901", f"{items['title'].iloc[movie]} Returns", "movie", 2024, ext_id=901,
                            collection="Zz Saga Collection")])
    new["themes"] = [[], []]
    target = np.stack([sem.embeddings_[game], sem.embeddings_[movie]])

    def embed(texts):
        return target + 0.01

    saved = {k: engine.__dict__.pop(k, None) for k in ("new_releases_", "universe_links")}
    engine.retrievers["semantic"] = sem
    try:
        index = build_index(engine, new, None, embed=embed)
    finally:
        del engine.retrievers["semantic"]
    # the film sequel shares a franchise with the catalog movie
    n_cat = len(engine.catalog)
    index.links.setdefault(movie, []).append(n_cat + 1)
    index.links.setdefault(n_cat + 1, []).append(movie)
    engine.attach_new_releases(index)
    yield engine, sem, items["item_id"].iloc[game], items["item_id"].iloc[movie]
    for k, v in saved.items():
        engine.__dict__.pop(k, None)
        if v is not None:
            engine.__dict__[k] = v


def test_stand_ins_are_the_closest_catalog_titles_of_the_same_medium(fresh):
    engine, _, game, _ = fresh
    pairs, used = engine.expand_profile([("n_g900", 5.0)])
    assert used["n_g900"][0]["item_id"] == game  # the nearest well-liked catalog game
    assert all(d["domain"] == "game" for d in used["n_g900"])
    assert pairs[0] == (game, 4.0) and len(pairs) == len(used["n_g900"])
    assert engine.expand_profile([(game, 5.0)]) == ([(game, 5.0)], {})  # catalog picks are untouched


def test_search_and_describe_include_new_releases(fresh):
    engine, *_ = fresh
    hits = engine.search("brand new game")
    assert hits[0]["item_id"] == "n_g900" and hits[0]["new"] is True
    assert engine.describe("n_g900")["kind"] == "game"
    assert engine.resolve("n_g900") == "n_g900"
    assert engine.describe("nope") is None


def test_new_releases_row_by_story(fresh):
    engine, sem, game, movie = fresh
    engine.retrievers["semantic"] = sem
    try:
        got = engine.new_releases([game], "game")
        assert [x["item_id"] for x in got] == ["n_g900"] and got[0]["via"]
        assert engine.new_releases([game], "game", exclude_ids=["n_g900"]) == []
        assert [x["item_id"] for x in engine.new_releases([movie], "film")] == ["n_m901"]
    finally:
        del engine.retrievers["semantic"]


def test_same_universe_reaches_new_releases(fresh):
    engine, _, _, movie = fresh
    got = engine.same_universe([movie], None)
    assert got[0]["item_id"] == "n_m901" and got[0]["new"] is True  # new releases first
    assert [x["item_id"] for x in engine.same_universe(["n_m901"], "film")] == [movie]


def test_api_with_a_new_release_picked(fresh, small_settings):
    engine, _, game, movie = fresh
    with TestClient(create_app(small_settings, engine)) as c:
        r = c.post("/recommend", json={"liked": [{"item": "n_g900"}], "target_domain": "game", "k": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["resolved_profile"][0]["new"] is True and not body["unresolved"]
        assert body["stand_ins"]["n_g900"][0]["item_id"] == game
        assert len(body["items"]) == 5 and game not in [i["item_id"] for i in body["items"]]
        r = c.post("/recommend", json={"liked": [{"item": movie}], "target_domain": "film", "k": 5})
        assert [x["item_id"] for x in r.json()["same_universe"]] == ["n_m901"]
        assert c.get("/items/n_m901").json()["new"] is True


def test_index_without_new_titles_is_harmless(trained):
    engine, _, _ = trained
    assert engine.fresh is None or isinstance(engine.fresh, NewReleaseIndex)
