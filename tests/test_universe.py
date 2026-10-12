import numpy as np
import pytest
from fastapi.testclient import TestClient

from crossverse.serving.api import create_app
from crossverse.serving.universe import collection_key, link_items, title_root

BASED_ON_GAME = ["based on video game"]


def links(rows, blocked=()):
    """rows: (title, kind, collection, keywords) -> {title: [linked titles]}"""
    titles = [r[0] for r in rows]
    out = link_items(titles, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                     np.arange(len(rows), 0, -1, dtype=float), blocked)
    return {titles[i]: [titles[j] for j in js] for i, js in out.items()}


def test_keys():
    assert title_root("Uncharted 4: A Thief's End") == "uncharted"
    assert title_root("Halo (2011)") == "halo"
    assert title_root("The Last of Us Part II") == "last of us"
    assert title_root("Final Fantasy VII") == "final fantasy"
    assert collection_key("Resident Evil (Animated) Collection") == "resident evil"
    assert collection_key("Pokémon") == "pokemon"
    assert collection_key("Marvel") == ""  # too broad to mean one story


def test_adaptation_links_both_ways():
    got = links([("The Last of Us", "game", "The Last of Us", []),
                 ("The Last of Us", "tv", "", BASED_ON_GAME),
                 ("The Last Ship", "tv", "", [])])
    assert got["The Last of Us"] == ["The Last of Us"]  # each side lists the other medium only
    assert "The Last Ship" not in got


def test_same_title_alone_is_not_a_link():
    # Prey (2017, Arkane) and Prey (2022, the Predator film) share only a name
    got = links([("Prey", "game", "", []), ("Prey", "movie", "Predator Collection", []),
                 ("Aliens vs. Predator", "game", "Predator", [])])
    assert got.get("Prey") == ["Aliens vs. Predator"]  # the film reaches its franchise, not the game


def test_franchise_names_link_without_tags():
    got = links([("Silent Hill", "movie", "Silent Hill Collection", []), ("Silent Hill 2", "game", "Silent Hill", [])])
    assert got["Silent Hill"] == ["Silent Hill 2"]


def test_franchise_name_inside_a_title():
    got = links([("Super Mario Bros.", "movie", "", BASED_ON_GAME), ("Super Mario Odyssey", "game", "Mario", []),
                 ("Destiny 2", "game", "Destiny", []),
                 ("Pokemon - Destiny Deoxys", "movie", "", BASED_ON_GAME), ("Pokemon X", "game", "Pokémon", [])])
    assert got["Super Mario Bros."] == ["Super Mario Odyssey"]
    assert got["Pokemon - Destiny Deoxys"] == ["Pokemon X"]  # its own franchise, not "Destiny"
    assert "Destiny 2" not in got


def test_title_root_is_not_found_inside_other_titles():
    # a tie-in's title holding another show's whole title is not enough: "Dinosaurs" is a sitcom
    got = links([("Ice Age: Dawn of the Dinosaurs", "game", "Ice Age", ["based on - movie"]),
                 ("Dinosaurs", "tv", "", []), ("Ice Age", "movie", "Ice Age Collection", [])])
    assert got["Ice Age: Dawn of the Dinosaurs"] == ["Ice Age"]


def test_blocked_and_compilations_never_link():
    got = links([("Resident Evil", "movie", "Resident Evil Collection", BASED_ON_GAME),
                 ("Resident Evil 4", "game", "Resident Evil", []),
                 ("Resident Evil Season Pass", "game", "Resident Evil", []),
                 ("Super NES Classic", "game", "Resident Evil, Mario, Zelda, Metroid, Kirby", [])], blocked=[2])
    assert got["Resident Evil"] == ["Resident Evil 4"]


@pytest.fixture()
def linked(trained):
    """Link the most popular movie to the two most popular games, then restore the shared engine."""
    engine, _, _ = trained
    items = engine.catalog.items
    movie = int(np.flatnonzero(engine.catalog.domain == 0)[np.argmax(engine.popularity[engine.catalog.domain == 0])])
    games = np.flatnonzero(engine.catalog.domain == 1)
    games = [int(g) for g in games[np.argsort(-engine.popularity[games])][:2]]
    before = engine.__dict__.pop("universe_links", None)
    engine.attach_universe({movie: games, games[0]: [movie], games[1]: [movie]})
    yield engine, items["item_id"].iloc[movie], [items["item_id"].iloc[g] for g in games]
    engine.__dict__.pop("universe_links", None)
    if before is not None:
        engine.__dict__["universe_links"] = before


def test_engine_same_universe(linked):
    engine, movie, games = linked
    got = engine.same_universe([movie], "game")
    assert [x["item_id"] for x in got] == games and all(x["via"] for x in got)
    assert engine.same_universe([movie], "tv") == []  # filtered to the requested target
    assert [x["item_id"] for x in engine.same_universe([movie], "game", exclude_ids=[games[0]])] == games[1:]
    assert [x["item_id"] for x in engine.same_universe([games[0]], None)] == [movie]


def test_engine_without_links(trained):
    engine, _, _ = trained
    if "universe_links" not in engine.__dict__:
        assert engine.same_universe([str(engine.catalog.item_ids[0])]) == []


def test_api_returns_same_universe(linked, small_settings):
    engine, movie, games = linked
    with TestClient(create_app(small_settings, engine)) as c:
        r = c.post("/recommend", json={"liked": [{"item": movie}], "target_domain": "game", "k": 5})
    assert r.status_code == 200, r.text
    assert [x["item_id"] for x in r.json()["same_universe"]] == games


def test_franchise_colon_subtitle_links_without_tags():
    got = links([("Cyberpunk: Edgerunners", "tv", "", []), ("Cyberpunk 2077", "game", "Cyberpunk", []),
                 ("The Rage: Carrie 2", "movie", "", []), ("Rage 2", "game", "Rage", [])])
    assert got["Cyberpunk: Edgerunners"] == ["Cyberpunk 2077"]
    assert "The Rage: Carrie 2" not in got  # too short a name to trust without an adaptation tag


def test_a_title_root_hit_does_not_hide_the_franchise():
    # "The Super Mario Collection" equals the root of "Super Mario 64" but the film belongs to all of Mario
    got = links([("The Super Mario Bros. Movie", "movie", "The Super Mario Collection", BASED_ON_GAME),
                 ("Super Mario 64", "game", "Mario", []), ("Super Mario Odyssey", "game", "Mario", [])])
    assert sorted(got["The Super Mario Bros. Movie"]) == ["Super Mario 64", "Super Mario Odyssey"]
