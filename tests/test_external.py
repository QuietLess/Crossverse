import pandas as pd
import pytest

from crossverse.data import external as X


def movie(title, date, votes, original=None):
    return {"id": hash((title, date)) % 10**6, "title": title, "original_title": original or title,
            "release_date": date, "vote_count": votes}


def tv(name, date, votes):
    return {"id": hash((name, date)) % 10**6, "name": name, "original_name": name, "first_air_date": date,
            "vote_count": votes}


@pytest.mark.parametrize("title,domain,expected", [
    ("Halo 4 - Xbox 360 (Standard Game)", "game", ("Halo 4", None)),
    ("MARVEL'S THE AVENGERS", "movie", ("THE AVENGERS", None)),
    ("Tomb Raider (2013)", "game", ("Tomb Raider", 2013)),
    ("Spider-Man - Into the Spider-Verse", "movie", ("Spider-Man - Into the Spider-Verse", None)),
    ("Unrated", "movie", ("Unrated", None)),
])
def test_search_title(title, domain, expected):
    assert X.search_title(title, domain) == expected


def test_subtitle_match_but_not_any_suffix():
    assert X.similarity("Skyrim VR", "The Elder Scrolls V: Skyrim VR", "game") == 0.95
    assert X.similarity("Tomb Raider", "Rise of the Tomb Raider", "game") < X.MIN_SIMILARITY


def test_tv_hint_and_popularity_pick_the_series():
    cands = [("movie", movie("The Walking Dead", "1936-02-29", 40)), ("tv", tv("The Walking Dead", "2010-10-31", 16000))]
    kind, c, _ = X.pick_movie("The Walking Dead", 1997, cands, tv_hint=True)
    assert kind == "tv" and c["first_air_date"].startswith("2010")


def test_year_in_title_is_a_hard_constraint_and_close_years_win():
    cands = [("movie", movie("Carrie", "1976-11-03", 3000)), ("movie", movie("Carrie", "2013-10-16", 4000))]
    assert X.pick_movie("Carrie (1976)", None, cands)[1]["release_date"].startswith("1976")
    old = [("movie", movie("Halloween", "1978-10-25", 5000)), ("movie", movie("Halloween", "2018-10-18", 5000))]
    assert X.pick_movie("Halloween", 1978, old)[1]["release_date"].startswith("1978")


def test_unrelated_titles_do_not_match():
    assert X.pick_movie("Alien", None, [("movie", movie("Alien Nation", "1988-10-07", 900))]) is None


def test_game_year_from_catalog_is_only_a_bonus():
    # Amazon says 2010 for BioShock Infinite (released 2013): still matched
    cands = [{"id": 1, "name": "BioShock Infinite", "first_release_date": 1364860800, "total_rating_count": 2700}]
    c, sim = X.pick_game("BioShock Infinite", 2010, cands)
    assert c["id"] == 1 and sim == 1.0


def test_merge_map_only_merges_confident_matches_into_the_most_liked():
    m = pd.DataFrame({
        "item_id": ["g_a", "g_b", "g_c", "m_x"], "domain": ["game", "game", "game", "movie"],
        "source": ["igdb"] * 3 + ["tmdb"], "kind": ["game"] * 3 + ["movie"], "ext_id": [7, 7, 7, 7],
        "similarity": [1.0, 0.97, 0.9, 1.0],
    })
    assert X.merge_map(m, {"g_a": 5, "g_b": 50, "g_c": 500}) == {"g_a": "g_b"}  # g_c too unsure; movie never


def test_enrich_catalog_replaces_store_genres_and_year():
    catalog = pd.DataFrame({"item_id": ["m_1", "m_2"], "domain": "movie", "title": ["The Dark Knight", "Other"],
                            "text": ["The Dark Knight Science Fiction a film", "Other x"],
                            "genres": [["Science Fiction"], ["Drama"]], "themes": [["sci-fi"], ["drama"]],
                            "year": [2009, 2001]})
    matches = pd.DataFrame({"item_id": ["m_1"], "source": ["tmdb"], "kind": ["movie"], "ext_id": [155],
                            "ext_year": [2008], "genres": [["Action", "Crime", "Thriller"]],
                            "keywords": [["joker", "vigilante", "dc comics"]], "collection": ["The Dark Knight Collection"],
                            "overview": ["Batman raises the stakes in his war on crime."]})
    out = X.enrich_catalog(catalog, matches).set_index("item_id")
    assert out.at["m_1", "year"] == 2008 and out.at["m_1", "ext_id"] == 155
    assert "sci-fi" not in out.at["m_1", "themes"] and "superhero" in out.at["m_1", "themes"]
    assert out.at["m_1", "text"].startswith("The Dark Knight. Batman raises the stakes")
    assert out.at["m_2", "ext_id"] == -1 and out.at["m_2", "themes"] == ["drama"]


def test_response_cache_survives_a_truncated_line(tmp_path):
    c = X.ResponseCache(tmp_path / "c.jsonl")
    c.put("a", [1])
    with open(tmp_path / "c.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"k": "b", "v": [')  # run interrupted mid-write
    assert X.ResponseCache(tmp_path / "c.jsonl").get("a") == [1]


def test_igdb_dates_before_1970():
    assert X._year(-315619200) == 1960 and X._year(1364860800) == 2013


def test_game_title_year_is_a_hint_within_three_years():
    mad_max = {"id": 1, "name": "Mad Max", "first_release_date": 1441065600, "total_rating_count": 300}  # 2015
    smash = {"id": 2, "name": "Super Smash Bros.", "first_release_date": 916963200, "total_rating_count": 900}  # 1999
    assert X.pick_game("Mad Max (2013)", 2013, [mad_max])[0]["id"] == 1
    assert X.pick_game("Super Smash Bros. (2011)", 2011, [smash]) is None


def test_publisher_prefix():
    assert X.PUBLISHER_PREFIX.sub("", "WB Games Mad Max") == "Mad Max"
    assert X.PUBLISHER_PREFIX.sub("", "Mad Max") == "Mad Max"
