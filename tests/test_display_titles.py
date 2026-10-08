import pandas as pd
import pytest

from crossverse.data.canonical import display_title, display_titles


@pytest.mark.parametrize("title,domain,expected", [
    # format / platform / edition noise at the end, in brackets, or at the start
    ("Avengers 4k UHD BLURAY Digital Steelbook", "movie", "Avengers"),
    ("Alien: Covenant 4k Digital 4K UHD", "movie", "Alien: Covenant"),
    ("Sleeping Beauty VHS", "movie", "Sleeping Beauty"),
    ("The Martian: Extended Edition 4K Ultra-HD", "movie", "The Martian"),
    ("Frozen (Widescreen)", "movie", "Frozen"),
    ("Godzilla (3D)", "movie", "Godzilla"),
    ("Toy Story - 10th Anniversary Edition", "movie", "Toy Story"),
    ("Looney Tunes: Golden Collection, 4-disc DVD collection", "movie", "Looney Tunes: Golden Collection"),
    ("Grand Theft Auto V Playstation 4", "game", "Grand Theft Auto V"),
    ("Injustice: Gods Among Us - Ultimate Edition /ps4", "game", "Injustice: Gods Among Us"),
    ("Skate 3 Xbox 360/ Xbox One", "game", "Skate 3"),
    ("XBOX 360 Skylanders Swap Force (Game Only)", "game", "Skylanders Swap Force"),
    ("Undertale Nintendo Switch Standard Edition [Physical]", "game", "Undertale"),
    ("Bloodborne PS4 Game of the Year Edition", "game", "Bloodborne"),
    ("Dissidia Final Fantasy - Sony PSP", "game", "Dissidia Final Fantasy"),
    ("Titanfall [Online Game Code]", "game", "Titanfall"),
    ("BioShock Infinite [Download]", "game", "BioShock Infinite"),
    ("Batman Begins w/Movie Ticket", "game", "Batman Begins"),
    # seasons collapse into the show, which is what the canonical item is
    ("Monk The Complete Series Season 1-8 DVD", "movie", "Monk"),
    ("The Big Bang Theory: The Complete Second Season", "movie", "The Big Bang Theory"),
    ("Simpsons Season 19, The", "movie", "The Simpsons"),
    # release years: one trailing year is kept and normalised; years inside lists stay put
    ("Titanic [1997] [Region Free]", "movie", "Titanic (1997)"),
    ("Good Doctor, The (2017) - Season 3", "movie", "The Good Doctor (2017)"),
    ("Men in Black (1997) / Men in Black II", "movie", "Men in Black (1997) / Men in Black II"),
    # real names that look like noise must survive
    ("Kill Switch", "movie", "Kill Switch"),
    ("The Switch", "movie", "The Switch"),
    ("Wii Fit U (Software Only)", "game", "Wii Fit U (Software Only)"),
    ("PlayStation 2", "game", "PlayStation 2"),
    ("Worms 3D - GameCube", "game", "Worms 3D"),
    ("The X-Files (Movie)", "movie", "The X-Files (Movie)"),
    ("K - The Complete Series", "movie", "K"),
    ("Unrated", "movie", "Unrated"),
])
def test_display_title(title, domain, expected):
    assert display_title(title, domain) == expected
    assert display_title(expected, domain) == expected  # idempotent


def test_display_titles_adds_year_only_on_game_collisions():
    items = pd.DataFrame({
        "domain": ["game", "game", "game", "movie", "movie", "movie"],
        "title": ["Doom", "Doom PS4", "Tomb Raider (1998)", "Doom", "Alien", "Alien VHS"],
        "year": [2001, 2016, 1998, 2005, 1999, 2003],  # movie years are DVD dates: never shown as release years
    })
    assert display_titles(items) == ["Doom (2001)", "Doom (2016)", "Tomb Raider (1998)", "Doom", "Alien", "Alien"]


def test_engine_refresh_cleans_titles_and_search(trained, monkeypatch):
    engine, _, _ = trained
    i = 0
    original = engine.catalog.items.copy()
    monkeypatch.setattr(engine.catalog, "items", original.copy())
    monkeypatch.setattr(engine, "_titles", engine._titles)
    monkeypatch.setattr(engine, "_norm_titles", engine._norm_titles)
    monkeypatch.setattr(engine.explainer, "titles", engine.explainer.titles)
    clean = engine.catalog.items.at[i, "title"]
    engine.catalog.items.at[i, "title"] = f"{clean} 4K UHD Steelbook"

    engine.refresh_display_titles()
    assert engine.item_dict(i)["title"] == clean
    assert engine.explainer.titles[i] == clean
    assert engine.search(clean, limit=1)[0]["item_id"] == engine.catalog.items.at[i, "item_id"]
    monkeypatch.undo()
    engine.__dict__.pop("_prefix", None), engine.__dict__.pop("_match", None)


def test_display_keeps_film_volumes():
    assert display_title("Kill Bill: Volume 2 [Blu-ray]", "movie") == "Kill Bill: Volume 2"
    assert display_title("Guardians of the Galaxy Vol. 2 (Bonus Content)", "movie") == "Guardians of the Galaxy Vol. 2"
    assert display_title("Cowboy Bebop, Vol. 1 DVD", "movie") == "Cowboy Bebop"
