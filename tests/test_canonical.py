import pandas as pd
import pytest

from crossverse.data.canonical import (
    audit_precision,
    audit_sample,
    canonicalize,
    extract_year,
    is_accessory,
    normalize_title,
)


@pytest.mark.parametrize(
    "title,domain,expected",
    [
        ("Blade Runner 2049 [Blu-ray]", "movie", "blade runner 2049"),
        ("Blade Runner 2049 (4K Ultra HD + Blu-ray + Digital)", "movie", "blade runner 2049"),
        ("The Witcher 3: Wild Hunt - Game of the Year Edition - PlayStation 4", "game", "witcher 3 wild hunt"),
        ("Witcher 3 Wild Hunt Complete Edition (Nintendo Switch)", "game", "witcher 3 wild hunt"),
        ("Breaking Bad: Season 3", "movie", "breaking bad"),
        ("Cyberpunk 2077 - Xbox One", "game", "cyberpunk 2077"),
        ("Baldur's Gate 3 - PC Deluxe Edition", "game", "baldurs gate 3"),
        # platform words are only noise for games
        ("Bait and Switch", "movie", "bait and switch"),
        # sequel numbers and numeric titles survive
        ("2012 (2009) [Blu-ray]", "movie", "2012"),
        ("1917", "movie", "1917"),
        # roman sequel numerals
        ("Frozen II 4K UHD", "movie", "frozen 2"),
        ("Frozen 2 Blu-Ray", "movie", "frozen 2"),
        ("Final Fantasy VII", "game", "final fantasy 7"),
        ("X-Men", "movie", "x men"),
        # audit regressions: real title words are not edition noise; ordinal seasons merge
        ("The Midnight Special", "movie", "midnight special"),
        ("Midnight Limited", "movie", "midnight limited"),
        ("Gold", "movie", "gold"),
        ("Skyrim - Special Edition - Xbox One", "game", "skyrim"),
        ("Star Trek Deep Space Nine - The Complete Fifth Season", "movie", "star trek deep space nine"),
        ("South Park: The Complete Twenty-First Season", "movie", "south park"),
        ("Duckman - Seasons One & Two", "movie", "duckman"),
    ],
)
def test_normalize_title(title, domain, expected):
    assert normalize_title(title, domain) == expected


def test_extract_year_only_from_brackets():
    assert extract_year("Dune (1984)") == 1984
    assert extract_year("2012") is None


def test_accessory_filter():
    assert is_accessory("Wireless Controller for Xbox One")
    assert is_accessory("Some Game", ["Video Games", "PlayStation 4", "Accessories"])
    assert not is_accessory("Halo 3")


def test_canonicalize_merges_editions_and_splits_remakes():
    products = pd.DataFrame(
        {
            "source_asin": list("ABCDEF"),
            "domain": ["movie", "movie", "movie", "movie", "game", "game"],
            "title": [
                "Dune (1984) [DVD]", "Dune (1984) Blu-ray", "Dune (2021)", "Dune",
                "Halo 3 - Xbox 360", "Halo 3 (Legendary Edition)",
            ],
        }
    )
    out = canonicalize(products).set_index("source_asin")["canonical_item_id"]
    assert out["A"] == out["B"]  # same film, different formats
    assert out["A"] != out["C"]  # remake split by explicit year
    assert out["E"] == out["F"]  # game editions merged
    assert out["A"].startswith("m_") and out["E"].startswith("g_")


def test_audit_sample_and_precision():
    products = pd.DataFrame(
        {"source_asin": [f"a{i}" for i in range(6)], "domain": ["movie"] * 6,
         "title": ["X [DVD]", "X [Blu-ray]", "Y", "Z", "W", "V"]}
    )
    m = canonicalize(products)
    s = audit_sample(m, n_per_stratum=5)
    assert set(s["stratum"]) == {"merged", "singleton"}
    s["is_correct"] = 1
    prec = audit_precision(s)
    assert (prec["precision"] == 1.0).all()


def test_games_with_same_title_but_distant_release_years_are_split():
    products = pd.DataFrame(
        {
            "source_asin": list("ABCDE"),
            "domain": ["game"] * 5,
            "title": ["Ratchet & Clank - PlayStation 2", "Ratchet & Clank - PS4", "Ratchet & Clank (PS4) Hits",
                      "Aliens vs. Predator - PC", "Aliens vs. Predator - Xbox 360"],
            "year": [2002, 2016, 2017, 1999, 2010],
        }
    )
    out = canonicalize(products).set_index("source_asin")["canonical_item_id"]
    assert out["A"] != out["B"]  # 2002 original vs 2016 remake
    assert out["B"] == out["C"]  # Hits re-release a year later = same game
    assert out["D"] != out["E"]


def test_movies_ignore_metadata_year():
    products = pd.DataFrame(
        {"source_asin": ["A", "B"], "domain": ["movie"] * 2,
         "title": ["Casablanca [DVD]", "Casablanca [Blu-ray]"], "year": [1998, 2012]}  # disc release dates
    )
    out = canonicalize(products)["canonical_item_id"]
    assert out.iloc[0] == out.iloc[1]


@pytest.mark.parametrize("a,b,domain", [
    ("Big Bang Theory, The: The Complete Series", "The Big Bang Theory: The Complete Second Season", "movie"),
    ("MARTIAN, THE", "The Martian 4K Ultra-HD", "movie"),
    ("Jurassic Park 25th Anniversary Collection", "Jurassic Park", "movie"),
    ("Red Dead Redemption: Game of the Year Edition - Xbox One and Xbox 360", "Red Dead Redemption", "game"),
    ("Far Cry 4 Ubisoft Connect", "Far Cry 4", "game"),
    ("Cowboy Bebop, Vol. 1", "Cowboy Bebop, Vol. 5", "movie"),  # TV volumes are one show
])
def test_editions_share_a_key(a, b, domain):
    assert normalize_title(a, domain) == normalize_title(b, domain)


@pytest.mark.parametrize("a,b", [
    ("Kill Bill, Vol. 1", "Kill Bill: Volume 2"),
    ("Guardians of the Galaxy", "Guardians of the Galaxy Vol. 2 (Bonus Content)"),
])
def test_film_volumes_are_different_films(a, b):
    assert normalize_title(a, "movie") != normalize_title(b, "movie")
    assert normalize_title("Kill Bill Vol. 1 [Blu-ray]", "movie") == normalize_title("Kill Bill: Volume One", "movie")
