import pytest

from crossverse.features.themes import extract_themes, item_themes


@pytest.mark.parametrize("text,expected", [
    # whole words only
    ("Romance", ["romance"]),
    ("a question of honor", []),
    ("Spyro the Dragon", ["dragons"]),
    ("Mobile Suit Gundam", []),
    ("a terrific cast", []),
    # franchise names that look like themes
    ("The Dark Knight rises", []),
    ("Batman: Arkham Knight", ["superhero"]),
    ("Kingdom Hearts II", []),
    ("Planet Earth", []),
    # the real thing still matches
    ("knights of the round table", ["medieval"]),
    ("A.I. Artificial Intelligence", ["artificial-intelligence"]),
    ("they explore distant planets", ["space", "adventure"]),
])
def test_extract_themes(text, expected):
    assert sorted(extract_themes(text)) == sorted(expected)


def test_distributor_labels_are_not_genres():
    themes = item_themes(["Walt Disney Studios Home Entertainment", "All Disney Titles", "Action"], "")
    assert themes == ["action"]


def test_combined_label_keeps_the_part_the_description_supports():
    desc = "A young wizard learns magic at a school of witchcraft."
    assert "sci-fi" not in item_themes(["Science Fiction & Fantasy"], desc)
    assert "fantasy" in item_themes(["Science Fiction & Fantasy"], desc)
    # no evidence either way: keep both rather than guess
    assert {"sci-fi", "fantasy"} <= set(item_themes(["Science Fiction & Fantasy"], "A film."))


def test_genre_labels_in_text_are_not_counted_twice():
    genres = ["Horror", "Comedy"]
    text = "Title " + " ".join(genres) + " A quiet story about a family."
    # 'Horror' / 'Comedy' appear once as labels; the text copy must not make them description evidence
    assert item_themes(genres, text) == ["horror", "comedy", "family"]


def test_superhero_items_drop_unsupported_shelf_genres():
    desc = "Batman faces the Joker. Batman must stop a criminal mastermind and his crime spree."
    themes = item_themes(["Science Fiction & Fantasy", "Science Fiction", "Action & Adventure"], desc)
    assert "superhero" in themes
    assert "sci-fi" not in themes and "fantasy" not in themes


def test_specific_themes_come_before_generic_ones():
    desc = "A superhero saga. The superhero fights crime and more crime."
    themes = item_themes(["Drama", "Action", "Adventure", "Comedy", "Kids & Family", "Mystery & Thrillers"], desc)
    assert set(themes[:2]) == {"superhero", "crime"}


def test_generic_description_themes_need_three_mentions():
    assert "comedy" not in item_themes([], "A war film. Soldiers at war. Some humor and a hint of self-parody.")
    assert "comedy" in item_themes([], "A war film. Soldiers at war. Funny, hilarious, with constant humor.")
