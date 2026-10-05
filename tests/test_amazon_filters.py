from crossverse.data.amazon import _is_game, _is_misfiled_media


def test_misfiled_movies_and_music_are_not_games():
    dvd = {"main_category": "Movies & TV", "store": "Joe Flanigan (Actor) Format: DVD", "details": {}}
    bluray_claiming_games = {"main_category": "Video Games", "store": "LAUGIT", "details": {"Type of item": "Blu-ray"}}
    cd = {"main_category": "Video Games", "store": "Il Volo   Format: Audio CD", "details": {}}
    mpaa = {"main_category": "", "store": "", "details": {"MPAA rating": "PG-13"}}
    for rec in (dvd, bluray_claiming_games, cd, mpaa):
        assert _is_misfiled_media(rec)
        assert not _is_game("Stargate Atlantis: The Complete Series", ["Video Games", "PC", "Games"], rec)


def test_real_games_with_odd_metadata_are_kept():
    # Amazon files some real games under unrelated main categories; that alone must not drop them.
    crash = {"main_category": "Books", "store": "Activision", "details": {}}
    defender = {"main_category": "Video Games", "store": "Atari", "details": {"Contributor": "ATARI AND WILLIAMS"}}
    for rec in (crash, defender):
        assert not _is_misfiled_media(rec)
        assert _is_game("Crash Team Racing Nitro-Fueled", ["Video Games", "Nintendo Switch", "Games"], rec)


def test_hardware_in_games_category_is_dropped():
    assert not _is_game("PS3 Street Fighter IV FightStick Tournament Edition", ["Video Games", "Games"], {})
    assert not _is_game("Xbox 360 250GB Halo Reach Console Bundle", ["Video Games", "Games"], {})
