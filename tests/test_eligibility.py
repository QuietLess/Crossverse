import pytest

from crossverse.data.canonical import display_title
from crossverse.serving.engine import _NOT_RECOMMENDABLE


@pytest.mark.parametrize("title", [
    "Xbox LIVE 1600 Microsoft Points", "Madden NFL 15: 2,200 Points", "Halo Infinite: 500 Halo Credits",
    "Roblox Digital Gift Code for 1,200 Robux", "60GB System", "Slim 120GB (Old Model)", "500 GB Destiny Bundle",
    "SNES/NES FC Twin Video Game System - Black", "Digital Innovations 4190100 Clean Dr. Laser Lens Cleaner for",
    "Kinect Sensor with Kinect Adventures!", "Visual Memory Unit - White", "Wii Stand (RVL-017)",
    "amFilm Tempered Screen Protector for Nintendo Switch Lite", "Switch Case Compatible with Nintendo Switch Lite",
    "Destiny Expansion II: House of Wolves", "Hitman - Upgrade Pack", "Super Smash Bros. Ultimate: Challenger Pack 2",
    "Skylanders SWAP Force: Rip Tide Character", "Skylanders Giants Triple Pack #6 (Eruptor, Stealth Elf & Terrafin)",
    "Disney Infinity 3.0 Edition: Pixar's The Good Dinosaur Power Disc Pack", "Minecraft Game Voucher",
    "Star Wars: The Old Republic 60-Day Pre-paid Time Card", "Wii Hardware Bundle - Black",
    "McAfee Total Protection 2015 | 3 Devices | PC Key Card",
])
def test_non_games_are_blocked(title):
    assert _NOT_RECOMMENDABLE.search(title)


@pytest.mark.parametrize("title", [
    "Resident Evil: Code Veronica X", "System Shock 2", "Code Vein", "Kingdom Hearts Melody of Memory",
    "SingStar Queen - Stand Alone", "Wii Play with Wii Remote", "Trace Memory", "Code Name: S.T.E.A.M.",
    "Gran Turismo Sport", "Halo 3", "Skylanders Giants Starter Pack", "LEGO Dimensions Starter Pack",
    "Borderlands 2 - 4 Pack", "UNCHARTED Greatest Hits Dual Pack", "Sonic Forces: Bonus Edition",
])
def test_real_games_are_not_blocked(title):
    assert not _NOT_RECOMMENDABLE.search(title)


@pytest.mark.parametrize("title,domain,expected", [
    ("The Last of Us Remastered Hits", "game", "The Last of Us"),
    ("Yakuza 0 - PlayStation Hits", "game", "Yakuza 0"),
    ("Interstellar (Collectible Gift Set)", "movie", "Interstellar"),
    ("Smash Hits", "movie", "Smash Hits"),
])
def test_display_title_budget_lines_and_gift_sets(title, domain, expected):
    assert display_title(title, domain) == expected
