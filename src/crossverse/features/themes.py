"""Shared cross-domain theme vocabulary.

Amazon metadata has no common genre taxonomy across Movies_and_TV and Video_Games (game
categories are mostly platforms). A curated lexicon maps free text from both domains into one
set of themes, which powers cold start, ranker overlap features and explanations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Patterns match whole words (a trailing \b is added at compile time): "roman" must not fire on
# "Romance", "quest" on "question", "spy" on "Spyro", "guns" on "Gundam". Suffixes are explicit.
THEME_PATTERNS: dict[str, str] = {
    "sci-fi": r"sci-?fi|science fiction|futuristic|aliens?|extraterrestrials?|interstellar|starships?|spaceships?",
    "cyberpunk": r"cyberpunk|cyber-?punk|neon[- ]lit|megacorp\w*|dystopian city|augmented humans?|implants?",
    "space": r"outer space|space stations?|galax(?:y|ies)|galactic|(?<!blue )planets?(?! earth)|astronauts?|spacecraft"
             r"|space opera|cosmos",
    "artificial-intelligence": r"artificial intelligence|A\.?I\.?|androids?|robots?|machine consciousness"
                               r"|sentient machines?|replicants?|synthetic humans?",
    "dystopia": r"dystopi\w*|totalitarian|post-?apocalyp\w*|apocalyp\w*|wasteland|collapse of civili[sz]ation",
    "fantasy": r"fantasy|magic\w*|wizard\w*|sorcer\w*|witch\w*|elves|elven|dwar(?:f|ves)|enchanted|mythical",
    "dragons": r"dragons?",
    # "Dark Knight", "Arkham Knight", "Knight Rider" and "Kingdom Hearts" are not medieval.
    "medieval": r"medieval|(?<!dark )(?<!arkham )knights?(?! rider)|kingdoms?(?! hearts)|castles?|swords?(?:man|play)?"
                r"|feudal",
    "mythology": r"mytholog\w*|gods? of|norse|greek gods|olymp\w*|ancient legends?",
    # Not "terrific" / "terrifyingly real" / "frightening": reviewers use them about any film.
    "horror": r"horrors?|haunt\w*|demonic|demons?|possessed|nightmares?|slashers?|scary",
    "zombies": r"zombies?|undead|walking dead|infected hordes?",
    "supernatural": r"supernatural|paranormal|ghosts?|spirits?|vampires?|werewol\w*|occult",
    "survival": r"survival|surviv(?:e|es|ed|ing)|stranded|scaveng\w*|crafting",
    "thriller": r"thrillers?|suspense\w*|tense|conspirac\w*|paranoi\w*",
    "mystery": r"myster\w*|whodunit|puzzl\w*|secrets?|clues?|enigma",
    "detective": r"detectives?|investigat\w*|sleuths?|noir|private eyes?|case files?",
    "crime": r"crimes?|criminals?|gangsters?|mafia|mob boss|cartels?|heists?|robbery|underworld",
    "action": r"action-?packed|action|explosive|high-octane|shootouts?|combat",
    "war": r"war|warfare|soldiers?|military|battlefield|world war|wwii|ww2|vietnam|army|armies|troops",
    "espionage": r"spy|spies|espionage|secret agents?|covert|stealth|infiltrat\w*|assassin\w*",
    "western": r"westerns?|cowboys?|wild west|outlaws?|frontiers?|gunslingers?",
    "superhero": r"superhero\w*|super heroes|marvel|dc comics|batman|superman|spider-?man|avengers|x-men"
                 r"|justice league|mutants?",
    "martial-arts": r"martial arts?|kung fu|karate|ninja\w*|samurai|fighting tournaments?",
    "adventure": r"adventure\w*|quests?|explor\w*|treasures?|expeditions?|journeys?",
    "open-world": r"open[- ]world|sandbox|free roam|vast world",
    "role-playing": r"role-?playing|rpg|character progression|party members?|dungeons?",
    "strategy": r"strategy|tactical|tactics|turn-?based|real-?time strategy|rts|commanders?|empire building",
    "simulation": r"simulat\w*|sim|tycoon|management|city-?building|flight simulator",
    "racing": r"racing|race cars?|street racing|formula (?:one|1)|motorsports?|drift\w*|nascar|rall(?:y|ies)",
    "sports": r"sports?|football|soccer|basketball|baseball|hockey|tennis|golf|boxing|wrestling|olympics?|nba|nfl|fifa",
    "music": r"music\w*|songs?|singers?|rhythm|bands?|concerts?|danc(?:e|es|ers?|ing)|guitars?",
    "comedy": r"comed\w*|hilarious|funny|laugh\w*|humou?r\w*|parod(?:y|ies)|satir\w*|sitcoms?",
    "romance": r"romance|romantic|love story|falls? in love|lovers?|relationships?",
    "drama": r"drama\w*|emotional|heartfelt|moving story|family saga",
    "family": r"family|kids|children|all ages|animated|animation|pixar|disney|cartoons?",
    "anime": r"anime|manga|japanese animation|studio ghibli",
    "history": r"histor\w*|ancient|rome|romans?|egypt\w*|vikings?|civil war|true story|biograph\w*|period piece",
    "pirates": r"pirates?|buccaneers?|high seas|naval|sailing",
    "post-apocalyptic": r"post-?apocalyp\w*|nuclear (?:war|fallout)|after the end|wasteland|fallout",
    "time-travel": r"time travel\w*|time machines?|time loops?|travel(?:s|ed)? (?:back )?in time",
    "identity": r"identity|memories|memory loss|amnesia|who (?:he|she|they) (?:really )?(?:is|are)"
                r"|what it means to be human|self-discovery",
    "philosophical": r"philosoph\w*|existential\w*|moral\w*|ethic\w*|consciousness|free will",
    "choices-matter": r"choices? (?:matter|that matter)|branching|multiple endings|your decisions?|consequences",
    "character-driven": r"character-?driven|complex characters?|rich characters?|companions?|personal story"
                        r"|coming[- ]of[- ]age",
    "psychological": r"psycholog\w*|mind-?bending|madness|insan\w*|trauma\w*|surreal\w*",
    "political": r"politic\w*|governments?|revolution\w*|rebellion|rebels?|resistance|propaganda",
    "monsters": r"monsters?|creatures?|beasts?|kaiju|godzilla",
    "shooter": r"shooters?|first-?person|third-?person shooter|fps|guns?|firearms?|snipers?",
    "platformer": r"platform(?:er|ers|ing)|jump(?:ing)? (?:and|&) run|side-?scroll\w*",
    "puzzle": r"puzzles?|brain teas\w*|logic",
    "multiplayer": r"multiplayer|co-?op|online play|split-?screen|pvp",
}

# Broad labels that say little about taste. They still count, but specific themes come first so the
# six theme slots aren't filled by "drama, action, adventure" before "superhero" gets in.
GENERIC_THEMES = frozenset({"drama", "action", "adventure", "comedy", "family", "thriller", "mystery"})
_SUPERHERO_SHELF = frozenset({"sci-fi", "fantasy"})

# Store labels that are distributors or storefront shelves, not genres ("All Disney Titles" would tag
# every Marvel and Star Wars release as family).
_NOT_A_GENRE = re.compile(r"home (?:entertainment|video)|\btitles\b|\bstudios?\b|\bdeals?\b|\bblowout\b|\bpictures\b",
                          re.IGNORECASE)

_COMPILED = {theme: re.compile(rf"\b(?:{pat})\b", re.IGNORECASE) for theme, pat in THEME_PATTERNS.items()}
THEMES: tuple[str, ...] = tuple(THEME_PATTERNS)


def theme_counts(text: str) -> dict[str, int]:
    if not text:
        return {}
    out = {}
    for theme, rx in _COMPILED.items():
        hits = len(rx.findall(text))
        if hits:
            out[theme] = hits
    return out


def extract_themes(text: str, min_hits: int = 1) -> list[str]:
    """Return themes mentioned in text (ordered by number of hits, then vocabulary order)."""
    counts = theme_counts(text)
    order = {t: i for i, t in enumerate(THEMES)}
    return [t for t in sorted(counts, key=lambda t: (-counts[t], order[t])) if counts[t] >= min_hits]


def item_themes(genres: Iterable[str], text: str, max_themes: int = 6) -> list[str]:
    """Themes for a catalog item, precision first.

    - Store genre labels are trusted, except distributor/storefront labels ("All Disney Titles").
    - A combined label ("Science Fiction & Fantasy") means one of its parts; the description decides
      which, and only if it supports none are all parts kept.
    - Description keywords need >= 2 mentions (3 for generic themes: reviews call anything "funny" or
      "dramatic"); single mentions only fill up to 3 themes. Single keyword hits ("a family of ...")
      are the main source of nonsense tags otherwise.
    - Amazon shelves superhero films under "Science Fiction & Fantasy"; for superhero items those two
      labels are kept only when the description supports them.
    - Specific themes are listed before generic ones, so broad store genres don't use up every slot.

    `text` may contain the genre labels joined by spaces (as the catalog builder writes it); that copy
    is ignored so a label isn't counted twice.
    """
    genres = list(genres)
    joined = " ".join(genres)
    if joined and joined in text:
        text = text.replace(joined, " ", 1)
    counts = theme_counts(text)

    trusted: list[str] = []
    for label in genres:
        if _NOT_A_GENRE.search(label):
            continue
        parts = extract_themes(label)
        if len(parts) > 1:
            parts = [t for t in parts if counts.get(t)] or parts
        trusted += [t for t in parts if t not in trusted]

    order = {t: i for i, t in enumerate(THEMES)}
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    described = [t for t in ranked if counts[t] >= (3 if t in GENERIC_THEMES else 2) and t not in trusted]
    if "superhero" in trusted or "superhero" in described:
        trusted = [t for t in trusted if t not in _SUPERHERO_SHELF or counts.get(t)]

    def specific_first(themes: list[str]) -> list[str]:
        return [t for t in themes if t not in GENERIC_THEMES] + [t for t in themes if t in GENERIC_THEMES]

    out = specific_first(trusted + described)
    if len(out) < 3:
        out += [t for t in ranked if counts[t] == 1 and t not in out][: 3 - len(out)]
    return out[:max_themes]


def theme_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
