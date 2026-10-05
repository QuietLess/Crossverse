"""Shared cross-domain theme vocabulary.

Amazon metadata has no common genre taxonomy across Movies_and_TV and Video_Games (game
categories are mostly platforms). A curated lexicon maps free text from both domains into one
set of themes, which powers cold start, ranker overlap features and explanations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

THEME_PATTERNS: dict[str, str] = {
    "sci-fi": r"sci-?fi|science fiction|futuristic|alien|extraterrestrial|interstellar|starship|spaceship",
    "cyberpunk": r"cyberpunk|cyber-?punk|neon[- ]lit|megacorp\w*|dystopian city|augmented humans?|implants?",
    "space": r"outer space|space station|galaxy|galactic|planets?|astronaut|spacecraft|space opera|cosmos",
    "artificial-intelligence": r"artificial intelligence|\bA\.?I\.?\b|androids?|robots?|machine consciousness|sentient machine|replicants?|synthetic humans?",
    "dystopia": r"dystopi\w*|totalitarian|post-?apocalyp\w*|apocalyp\w*|wasteland|collapse of civili[sz]ation",
    "fantasy": r"fantasy|magic\w*|wizard\w*|sorcer\w*|witch\w*|elves|elven|dwar(?:f|ves)|enchanted|mythical",
    "dragons": r"dragons?",
    "medieval": r"medieval|knights?|kingdoms?|castles?|swords?(?:man|play)?|feudal",
    "mythology": r"mytholog\w*|gods? of|norse|greek gods|olymp\w*|ancient legends?",
    "horror": r"horror|terrif\w*|haunt\w*|demonic|demons?|possessed|nightmare|slasher|scary|frightening",
    "zombies": r"zombies?|undead|walking dead|infected horde",
    "supernatural": r"supernatural|paranormal|ghosts?|spirits?|vampires?|werewol\w*|occult",
    "survival": r"survival|survive|stranded|scaveng\w*|crafting",
    "thriller": r"thriller|suspense\w*|tense|conspiracy|paranoi\w*",
    "mystery": r"myster\w*|whodunit|puzzl\w*|secrets?|clues?|enigma",
    "detective": r"detective|investigat\w*|sleuth|noir|private eye|case files?",
    "crime": r"crime|criminal|gangsters?|mafia|mob boss|cartel|heists?|robbery|underworld",
    "action": r"action-?packed|\baction\b|explosive|high-octane|shootouts?|combat",
    "war": r"\bwar\b|warfare|soldiers?|military|battlefield|world war|wwii|ww2|vietnam|army|troops",
    "espionage": r"spy|spies|espionage|secret agent|covert|stealth|infiltrat\w*|assassin\w*",
    "western": r"western|cowboys?|wild west|outlaws?|frontier|gunslinger",
    "superhero": r"superhero\w*|super heroes|marvel|dc comics|batman|superman|spider-?man|avengers|x-men|justice league|mutants?",
    "martial-arts": r"martial arts?|kung fu|karate|ninja\w*|samurai|fighting tournament",
    "adventure": r"adventure\w*|quest|explor\w*|treasure|expedition|journey",
    "open-world": r"open[- ]world|sandbox|free roam|vast world",
    "role-playing": r"role-?playing|\brpg\b|character progression|party members?|dungeons?",
    "strategy": r"strategy|tactical|tactics|turn-?based|real-?time strategy|\brts\b|commanders?|empire building",
    "simulation": r"simulat\w*|sim\b|tycoon|management|city-?building|flight simulator",
    "racing": r"racing|race cars?|street racing|formula|motorsport|drift\w*|nascar|rally",
    "sports": r"sports?|football|soccer|basketball|baseball|hockey|tennis|golf|boxing|wrestling|olympic|nba|nfl|fifa",
    "music": r"music\w*|songs?|singers?|rhythm|band|concert|dance|guitar",
    "comedy": r"comed\w*|hilarious|funny|laugh\w*|humou?r\w*|parody|satir\w*|sitcom",
    "romance": r"romance|romantic|love story|falls? in love|lovers?|relationship",
    "drama": r"drama\w*|emotional|heartfelt|moving story|family saga",
    "family": r"family|kids|children|all ages|animated|animation|pixar|disney|cartoon",
    "anime": r"anime|manga|japanese animation|studio ghibli",
    "history": r"histor\w*|ancient|rome|roman|egypt\w*|vikings?|civil war|true story|biograph\w*|period piece",
    "pirates": r"pirates?|buccaneers?|high seas|naval|sailing",
    "post-apocalyptic": r"post-?apocalyp\w*|nuclear (?:war|fallout)|after the end|wasteland|fallout",
    "time-travel": r"time travel\w*|time machine|time loop|travel(?:s|ed)? (?:back )?in time",
    "identity": r"identity|memories|memory loss|amnesia|who (?:he|she|they) (?:really )?(?:is|are)|what it means to be human|self-discovery",
    "philosophical": r"philosoph\w*|existential\w*|moral\w*|ethic\w*|consciousness|free will",
    "choices-matter": r"choices? (?:matter|that matter)|branching|multiple endings|your decisions?|consequences",
    "character-driven": r"character-?driven|complex characters?|rich characters?|companions?|personal story|coming[- ]of[- ]age",
    "psychological": r"psycholog\w*|mind-?bending|madness|insan\w*|trauma|surreal",
    "political": r"politic\w*|government|revolution\w*|rebellion|rebels?|resistance|propaganda",
    "monsters": r"monsters?|creatures?|beasts?|kaiju|godzilla",
    "shooter": r"shooter|first-?person|third-?person shooter|\bfps\b|guns?|firearms?|sniper",
    "platformer": r"platform(?:er|ing)|jump(?:ing)? (?:and|&) run|side-?scroll\w*",
    "puzzle": r"puzzles?|brain teas\w*|logic",
    "multiplayer": r"multiplayer|co-?op|online play|split-?screen|pvp",
}

_COMPILED = {theme: re.compile(rf"\b(?:{pat})", re.IGNORECASE) for theme, pat in THEME_PATTERNS.items()}
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

    Curated genre labels map with full trust; description keywords need >= 2 mentions, and single
    mentions only fill up to 3 themes. Single keyword hits ("a family of ...") are the main source
    of nonsense tags otherwise.
    """
    out = list(dict.fromkeys(extract_themes(" ; ".join(genres))))
    counts = theme_counts(text)
    order = {t: i for i, t in enumerate(THEMES)}
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    out += [t for t in ranked if counts[t] >= 2 and t not in out]
    if len(out) < 3:
        out += [t for t in ranked if counts[t] == 1 and t not in out][: 3 - len(out)]
    return out[:max_themes]


def theme_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
