"""Canonical item resolution.

Amazon items are *products* (DVD, Blu-ray, 4K, PS4 edition, GOTY bundle ...), not entertainment
entities. We collapse product variants into a canonical_item_id so that one movie or game is one
item. Strategy:

1. Normalise titles: lowercase, strip media-format / platform / edition tokens, punctuation.
2. Group by (domain, normalised title).
3. Inside a group, split by explicit release year when titles carry conflicting years
   (e.g. "Dune (1984)" vs "Dune (2021)"); items without a year join the largest cluster.
4. Keep the source_asin -> canonical_item_id mapping and export a stratified audit sample.

TV seasons are deliberately merged into their series ("Breaking Bad: Season 3" -> "breaking bad"):
taste is expressed for the show, and keeping seasons separate would inflate franchise repetition.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

_FORMAT = (
    r"dvd|blu[\s-]?ray|bluray|4k(?:\s*ultra\s*hd)?|ultra\s*hd|uhd|hd\s*dvd|vhs|digital(?:\s*(?:copy|hd|code|download))?"
    r"|widescreen|full\s*screen|fullscreen|region\s*\d|ntsc|pal|steelbook|3d|2d|dts|dolby|subtitled|dubbed"
    r"|prime\s*video|amazon\s*original|theatrical(?:\s*(?:cut|version))?|unrated|rated|import|multi[\s-]?format"
    r"|digibook|slipcover|combo(?:\s*pack)?|\d+\s*-?\s*discs?|\d+\s*-?\s*disc\s*set|box\s*set|boxed\s*set"
)
_PLATFORM = (
    r"ps[1-5]|ps\s*vita|psp|playstation(?:\s*[1-5])?(?:\s*vita|\s*portable)?|xbox(?:\s*(?:one|360|series\s*[xs](?:\s*\|\s*[xs])?))?"
    r"|nintendo\s*(?:switch|3ds|ds|wii\s*u|wii|64|gamecube)|switch|3ds|nds|wii\s*u|wii|gamecube|n64|game\s*boy(?:\s*(?:advance|color))?"
    r"|pc(?:\s*(?:dvd|cd|download|code|online\s*game\s*code))?|mac|windows(?:\s*\d+)?|steam|online\s*game\s*code|game\s*code"
    r"|sega\s*genesis|dreamcast|stadia"
)
_EDITION_SUFFIX = r"(?:edition|collection|cut|version|pack)"
_EDITION = (
    # Unambiguous on their own.
    r"(?:(?:game\s*of\s*the\s*year|goty|deluxe|collector'?s?|remastered|remaster|director'?s?\s*cut|criterion"
    r"|greatest\s*hits|player'?s?\s*choice|steelbook|digital\s*deluxe|season\s*pass)"
    rf"(?:\s*{_EDITION_SUFFIX})?"
    # Real title words too ("Midnight Special", "Gold", "Ultimate Spider-Man"): noise only with a suffix.
    r"|(?:standard|limited|special|complete|definitive|gold|ultimate|premium|day\s*one|launch|anniversary|extended"
    rf"|signature|legendary|platinum|essentials|hits|enhanced|bundle)\s*{_EDITION_SUFFIX})"
)
_NUM_WORD = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_ORDINAL = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth"
    r"|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth"
    r"|(?:twenty|thirty)[\s-]?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|\d+(?:st|nd|rd|th))"
)
_SEASON = (
    rf"(?:the\s*)?(?:complete\s*)?(?:season|series|volume|vol\.?|seasons)\s*{_NUM_WORD}"
    rf"(?:\s*(?:-|to|&|and)\s*{_NUM_WORD})?"
    rf"|(?:the\s*)?(?:complete\s*)?{_ORDINAL}\s*(?:season|series)"
)

GAME_YEAR_TOLERANCE = 3  # same-titled games released > 3 years apart are treated as different games

_BRACKETS = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
_TOKENS = {
    "movie": re.compile(rf"\b(?:{_FORMAT}|{_EDITION}|{_SEASON})\b", re.IGNORECASE),
    "game": re.compile(rf"\b(?:{_FORMAT}|{_PLATFORM}|{_EDITION})\b", re.IGNORECASE),
}
_EDITION_WORDS = re.compile(r"\b(?:edition|version|collection|the\s*complete\s*series|complete\s*series)\b", re.I)
_YEAR = re.compile(r"\b(19[2-9]\d|20[0-3]\d)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
_ROMAN_RX = re.compile(r"\b(" + "|".join(sorted(_ROMAN, key=len, reverse=True)) + r")\b")

# Products in the Video_Games category that are not games.
_ACCESSORY = re.compile(
    r"\b(controller|gamepad|joystick|headset|headphones?|charger|charging|cable|adapter|case|skin|sticker|decal"
    r"|stand|mount|grip|thumb\s*grips?|memory\s*card|hard\s*drive|ssd|power\s*supply|battery|batteries|keyboard"
    r"|mouse|mice|mousepad|mouse\s*pad|gift\s*card|membership|subscription|console(?!\s*game)|faceplate|cooling"
    r"|fan|remote|sensor|camera|steering\s*wheel|pedals|protector|screen\s*protector|carrying|travel\s*bag|dock"
    r"|amiibo|figure|poster|t-?shirt|hoodie|mug|replacement|repair|tool\s*kit|vr\s*headset|guitar\s*strap"
    r"|hdmi|splitter|usb|bluetooth|fight\s*stick|arcade\s*stick|neo\s*geo\s*mini|go\s*plus)\b",
    re.I,
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def extract_year(title: str) -> int | None:
    """Year from a bracketed segment, e.g. "Dune (1984)". Bare numbers ("2012") are titles."""
    for seg in _BRACKETS.findall(title or ""):
        m = _YEAR.search(seg)
        if m:
            return int(m.group(1))
    return None


def normalize_title(title: str, domain: str = "movie") -> str:
    """Normalise a product title to an entity key (format/platform/edition agnostic)."""
    if not isinstance(title, str):
        return ""
    t = _strip_accents(title).lower().replace("&", " and ")

    # Bracketed segments are almost always format/platform/edition noise: drop them entirely
    # unless they contain the only alphanumerics in the title.
    stripped = _BRACKETS.sub(" ", t)
    if _NON_ALNUM.sub("", stripped):
        t = stripped
    t = _TOKENS[domain].sub(" ", t)
    t = _EDITION_WORDS.sub(" ", t)
    t = t.replace("'", "")
    t = _NON_ALNUM.sub(" ", t).strip()
    # Sequel numbering: "Frozen II" == "Frozen 2". A lone "v"/"x" is only a numeral after a word.
    t = _ROMAN_RX.sub(lambda m: _ROMAN[m.group(1)] if m.start() > 0 else m.group(1), t)
    # Leading article does not change identity ("The Witcher 3" == "Witcher 3").
    t = re.sub(r"^(the|a|an)\s+", "", t)
    if domain == "game":
        # Budget re-release lines: "Ratchet & Clank (PS4) Hits", "Nintendo Selects: Lego City Undercover".
        t = re.sub(r"\s+hits$", "", re.sub(r"^nintendo\s+selects\s+", "", t))
    t = re.sub(r"\s+", " ", t)
    if not t:  # the whole title was "noise" (e.g. a film literally called "Unrated")
        t = _NON_ALNUM.sub(" ", _strip_accents(title).lower()).strip()
    return t


def is_accessory(title: str, categories: list[str] | None = None) -> bool:
    cats = " ".join(categories or []).lower()
    if "accessories" in cats or "consoles" in cats or "virtual reality" in cats:
        return True
    return bool(_ACCESSORY.search(title or ""))


def _stable_id(domain: str, key: str) -> str:
    digest = hashlib.sha1(f"{domain}|{key}".encode()).hexdigest()[:10]
    return f"{domain[0]}_{digest}"


def canonicalize(products: pd.DataFrame) -> pd.DataFrame:
    """Assign canonical_item_id to product rows.

    products columns: source_asin, domain, title (+ optional year)
    Returns products with norm_title, canonical_item_id.
    """
    df = products.copy()
    df["norm_title"] = [normalize_title(t, d) for t, d in zip(df["title"], df["domain"], strict=True)]
    df["title_year"] = df["title"].map(extract_year)
    # Year used to split same-titled groups: an explicit "(1984)" in the title always counts. For
    # games the metadata release year is reliable too (remakes like Ratchet & Clank 2002/2016 share
    # a title); for movies it is usually the DVD release date, so it is ignored.
    meta_year = df["year"] if "year" in df else pd.Series(np.nan, index=df.index)
    use_meta = (df["domain"] == "game") & df["title_year"].isna()
    df["cluster_year"] = df["title_year"].where(~use_meta, meta_year)
    df["year_tol"] = np.where(df["title_year"].notna(), 1, GAME_YEAR_TOLERANCE)
    df = df[df["norm_title"].str.len() > 0].copy()

    df = df.reset_index(drop=True)
    canon = np.array([_stable_id(d, n) for d, n in zip(df["domain"], df["norm_title"], strict=True)], dtype=object)
    # Only groups with conflicting years need the (slow) per-group clustering.
    n_years = df.groupby(["domain", "norm_title"])["cluster_year"].transform("nunique")
    conflicted = df[n_years > 1]
    for (domain, norm), idx in conflicted.groupby(["domain", "norm_title"]).groups.items():
        idx = np.asarray(idx)
        years = df.loc[idx, "cluster_year"]
        known = years.dropna().astype(int)
        tol = int(df.loc[idx, "year_tol"].min())
        # Cluster distinct years with a tolerance (re-releases / GOTY editions drift by a year or two).
        clusters: list[list[int]] = []
        for y in sorted(known.unique()):
            if clusters and y - clusters[-1][-1] <= tol:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        if len(clusters) == 1:
            continue
        sizes = [int(known.isin(c).sum()) for c in clusters]
        largest = clusters[int(np.argmax(sizes))]
        for i, y in zip(idx, years, strict=True):
            cluster = largest if pd.isna(y) else next(c for c in clusters if int(y) in c)
            canon[i] = _stable_id(domain, f"{norm}|{cluster[0]}")
    df["canonical_item_id"] = canon
    return df


def audit_sample(mapping: pd.DataFrame, n_per_stratum: int = 25, seed: int = 0) -> pd.DataFrame:
    """Stratified sample of canonical groups for manual precision audit.

    Strata = domain x {merged groups (>1 product), singletons}. Reviewers fill `is_correct`
    (1 = every product in the group is the same entertainment entity).
    """
    groups = (
        mapping.groupby(["domain", "canonical_item_id"])
        .agg(n_products=("source_asin", "size"), titles=("title", lambda s: " || ".join(s.astype(str).head(8))))
        .reset_index()
    )
    groups["stratum"] = np.where(groups["n_products"] > 1, "merged", "singleton")
    parts = []
    for _, g in groups.groupby(["domain", "stratum"]):
        parts.append(g.sample(min(n_per_stratum, len(g)), random_state=seed))
    out = pd.concat(parts, ignore_index=True) if parts else groups.head(0)
    out["is_correct"] = ""
    return out


def audit_precision(labeled: pd.DataFrame) -> pd.DataFrame:
    """Precision per stratum from a labelled audit sample (is_correct in {0,1})."""
    lab = labeled[labeled["is_correct"].astype(str).isin(["0", "1"])].copy()
    lab["is_correct"] = lab["is_correct"].astype(int)
    return lab.groupby(["domain", "stratum"])["is_correct"].agg(["mean", "count"]).rename(columns={"mean": "precision"})
