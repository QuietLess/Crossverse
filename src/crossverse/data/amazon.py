"""Amazon Reviews 2023 ingestion -> canonical interactions + item catalog.

Only Movies_and_TV and Video_Games are touched. Rating files are the official "rating_only"
benchmark CSVs (user_id, parent_asin, rating, timestamp); metadata is streamed line by line and
only rows for items that survive user filtering are parsed, so the 600MB+ metadata never needs to
sit in memory.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from crossverse.config import DataConfig
from crossverse.data import canonical
from crossverse.data.schema import validate_interactions, validate_items
from crossverse.features.themes import item_themes

log = logging.getLogger(__name__)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
CATEGORIES = {"movie": "Movies_and_TV", "game": "Video_Games"}
NON_GENRE_CATEGORIES = {
    "movies & tv", "movies", "tv", "prime video", "video games", "games", "pc", "mac", "featured categories",
    "genre for featured categories", "boxed sets", "independently distributed", "studio specials",
    "blu-ray", "4k ultra hd", "all titles", "digital games", "online game services", "legacy systems",
}


def rating_url(category: str, core: str = "0core") -> str:
    return f"{BASE_URL}/benchmark/{core}/rating_only/{category}.csv.gz"


def meta_url(category: str) -> str:
    return f"{BASE_URL}/raw/meta_categories/meta_{category}.jsonl.gz"


def download(raw_dir: Path, core: str = "0core", force: bool = False) -> list[Path]:
    """Download rating + metadata files for both domains (streamed, resumable by file)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for category in CATEGORIES.values():
        for url in (rating_url(category, core), meta_url(category)):
            dest = raw_dir / url.rsplit("/", 1)[-1]
            out.append(dest)
            if dest.exists() and not force:
                log.info("exists: %s", dest.name)
                continue
            log.info("downloading %s", url)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
    return out


# --------------------------------------------------------------------------------------------
# Ratings
# --------------------------------------------------------------------------------------------


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for domain, category in CATEGORIES.items():
        path = raw_dir / f"{category}.csv.gz"
        log.info("reading %s", path.name)
        df = pd.read_csv(
            path,
            dtype={"user_id": "string", "parent_asin": "string", "rating": "float32", "timestamp": "int64"},
            engine="pyarrow",
        )
        df["domain"] = domain
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["domain"] = df["domain"].astype("category")
    # A user can review the same product twice; keep the most recent opinion.
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")
    return df


def select_users(ratings: pd.DataFrame, cfg: DataConfig) -> tuple[pd.Index, pd.Index]:
    """Return (bridge_users, single_domain_users)."""
    pos = ratings[ratings["rating"] >= cfg.positive_threshold]
    counts = pos.groupby(["user_id", "domain"], observed=True).size().unstack(fill_value=0)
    for d in CATEGORIES:
        if d not in counts:
            counts[d] = 0
    bridge_mask = (counts["movie"] >= cfg.min_bridge_positives) & (counts["game"] >= cfg.min_bridge_positives)
    bridge = counts.index[bridge_mask]
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_bridge_users and len(bridge) > cfg.max_bridge_users:
        bridge = pd.Index(rng.choice(bridge.to_numpy(), cfg.max_bridge_users, replace=False))
    singles: list[np.ndarray] = []
    for d in CATEGORIES:
        other = "game" if d == "movie" else "movie"
        mask = (counts[d] >= cfg.single_domain_min_positives) & (counts[other] == 0)
        pool = counts.index[mask].to_numpy()
        n = min(cfg.single_domain_users_per_domain, len(pool))
        if n:
            singles.append(rng.choice(pool, n, replace=False))
    single = pd.Index(np.concatenate(singles)) if singles else pd.Index([])
    return bridge, single


# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------

_YEAR_FEATURE = re.compile(r"^(19[2-9]\d|20[0-3]\d)$")
_ANY_YEAR = re.compile(r"\b(19[2-9]\d|20[0-3]\d)\b")


def _year_from_meta(rec: dict) -> int | None:
    for f in rec.get("features") or []:
        if isinstance(f, str) and _YEAR_FEATURE.match(f.strip()):
            return int(f.strip())
    details = rec.get("details") or {}
    for key in ("Release date", "Release Date", "Original Release Date", "Date First Available"):
        v = details.get(key)
        if isinstance(v, str):
            m = _ANY_YEAR.search(v)
            if m:
                return int(m.group(1))
    return canonical.extract_year(rec.get("title") or "")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_as_text(v) for v in value.values())
    return str(value)


_HARDWARE = re.compile(
    r"\b(hdmi|splitter|cable|charger|charging|controller|headset|adapter|memory card|gift card|power supply|membership|subscription|console bundle|fight ?stick|arcade stick|neo ?geo mini|go plus|light gun)\b", re.I
)


# main_category is noisy for real games too ("Crash Team Racing" is filed under Books), so only the
# media categories are decisive; disc-format metadata catches the rest.
NON_GAME_MAIN_CATEGORIES = {"Movies & TV", "Digital Music", "Prime Video"}
_DISC_MEDIA = re.compile(r"Format:\s*(DVD|Blu-ray|Audio CD|VHS|Prime Video|MP3)|\(Actor", re.I)


def _is_misfiled_media(rec: dict) -> bool:
    """Movies/TV/music products that sellers listed under Video_Games (e.g. 'Stargate Atlantis: The
    Complete Series' with categories ['Video Games', 'PC', 'Games']). They create fake movie<->game links."""
    if (rec.get("main_category") or "") in NON_GAME_MAIN_CATEGORIES:
        return True
    if _DISC_MEDIA.search(str(rec.get("store") or "")):
        return True
    details = rec.get("details") or {}
    if str(details.get("Type of item", "")).lower() in {"blu-ray", "dvd", "vhs"}:
        return True
    return "MPAA rating" in details


def _is_game(title: str, categories: list[str], rec: dict | None = None) -> bool:
    """Video_Games also sells controllers, consoles, gift cards and misfiled DVDs; keep actual games only."""
    if rec is not None and _is_misfiled_media(rec):
        return False
    if any(k in c.lower() for c in categories for k in ("accessor", "consoles", "virtual reality", "hardware")):
        return False
    if "Games" in categories:
        # Trust Amazon's taxonomy: broad title keywords ("Mount & Blade", "Phoenix Wright: Case 1")
        # would misfire on real games, so only unmistakable hardware words are checked.
        return not _HARDWARE.search(title)
    # Sparse taxonomy ("Video Games > Xbox One", "Legacy Systems"): fall back to title heuristics.
    return not canonical.is_accessory(title)


def iter_metadata(path: Path, wanted: set[str]) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            # Cheap pre-filter on the ASIN before paying for json.loads.
            i = line.find('"parent_asin": "')
            if i < 0:
                continue
            asin = line[i + 16 : i + 26]
            if asin not in wanted:
                continue
            yield json.loads(line)


def load_metadata(raw_dir: Path, wanted: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for domain, category in CATEGORIES.items():
        path = raw_dir / f"meta_{category}.jsonl.gz"
        log.info("streaming %s (%d wanted)", path.name, len(wanted[domain]))
        for rec in iter_metadata(path, wanted[domain]):
            title = (rec.get("title") or "").strip()
            if not title:
                continue
            cats = [c for c in (rec.get("categories") or []) if isinstance(c, str)]
            if domain == "game" and not _is_game(title, cats, rec):
                continue
            details = rec.get("details") or {}
            directors = details.get("Directors")
            creator = (
                details.get("Studio")
                or details.get("Manufacturer")
                or rec.get("store")
                or (", ".join(directors[:2]) if isinstance(directors, list) else "")
            )
            if isinstance(creator, str) and creator.startswith("Format:"):
                creator = ", ".join(directors[:2]) if isinstance(directors, list) else ""
            genres = [c for c in cats if c.lower() not in NON_GENRE_CATEGORIES]
            if domain == "game":
                genres = [g for g in genres if not re.search(r"playstation|xbox|nintendo|wii|pc|mac|sega|ds\b", g, re.I)]
            rows.append(
                {
                    "source_asin": rec["parent_asin"],
                    "domain": domain,
                    "title": title,
                    "description": _as_text(rec.get("description"))[:3000],
                    "features": _as_text(rec.get("features"))[:1000],
                    "genres": genres[:6],
                    "creator": str(creator or "")[:120],
                    "year": _year_from_meta(rec),
                    "rating_number": rec.get("rating_number") or 0,
                    "main_category": rec.get("main_category") or "",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def build_catalog(meta: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    m = meta.merge(mapping[["source_asin", "canonical_item_id"]], on="source_asin")
    m = m.sort_values("rating_number", ascending=False)
    rows = []
    for cid, g in m.groupby("canonical_item_id", sort=False):
        head = g.iloc[0]
        genres = list(dict.fromkeys(x for gs in g["genres"] for x in gs))[:6]
        desc = max(g["description"], key=len)
        title = canonical.display_title(head["title"], head["domain"])
        text = " ".join([title, " ".join(genres), desc, head["features"]])
        years = g["year"].dropna()
        rows.append(
            {
                "item_id": cid,
                "domain": head["domain"],
                "title": title,
                "text": text[:4000],
                "genres": genres,
                "themes": item_themes(genres, text),
                "creator": head["creator"],
                "year": int(years.min()) if len(years) else -1,
                "n_products": len(g),
            }
        )
    catalog = pd.DataFrame(rows)
    catalog["title"] = canonical.display_titles(catalog)
    return catalog


def build_dataset(cfg: DataConfig | None = None) -> dict[str, object]:
    cfg = cfg or DataConfig()
    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(cfg.raw_dir)
    report: dict[str, object] = {"raw_interactions": int(len(ratings))}
    users_per_domain = ratings.groupby("domain", observed=True)["user_id"].nunique()
    report["raw_users"] = {d: int(n) for d, n in users_per_domain.items()}
    overlap = set(ratings.loc[ratings["domain"] == "movie", "user_id"]) & set(
        ratings.loc[ratings["domain"] == "game", "user_id"]
    )
    report["users_in_both_domains"] = len(overlap)
    del overlap

    bridge, single = select_users(ratings, cfg)
    report["bridge_users_selected"] = int(len(bridge))
    report["single_domain_users_selected"] = int(len(single))
    keep_users = bridge.union(single)
    r = ratings[ratings["user_id"].isin(keep_users)].copy()
    del ratings
    log.info("kept %d interactions from %d users", len(r), len(keep_users))

    wanted = {d: set(r.loc[r["domain"] == d, "parent_asin"].astype(str)) for d in CATEGORIES}
    meta = load_metadata(cfg.raw_dir, wanted)
    report["products_with_metadata"] = int(len(meta))
    mapping = canonical.canonicalize(meta[["source_asin", "domain", "title", "year"]])
    report["canonical_items"] = int(mapping["canonical_item_id"].nunique())
    report["products_merged_away"] = int(len(mapping) - mapping["canonical_item_id"].nunique())

    r = r.merge(mapping[["source_asin", "canonical_item_id"]], left_on="parent_asin", right_on="source_asin")
    r = (
        r.groupby(["user_id", "canonical_item_id"], observed=True)
        .agg(domain=("domain", "first"), rating=("rating", "max"), timestamp=("timestamp", "min"))
        .reset_index()
        .rename(columns={"canonical_item_id": "item_id"})
    )
    r["domain"] = r["domain"].astype(str)

    # Iterative pruning: items need enough positives, users need enough history.
    for _ in range(3):
        pos = r[r["rating"] >= cfg.positive_threshold]
        item_ok = pos.groupby("item_id").size()
        item_ok = item_ok[item_ok >= cfg.min_item_interactions].index
        r = r[r["item_id"].isin(item_ok)]
        pos = r[r["rating"] >= cfg.positive_threshold]
        upos = pos.groupby("user_id").size()
        r = r[r["user_id"].isin(upos[upos >= 3].index)]

    catalog = build_catalog(meta, mapping)
    catalog = catalog[catalog["item_id"].isin(r["item_id"].unique())].reset_index(drop=True)
    pop = r[r["rating"] >= cfg.positive_threshold].groupby("item_id").size()
    catalog["popularity"] = catalog["item_id"].map(pop).fillna(0).astype(int)
    stats = r.groupby("item_id")["rating"].agg(["mean", "count"])
    catalog["rating_mean"] = catalog["item_id"].map(stats["mean"]).astype(float)
    catalog["rating_count"] = catalog["item_id"].map(stats["count"]).astype(int)

    # Stable pseudonymous user ids: the same Amazon user gets the same id in every rebuild, so
    # splits and per-user results stay comparable across dataset versions.
    uniq = r["user_id"].astype(str).unique()
    pseudo = {u: "u" + hashlib.sha1(u.encode()).hexdigest()[:12] for u in uniq}
    r = r.assign(user_id=r["user_id"].astype(str).map(pseudo))
    r = r[["user_id", "item_id", "domain", "rating", "timestamp"]].reset_index(drop=True)
    r["rating"] = r["rating"].astype(float)

    validate_items(catalog)
    validate_interactions(r, catalog)

    mapping = mapping[mapping["canonical_item_id"].isin(catalog["item_id"])]
    r.to_parquet(out_dir / "interactions.parquet", index=False)
    catalog.to_parquet(out_dir / "items.parquet", index=False)
    mapping[["source_asin", "domain", "title", "norm_title", "canonical_item_id"]].to_parquet(
        out_dir / "asin_map.parquet", index=False
    )
    canonical.audit_sample(mapping, seed=cfg.seed).to_csv(out_dir / "canonical_audit_sample.csv", index=False)

    final_pos = r[r["rating"] >= cfg.positive_threshold]
    dom_counts = final_pos.groupby(["user_id", "domain"]).size().unstack(fill_value=0)
    report.update(
        {
            "final_interactions": int(len(r)),
            "final_users": int(r["user_id"].nunique()),
            "final_bridge_users": int(((dom_counts.get("movie", 0) >= 1) & (dom_counts.get("game", 0) >= 1)).sum()),
            "final_items": {d: int(n) for d, n in catalog["domain"].value_counts().items()},
            "positive_rate": float((r["rating"] >= cfg.positive_threshold).mean()),
            "items_with_themes": float((catalog["themes"].map(len) > 0).mean()),
            "config": {k: str(v) for k, v in asdict(cfg).items()},
        }
    )
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    log.info("dataset written to %s", out_dir)
    return report
