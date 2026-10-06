"""Deterministic synthetic catalog + interactions with *planted* cross-domain taste.

Users have a latent preference over shared themes; items in both domains carry themes. The
fixture lets tests, CI and offline demos run end-to-end without downloading Amazon data, and its
planted structure gives golden expectations (a cyberpunk-movie lover should get cyberpunk games).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossverse.data.schema import validate_interactions, validate_items

THEME_WORDS = {
    "cyberpunk": ["Neon", "Chrome", "Circuit", "Augment", "Grid"],
    "sci-fi": ["Star", "Nova", "Orbit", "Quantum", "Nebula"],
    "fantasy": ["Dragon", "Rune", "Elder", "Myth", "Crown"],
    "horror": ["Dread", "Hollow", "Crypt", "Shade", "Grave"],
    "crime": ["Heist", "Syndicate", "Vice", "Cartel", "Alibi"],
    "war": ["Front", "Siege", "Battalion", "Trench", "Valor"],
    "comedy": ["Goofy", "Prank", "Jolly", "Wacky", "Giggle"],
    "romance": ["Heart", "Kiss", "Promise", "Blossom", "Sonnet"],
    "sports": ["Goal", "Slam", "Champion", "League", "Victory"],
    "racing": ["Turbo", "Drift", "Nitro", "Apex", "Throttle"],
    "western": ["Outlaw", "Dust", "Frontier", "Saloon", "Bounty"],
    "mystery": ["Riddle", "Cipher", "Clue", "Secret", "Enigma"],
}
THEME_TEXT = {
    "cyberpunk": "a cyberpunk neon-lit dystopian city of implants and megacorps",
    "sci-fi": "science fiction among alien worlds and starships",
    "fantasy": "a fantasy realm of magic, wizards and dragons",
    "horror": "horror, haunted halls and demonic nightmares",
    "crime": "crime, gangsters and a daring heist",
    "war": "war on the battlefield with soldiers and the military",
    "comedy": "a hilarious comedy full of laughs",
    "romance": "a romantic love story",
    "sports": "sports, a championship football league",
    "racing": "street racing and motorsport",
    "western": "a western of cowboys and outlaws on the wild west frontier",
    "mystery": "a mystery with clues and secrets for a detective",
}
NOUNS = ["Protocol", "Legacy", "Rising", "Chronicles", "Requiem", "Horizon", "Odyssey", "Reckoning", "Saga", "Code"]


def generate(n_users: int = 3000, n_movies: int = 400, n_games: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    themes = list(THEME_WORDS)
    T = len(themes)

    rows = []
    item_theme = []
    used_titles: set[str] = set()
    for domain, n in (("movie", n_movies), ("game", n_games)):
        for i in range(n):
            k = rng.choice([1, 2, 2, 3])
            tidx = rng.choice(T, size=k, replace=False)
            vec = np.zeros(T)
            vec[tidx] = rng.dirichlet(np.ones(k) * 2)
            main = themes[tidx[0]]
            title = f"{rng.choice(THEME_WORDS[main])} {rng.choice(NOUNS)}"
            if domain == "game":
                title += f" {rng.integers(1, 5)}" if rng.random() < 0.3 else ""
            base = title
            j = 2
            while title in used_titles:
                title = f"{base} {'II' if j == 2 else j}"
                j += 1
            used_titles.add(title)
            kind = "film" if domain == "movie" else "game"
            text = f"{title}. A {kind} about " + "; ".join(THEME_TEXT[themes[t]] for t in tidx) + "."
            rows.append(
                {
                    "item_id": f"{domain[0]}_{i:05d}",
                    "domain": domain,
                    "title": title,
                    "text": text,
                    "genres": [themes[t] for t in tidx],
                    "themes": [themes[t] for t in tidx],
                    "creator": f"Studio {rng.integers(1, 40)}",
                    "year": int(rng.integers(1980, 2024)),
                    "n_products": 1,
                }
            )
            item_theme.append(vec)
    items = pd.DataFrame(rows)
    item_vec = np.vstack(item_theme)
    quality = rng.normal(0, 0.6, len(items))
    is_movie = (items["domain"] == "movie").to_numpy()

    inter = []
    t0 = 1_400_000_000_000
    for u in range(n_users):
        pref = rng.dirichlet(np.ones(T) * 0.3)
        bridge = rng.random() < 0.65
        domains = ["movie", "game"] if bridge else [rng.choice(["movie", "game"])]
        t = t0 + int(rng.integers(0, 300_000_000_000))
        for d in domains:
            mask = is_movie if d == "movie" else ~is_movie
            idx = np.flatnonzero(mask)
            affinity = item_vec[idx] @ pref
            logits = 6.0 * affinity + quality[idx]
            p = np.exp(logits - np.max(logits))
            p /= p.sum()
            n = int(np.clip(rng.geometric(1 / 12), 4, 60))
            chosen = rng.choice(idx, size=min(n, len(idx)), replace=False, p=p)
            for it in chosen:
                aff = item_vec[it] @ pref
                score = 2.2 + 9.0 * aff + quality[it] + rng.normal(0, 0.5)
                rating = float(np.clip(np.round(score), 1, 5))
                t += int(rng.integers(100_000_000, 5_000_000_000))
                inter.append((f"u{u}", items.at[it, "item_id"], d, rating, t))
    interactions = pd.DataFrame(inter, columns=["user_id", "item_id", "domain", "rating", "timestamp"])
    interactions = interactions.drop_duplicates(["user_id", "item_id"])
    pos = interactions[interactions["rating"] >= 4].groupby("item_id").size()
    items["popularity"] = items["item_id"].map(pos).fillna(0).astype(int)
    stats = interactions.groupby("item_id")["rating"].agg(["mean", "count"])
    items["rating_mean"] = items["item_id"].map(stats["mean"]).fillna(3.0).astype(float)
    items["rating_count"] = items["item_id"].map(stats["count"]).fillna(0).astype(int)
    return interactions.reset_index(drop=True), items


def write_fixture(out_dir: Path, n_users: int = 3000, seed: int = 42) -> dict[str, object]:
    interactions, items = generate(n_users=n_users, seed=seed)
    validate_items(items)
    validate_interactions(interactions, items)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    report = {
        "source": "synthetic",
        "final_interactions": len(interactions),
        "final_users": int(interactions["user_id"].nunique()),
        "final_items": items["domain"].value_counts().to_dict(),
    }
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    return report
