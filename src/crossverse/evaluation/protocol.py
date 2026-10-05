"""Leakage-safe splitting and evaluation tasks (blueprint section 6).

Two mechanisms, both chronological / user-level, never random row splits:

* Regular users: per-user chronological split. The last `test_frac` of a user's interactions
  are test, the preceding `val_frac` validation, the rest train. Within-domain, mixed and
  cold-start tasks are built from these users.
* Cross-domain holdout users: disjoint groups of bridge users whose *entire* target-domain
  history is removed from training. movie_to_game users keep only their movies in train and must
  be predicted their games from movies alone (and vice versa). This is the honest version of
  "can one domain predict the other?": the model has never seen any of these users' games.

Validation tasks (built from val periods / val holdout groups) are used for ranker training and
tuning; test tasks are touched only for the final benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crossverse.config import SplitConfig
from crossverse.retrieval.base import Catalog, History, rating_to_weight

TASKS = ("within_movie", "within_game", "movie_to_game", "game_to_movie", "mixed", "cold_start")
TASK_TARGET = {
    "within_movie": "movie",
    "within_game": "game",
    "movie_to_game": "game",
    "game_to_movie": "movie",
    "mixed": None,
    "cold_start": None,
}


@dataclass
class EvalCase:
    user_id: str
    history: History
    relevant: set[int]
    exclude: np.ndarray
    segments: tuple[str, ...] = ()
    preferences: list[str] = field(default_factory=list)


@dataclass
class EvalTask:
    name: str
    target_domain: str | None
    cases: list[EvalCase]


@dataclass
class Split:
    train: pd.DataFrame
    val_tasks: dict[str, EvalTask]
    test_tasks: dict[str, EvalTask]
    holdout_users: dict[str, list[str]]
    summary: dict[str, int]


def _history(catalog: Catalog, df: pd.DataFrame) -> History:
    """df rows are already time-ordered and carry precomputed `item_idx` / `w` columns."""
    return History(df["item_idx"].to_numpy(dtype=np.int64), df["w"].to_numpy(dtype=np.float32))


def _positives(catalog: Catalog, df: pd.DataFrame, threshold: float) -> set[int]:
    return set(df.loc[df["rating"] >= threshold, "item_idx"].tolist())


def _top_themes(catalog: Catalog, hist: History, n: int = 3) -> list[str]:
    c: Counter[str] = Counter()
    themes = catalog.items["themes"]
    for i in hist.positives().idx:
        c.update(themes.iat[int(i)][:3])
    return [t for t, _ in c.most_common(n)]


def make_split(interactions: pd.DataFrame, catalog: Catalog, cfg: SplitConfig | None = None,
               positive_threshold: float = 4.0, max_cases: int = 3000) -> Split:
    cfg = cfg or SplitConfig()
    rng = np.random.default_rng(cfg.seed)
    df = interactions[interactions["item_id"].isin(catalog.index)].sort_values(["user_id", "timestamp"])
    # Vectorised once: per-user Series.map(dict) calls dominate runtime on 700k+ rows otherwise.
    df = df.assign(item_idx=df["item_id"].map(catalog.index).astype(np.int64),
                   w=df["rating"].map(rating_to_weight).astype(np.float32))
    pos = df[df["rating"] >= positive_threshold]
    pc = pos.groupby(["user_id", "domain"]).size().unstack(fill_value=0)
    for d in ("movie", "game"):
        if d not in pc:
            pc[d] = 0
    bridge_all = set(pc.index[(pc["movie"] >= 1) & (pc["game"] >= 1)])
    eligible = pc.index[(pc["movie"] >= cfg.min_history) & (pc["game"] >= cfg.min_history)].to_numpy()
    rng.shuffle(eligible)
    n_group = max(1, int(len(eligible) * cfg.cross_holdout_frac))
    groups = {
        "movie_to_game_test": list(eligible[0:n_group]),
        "game_to_movie_test": list(eligible[n_group : 2 * n_group]),
        "movie_to_game_val": list(eligible[2 * n_group : 3 * n_group]),
        "game_to_movie_val": list(eligible[3 * n_group : 4 * n_group]),
    }
    hidden_domain = {}
    for g, users in groups.items():
        for u in users:
            hidden_domain[u] = "game" if g.startswith("movie_to_game") else "movie"

    # ---- per-user chronological split for regular users -----------------------------------
    is_holdout = df["user_id"].isin(hidden_domain)
    reg = df[~is_holdout].copy()
    reg["rank"] = reg.groupby("user_id").cumcount()
    reg["n"] = reg.groupby("user_id")["item_id"].transform("size")
    n_test = np.where(reg["n"] >= 5, np.maximum(1, np.round(reg["n"] * cfg.test_frac)), 0)
    n_val = np.where(reg["n"] >= 5, np.maximum(1, np.round(reg["n"] * cfg.val_frac)), 0)
    reg["part"] = np.select(
        [reg["rank"] >= reg["n"] - n_test, reg["rank"] >= reg["n"] - n_test - n_val], ["test", "val"], "train"
    )

    hold = df[is_holdout].copy()
    hold["hidden"] = hold["user_id"].map(hidden_domain)
    hold_kept = hold[hold["domain"] != hold["hidden"]]
    hold_hidden = hold[hold["domain"] == hold["hidden"]]

    train = pd.concat([reg[reg["part"] == "train"], hold_kept], ignore_index=True)[interactions.columns]

    heavy_cut = df.groupby("user_id").size().quantile(0.75)
    seg_cache: dict[str, tuple[str, ...]] = {}

    def segments(user: str, hist_len: int) -> tuple[str, ...]:
        if user not in seg_cache:
            s = ["all"]
            s.append("heavy" if hist_len >= heavy_cut else ("sparse" if hist_len <= 5 else "medium"))
            if user in bridge_all:
                s.append("bridge")
            seg_cache[user] = tuple(s)
        return seg_cache[user]

    def build_regular(target_part: str) -> dict[str, EvalTask]:
        hist_parts = ["train"] if target_part == "val" else ["train", "val"]
        h_df = reg[reg["part"].isin(hist_parts)]
        t_df = reg[reg["part"] == target_part]
        h_groups = dict(tuple(h_df.groupby("user_id")))
        tasks = {name: EvalTask(name, TASK_TARGET[name], []) for name in ("within_movie", "within_game", "mixed", "cold_start")}
        users = t_df["user_id"].unique()
        rng.shuffle(users)
        t_groups = dict(tuple(t_df[t_df["user_id"].isin(users)].groupby("user_id")))
        for u in users:
            if all(len(t.cases) >= max_cases for t in tasks.values()):
                break
            if u not in h_groups:
                continue
            hd, td = h_groups[u], t_groups[u]
            full_hist = _history(catalog, hd)
            for dom, name in (("movie", "within_movie"), ("game", "within_game")):
                if len(tasks[name].cases) >= max_cases:
                    continue
                rel = _positives(catalog, td[td["domain"] == dom], positive_threshold)
                h = full_hist.restrict(catalog, dom)
                rel -= set(h.idx.tolist())
                if rel and len(h.positives()):
                    tasks[name].cases.append(EvalCase(u, h, rel, h.idx, segments(u, len(full_hist))))
            rel_all = _positives(catalog, td, positive_threshold) - set(full_hist.idx.tolist())
            if not rel_all:
                continue
            doms = set(catalog.domain[full_hist.positives().idx].tolist())
            if len(doms) == 2 and len(tasks["mixed"].cases) < max_cases:
                tasks["mixed"].cases.append(EvalCase(u, full_hist, rel_all, full_hist.idx, segments(u, len(full_hist))))
            if len(tasks["cold_start"].cases) < max_cases:
                prefs = _top_themes(catalog, full_hist)
                if prefs:
                    tasks["cold_start"].cases.append(
                        EvalCase(u, History.empty(), rel_all, full_hist.idx, segments(u, len(full_hist)), prefs)
                    )
        return tasks

    def build_cross(group: str) -> EvalTask:
        name = "movie_to_game" if group.startswith("movie_to_game") else "game_to_movie"
        src = "movie" if name == "movie_to_game" else "game"
        task = EvalTask(name, TASK_TARGET[name], [])
        users = set(groups[group])
        kept = dict(tuple(hold_kept[hold_kept["user_id"].isin(users)].groupby("user_id")))
        hidden = dict(tuple(hold_hidden[hold_hidden["user_id"].isin(users)].groupby("user_id")))
        for u in groups[group][:max_cases]:
            if u not in kept or u not in hidden:
                continue
            h = _history(catalog, kept[u]).restrict(catalog, src)
            rel = _positives(catalog, hidden[u], positive_threshold)
            if rel and len(h.positives()):
                task.cases.append(EvalCase(u, h, rel, np.zeros(0, dtype=np.int64), segments(u, len(h))))
        return task

    val_tasks = build_regular("val")
    val_tasks["movie_to_game"] = build_cross("movie_to_game_val")
    val_tasks["game_to_movie"] = build_cross("game_to_movie_val")
    test_tasks = build_regular("test")
    test_tasks["movie_to_game"] = build_cross("movie_to_game_test")
    test_tasks["game_to_movie"] = build_cross("game_to_movie_test")

    summary = {
        "train_interactions": int(len(train)),
        "train_users": int(train["user_id"].nunique()),
        "bridge_users": len(bridge_all),
        **{f"holdout_{g}": len(u) for g, u in groups.items()},
        **{f"val_{k}": len(t.cases) for k, t in val_tasks.items()},
        **{f"test_{k}": len(t.cases) for k, t in test_tasks.items()},
    }
    return Split(train, val_tasks, test_tasks, groups, summary)


def assert_no_leakage(split: Split, interactions: pd.DataFrame) -> None:
    """Hidden-domain interactions of cross-domain holdout users must never be in training."""
    train_keys = set(zip(split.train["user_id"], split.train["domain"], strict=True))
    for group, users in split.holdout_users.items():
        hidden = "game" if group.startswith("movie_to_game") else "movie"
        for u in users:
            if (u, hidden) in train_keys:
                raise AssertionError(f"leakage: {u} has {hidden} interactions in train ({group})")
    # Targets must never be visible in the input history.
    for task in split.test_tasks.values():
        for case in task.cases:
            if case.relevant & set(case.history.idx.tolist()):
                raise AssertionError(f"leakage: target items in history for {case.user_id} ({task.name})")
