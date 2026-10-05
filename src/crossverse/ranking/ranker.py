"""Stage B: LightGBM LambdaRank over multi-source candidates + post-ranking diversity rules."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from crossverse.evaluation.protocol import EvalTask
from crossverse.ranking.candidates import CandidateGenerator
from crossverse.ranking.features import FeatureBuilder

log = logging.getLogger(__name__)


def build_ranking_dataset(tasks: dict[str, EvalTask], generator: CandidateGenerator, features: FeatureBuilder,
                          queries_per_task: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Turn validation tasks into (X, y, group) for learning-to-rank.

    Queries whose candidates contain no positive are dropped: they carry no LambdaRank gradient.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, groups = [], [], []
    for name, task in tasks.items():
        cases = list(task.cases)
        rng.shuffle(cases)
        used = 0
        for case in cases:
            if used >= queries_per_task:
                break
            cands = generator.generate(case.history, task.target_domain, case.exclude, case.preferences)
            y = np.array([int(i) in case.relevant for i in cands.idx], dtype=np.int32)
            if y.sum() == 0:
                continue
            Xs.append(features.build(case.history, cands, case.preferences))
            ys.append(y)
            groups.append(len(y))
            used += 1
        log.info("ranker data: %s -> %d queries", name, used)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.asarray(groups)


_RANK_WORKER: dict = {}


def _init_rank_worker(engine_path: str, split_path: str) -> None:
    import pickle

    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    with open(engine_path, "rb") as fh:
        _RANK_WORKER["engine"] = pickle.load(fh)
    with open(split_path, "rb") as fh:
        _RANK_WORKER["tasks"] = pickle.load(fh).val_tasks


def _rank_chunk(job: tuple[str, list[int]]) -> tuple[str, list[tuple[int, pd.DataFrame, np.ndarray]]]:
    task_name, case_ids = job
    eng, task = _RANK_WORKER["engine"], _RANK_WORKER["tasks"][task_name]
    out = []
    for cid in case_ids:
        case = task.cases[cid]
        cands = eng.generator.generate(case.history, task.target_domain, case.exclude, case.preferences)
        y = np.array([int(i) in case.relevant for i in cands.idx], dtype=np.int32)
        if y.sum():
            out.append((cid, eng.features.build(case.history, cands, case.preferences), y))
    return task_name, out


def build_ranking_dataset_parallel(engine_path: str, split_path: str, tasks: dict[str, EvalTask],
                                   queries_per_task: int, seed: int = 0, n_jobs: int = 8,
                                   chunk: int = 100) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Parallel `build_ranking_dataset`: same shuffled order and per-task cap, cases fanned out."""
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(seed)
    orders = {}
    jobs = []
    for name, task in tasks.items():
        order = np.arange(len(task.cases))
        rng.shuffle(order)
        orders[name] = order
        for s in range(0, len(order), chunk):
            jobs.append((name, order[s : s + chunk].tolist()))
    results: dict[str, dict[int, tuple[pd.DataFrame, np.ndarray]]] = {n: {} for n in tasks}
    with ProcessPoolExecutor(n_jobs, initializer=_init_rank_worker, initargs=(engine_path, split_path)) as ex:
        for name, rows in ex.map(_rank_chunk, jobs):
            for cid, X, y in rows:
                results[name][cid] = (X, y)
    Xs, ys, groups = [], [], []
    for name in tasks:
        used = 0
        for case_id in orders[name].tolist():
            if used >= queries_per_task:
                break
            if case_id in results[name]:
                X, y = results[name][case_id]
                Xs.append(X)
                ys.append(y)
                groups.append(len(y))
                used += 1
        log.info("ranker data: %s -> %d queries", name, used)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.asarray(groups)


class LightGBMRanker:
    def __init__(self, params: dict | None = None, num_boost_round: int = 400, seed: int = 0):
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10, 20],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 30,
            "verbose": -1,
            "seed": seed,
            **(params or {}),
        }
        self.num_boost_round = num_boost_round

    def fit(self, X: pd.DataFrame, y: np.ndarray, group: np.ndarray, valid_frac: float = 0.15) -> LightGBMRanker:
        n_q = len(group)
        n_valid = max(1, int(n_q * valid_frac))
        bounds = np.concatenate([[0], np.cumsum(group)])
        # Queries arrive grouped by task; shuffle whole queries so the early-stopping set mixes all
        # tasks instead of being, e.g., only game_to_movie.
        perm = np.random.default_rng(self.params["seed"]).permutation(n_q)
        rows = np.concatenate([np.arange(bounds[q], bounds[q + 1]) for q in perm])
        X, y, group = X.iloc[rows].reset_index(drop=True), y[rows], group[perm]
        bounds = np.concatenate([[0], np.cumsum(group)])
        split = bounds[n_q - n_valid]
        dtrain = lgb.Dataset(X.iloc[:split], y[:split], group=group[: n_q - n_valid])
        dvalid = lgb.Dataset(X.iloc[split:], y[split:], group=group[n_q - n_valid :], reference=dtrain)
        self.booster_ = lgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        self.feature_names_ = list(X.columns)
        log.info("ranker best iteration %d", self.booster_.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster_.predict(X[self.feature_names_], num_iteration=self.booster_.best_iteration))

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.booster_.feature_importance("gain"), index=self.feature_names_).sort_values(ascending=False)


def is_cross_query(X: pd.DataFrame) -> bool:
    """A query whose history has no items in the candidates' domain (pure movie->game / game->movie)."""
    return bool((X["n_hist"] > 0).all() and (X["n_hist_same"] == 0).all() and (X["cold_start"] == 0).all())


class RoutedRanker:
    """Two LambdaRank models: a cross-domain specialist and a general model, routed per query.

    Cross-domain queries behave differently: no same-domain evidence exists, so behavioural
    features like item_knn barely fire and the transfer signals (co-preference, two-tower) must
    carry the ranking. A single model trained on the task mixture under-fits that regime.
    """

    def __init__(self, seed: int = 0, num_boost_round: int = 400):
        self.cross = LightGBMRanker(seed=seed, num_boost_round=num_boost_round)
        self.general = LightGBMRanker(seed=seed, num_boost_round=num_boost_round)

    def fit(self, X: pd.DataFrame, y: np.ndarray, group: np.ndarray) -> RoutedRanker:
        bounds = np.concatenate([[0], np.cumsum(group)])
        route = np.array([is_cross_query(X.iloc[bounds[q] : bounds[q + 1]]) for q in range(len(group))])
        for model, mask in ((self.cross, route), (self.general, ~route)):
            rows = np.concatenate([np.arange(bounds[q], bounds[q + 1]) for q in np.flatnonzero(mask)])
            model.fit(X.iloc[rows].reset_index(drop=True), y[rows], group[mask])
        self.n_queries_ = {"cross": int(route.sum()), "general": int((~route).sum())}
        self.feature_names_ = self.general.feature_names_
        log.info("routed ranker: %s", self.n_queries_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.cross if is_cross_query(X) else self.general).predict(X)

    def feature_importance(self) -> pd.Series:
        return (self.cross.feature_importance().add(self.general.feature_importance(), fill_value=0)
                .sort_values(ascending=False))


def diversify(idx: np.ndarray, scores: np.ndarray, embeddings: np.ndarray, k: int, lam: float = 0.15,
              creators: np.ndarray | None = None, creator_penalty: float = 0.05) -> np.ndarray:
    """Greedy MMR: trade relevance for novelty against already-selected items.

    Penalises near-duplicates (same franchise / sequels have near-identical metadata embeddings)
    and repeated creators. Returns indices into `idx`, in final order.
    """
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores)
    pool = order[: max(k * 5, 50)]
    s = scores[pool]
    rel = (s - s.min()) / (s.max() - s.min() + 1e-9)
    E = embeddings[idx[pool]]
    chosen: list[int] = []
    max_sim = np.zeros(len(pool))
    seen_creators: dict[str, int] = {}
    available = np.ones(len(pool), dtype=bool)
    for _ in range(min(k, len(pool))):
        penalty = lam * max_sim
        if creators is not None:
            penalty = penalty + creator_penalty * np.array(
                [seen_creators.get(c, 0) if c else 0 for c in creators[idx[pool]]]
            )
        val = np.where(available, rel - penalty, -np.inf)
        j = int(np.argmax(val))
        chosen.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, E @ E[j])
        if creators is not None and creators[idx[pool[j]]]:
            c = creators[idx[pool[j]]]
            seen_creators[c] = seen_creators.get(c, 0) + 1
    return pool[np.asarray(chosen)]
