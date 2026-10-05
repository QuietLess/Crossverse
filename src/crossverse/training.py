"""End-to-end training: split -> retrievers (train only) -> ranker (validation tasks) -> engine."""

from __future__ import annotations

import logging
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from crossverse.config import Settings
from crossverse.evaluation.protocol import Split, assert_no_leakage, make_split
from crossverse.ranking.ranker import (
    LightGBMRanker,
    RoutedRanker,
    build_ranking_dataset,
    build_ranking_dataset_parallel,
)
from crossverse.retrieval.base import Catalog, Retriever, TrainData
from crossverse.retrieval.baselines import (
    ALSRetriever,
    CoPreferenceRetriever,
    ItemKNNRetriever,
    PopularityRetriever,
)
from crossverse.retrieval.content import ContentRetriever
from crossverse.retrieval.two_tower import TwoTowerRetriever
from crossverse.serving.engine import CrossVerseEngine

log = logging.getLogger(__name__)


def fit_retrievers(data: TrainData, settings: Settings) -> tuple[dict[str, Retriever], dict[str, float]]:
    m = settings.model
    timings: dict[str, float] = {}

    def timed(name: str, r: Retriever) -> Retriever:
        t = time.perf_counter()
        r.fit(data)
        timings[name] = round(time.perf_counter() - t, 2)
        log.info("fitted %-11s in %6.1fs", name, timings[name])
        return r

    content = timed("content", ContentRetriever(m.content_dim, seed=m.seed))
    assert isinstance(content, ContentRetriever)
    als_kw = dict(factors=m.als_factors, iterations=m.als_iterations, regularization=m.als_regularization,
                  alpha=m.als_alpha, seed=m.seed)
    retrievers: dict[str, Retriever] = {
        "popularity": timed("popularity", PopularityRetriever()),
        "item_knn": timed("item_knn", ItemKNNRetriever(m.knn_neighbors, m.knn_shrink)),
        "als": timed("als", ALSRetriever(**als_kw)),  # type: ignore[arg-type]
        "als_movie": timed("als_movie", ALSRetriever(domain="movie", **als_kw)),  # type: ignore[arg-type]
        "als_game": timed("als_game", ALSRetriever(domain="game", **als_kw)),  # type: ignore[arg-type]
        "copref": timed("copref", CoPreferenceRetriever(m.copref_alpha)),
        "content": content,
        "two_tower": timed(
            "two_tower",
            TwoTowerRetriever(content.embeddings_, dim=m.tt_dim, epochs=m.tt_epochs, lr=m.tt_lr,
                              batch_size=m.tt_batch_size, cross_prob=m.tt_cross_prob,
                              max_history=m.tt_max_history, seed=m.seed),
        ),
    }
    return retrievers, timings


def train(interactions: pd.DataFrame, items: pd.DataFrame, settings: Settings, version: str = "dev",
          max_eval_cases: int = 2000, n_jobs: int = 1) -> tuple[CrossVerseEngine, Split, dict[str, Any]]:
    t0 = time.perf_counter()
    catalog = Catalog(items)
    split = make_split(interactions, catalog, settings.split, settings.data.positive_threshold, max_eval_cases)
    assert_no_leakage(split, interactions)
    log.info("split: %s", split.summary)

    data = TrainData.build(catalog, split.train, settings.data.positive_threshold)
    retrievers, timings = fit_retrievers(data, settings)

    base = CrossVerseEngine(catalog, retrievers, None, settings.model.candidates_per_source,
                            settings.model.diversity_lambda, version)
    t = time.perf_counter()
    if n_jobs > 1:
        # Workers need the retrievers and validation tasks on disk (spawn-based multiprocessing).
        with tempfile.TemporaryDirectory(dir=settings.serving.artifacts_dir) as tmp:
            base_path, split_path = Path(tmp) / "base.pkl", Path(tmp) / "split.pkl"
            base.save(base_path)
            with open(split_path, "wb") as fh:
                pickle.dump(split, fh, protocol=pickle.HIGHEST_PROTOCOL)
            X, y, groups = build_ranking_dataset_parallel(str(base_path), str(split_path), split.val_tasks,
                                                          settings.model.ranker_queries_per_task,
                                                          settings.model.seed, n_jobs)
    else:
        X, y, groups = build_ranking_dataset(split.val_tasks, base.generator, base.features,
                                             settings.model.ranker_queries_per_task, settings.model.seed)
    ranker: LightGBMRanker | RoutedRanker
    if settings.model.routed_ranker:
        ranker = RoutedRanker(seed=settings.model.seed).fit(X, y, groups)
    else:
        ranker = LightGBMRanker(seed=settings.model.seed).fit(X, y, groups)
    timings["ranker"] = round(time.perf_counter() - t, 2)

    engine = CrossVerseEngine(catalog, retrievers, ranker, settings.model.candidates_per_source,
                              settings.model.diversity_lambda, version)
    report = {
        "version": version,
        "split": split.summary,
        "fit_seconds": timings,
        "ranker_rows": int(len(y)),
        "ranker_queries": int(len(groups)),
        "ranker_positive_rate": float(y.mean()),
        "two_tower_loss": getattr(retrievers["two_tower"], "loss_history_", []),
        "feature_importance": ranker.feature_importance().head(25).round(1).to_dict(),
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    engine.metadata = report
    return engine, split, report
