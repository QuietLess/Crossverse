from __future__ import annotations

import pytest

from crossverse.config import DataConfig, ModelConfig, ServingConfig, Settings, SplitConfig
from crossverse.data.synthetic import generate


@pytest.fixture(scope="session")
def fixture_data():
    interactions, items = generate(n_users=900, n_movies=200, n_games=150, seed=3)
    return interactions, items


@pytest.fixture(scope="session")
def small_settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("cv")
    return Settings(
        data=DataConfig(processed_dir=root / "processed"),
        split=SplitConfig(cross_holdout_frac=0.08),
        model=ModelConfig(als_factors=16, als_iterations=3, content_dim=32, tt_dim=16, tt_epochs=3,
                          ranker_queries_per_task=150, candidates_per_source=50),
        serving=ServingConfig(artifacts_dir=root / "artifacts", cache_ttl_seconds=60),
    )


@pytest.fixture(scope="session")
def trained(fixture_data, small_settings):
    from crossverse.training import train

    interactions, items = fixture_data
    engine, split, report = train(interactions, items, small_settings, version="test", max_eval_cases=150)
    return engine, split, report
