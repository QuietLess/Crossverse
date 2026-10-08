"""Central configuration. Every field can be overridden with a CROSSVERSE_<FIELD> env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_override(obj: object) -> None:
    for f in fields(obj):  # type: ignore[arg-type]
        raw = os.environ.get(f"CROSSVERSE_{f.name.upper()}")
        if raw is None:
            continue
        current = getattr(obj, f.name)
        if isinstance(current, bool):
            value: object = raw.lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif isinstance(current, Path):
            value = Path(raw)
        else:
            value = raw
        setattr(obj, f.name, value)


@dataclass
class DataConfig:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    # TMDB/IGDB matches from pipelines/enrich.py; used by the build when the file exists (ds5+).
    external_metadata: Path = PROJECT_ROOT / "data" / "external" / "metadata.parquet"
    use_external: bool = True
    # Ratings >= this are positive implicit events; <= negative_max are explicit dislikes.
    positive_threshold: float = 4.0
    negative_max: float = 2.0
    # Bridge population: users with >= N positives in BOTH domains.
    # 1 = every user with >=1 liked movie AND >=1 liked game (342k users); dataset v1 used 3 (48k).
    min_bridge_positives: int = 1
    # Optional single-domain users (>= this many positives) that strengthen within-domain CF.
    single_domain_users_per_domain: int = 20000
    single_domain_min_positives: int = 5
    max_bridge_users: int = 0  # 0 = keep all
    min_item_interactions: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        _env_override(self)


@dataclass
class SplitConfig:
    test_frac: float = 0.2
    val_frac: float = 0.1
    # Fraction of bridge users whose entire target-domain history is held out (per task & split).
    cross_holdout_frac: float = 0.06
    min_history: int = 2
    seed: int = 7

    def __post_init__(self) -> None:
        _env_override(self)


@dataclass
class ModelConfig:
    als_factors: int = 64
    als_iterations: int = 12
    als_regularization: float = 0.05
    als_alpha: float = 10.0
    knn_neighbors: int = 100
    # Standalone-validation optimum is shrink=500 / copref alpha=0, but both made the *full pipeline*
    # worse twice (v3 on dataset 1, v6 on dataset 2): they make the retrievers nominate the same popular
    # items, so the ranker loses complementary candidates. Defaults = production recipe (v4).
    knn_shrink: float = 10.0
    content_dim: int = 128
    copref_alpha: float = 0.5  # target-popularity normalisation exponent (see knn_shrink note)
    # Minimum users behind a pair before it counts (v10). Fixed before seeing test results.
    knn_min_support: int = 2
    copref_min_support: int = 3
    # Sentence-embedding model for the "semantic" retriever (v11); "" disables it. Needs the optional
    # `semantic` extra (fastembed); without it training skips the retriever with a warning.
    semantic_model: str = "BAAI/bge-small-en-v1.5"
    tt_dim: int = 64
    tt_epochs: int = 15
    tt_lr: float = 0.05
    tt_batch_size: int = 512
    tt_cross_prob: float = 0.5  # prob. a training example only sees the *other* domain
    tt_max_history: int = 50
    candidates_per_source: int = 100
    ranker_queries_per_task: int = 1500
    routed_ranker: bool = False  # cross-domain specialist ranker; lost to the joint ranker (v5)
    diversity_lambda: float = 0.15
    seed: int = 13

    def __post_init__(self) -> None:
        _env_override(self)


@dataclass
class ServingConfig:
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    model_version: str = "production"
    database_url: str = ""  # postgres://... ; empty -> SQLite under artifacts_dir
    redis_url: str = ""  # redis://... ; empty -> in-process cache
    cache_ttl_seconds: int = 300
    default_k: int = 10

    def __post_init__(self) -> None:
        _env_override(self)


@dataclass
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def get_settings() -> Settings:
    return Settings()


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read KEY=value lines from the project's .env (git-ignored; holds API keys). Values already in the
    process environment win, so CI/containers can inject secrets without a file."""
    path = path or PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k in values or k.startswith(("TMDB_", "IGDB_"))})
    return values
