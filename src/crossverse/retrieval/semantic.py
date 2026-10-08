"""Semantic retrieval: sentence embeddings of each item's title, themes and description.

TF-IDF (retrieval/content.py) compares words, so "outlaw on the frontier" and "cowboy gunslinger"
look unrelated. A small sentence-embedding model maps text with similar meaning to nearby vectors.
The model runs on CPU through fastembed (ONNX runtime, no PyTorch): `pip install -e ".[semantic]"`.

Embeddings are cached per item text under <cache_dir>/semantic/, so retraining re-encodes only
items whose text changed (the first run encodes ~40k descriptions). The model itself is never pickled with
the engine; it is loaded lazily when a cold-start query needs encoding.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from crossverse.progress import Progress
from crossverse.retrieval.base import Catalog, History, Retriever, TrainData, normalize_rows

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
Embedder = Callable[[Sequence[str]], np.ndarray]


def available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def item_text(title: str, themes: Sequence[str], text: str, max_chars: int = 1500) -> str:
    """What gets embedded: the title and themes first (they survive the model's 512-token cut)."""
    head = f"{title}. Themes: {', '.join(themes)}." if len(themes) else f"{title}."
    body = text[len(title):] if text.startswith(title) else text  # catalog text repeats the title
    return f"{head} {body}"[:max_chars]


class SemanticRetriever(Retriever):
    name = "semantic"

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None, batch_size: int = 64,
                 embedder: Embedder | None = None):
        self.model = model
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._embedder = embedder  # tests inject a fake; otherwise fastembed, loaded lazily

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_embedder"] = None  # an ONNX session is not picklable (and not needed to score histories)
        state["cache_dir"] = None  # training-only, and a WindowsPath cannot be unpickled on Linux (Cloud Run)
        return state

    # ------------------------------------------------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._embedder is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(self.model)

            def encode(xs: Sequence[str]) -> np.ndarray:
                bar = Progress(len(xs), "texts", "semantic ") if len(xs) > 100 else None  # skip for queries
                out = []
                for v in model.embed(list(xs), batch_size=self.batch_size):
                    out.append(v)
                    if bar:
                        bar.update()
                return np.asarray(out)

            self._embedder = encode
        return normalize_rows(np.asarray(self._embedder(texts), dtype=np.float32))

    def _cache_file(self) -> Path | None:
        if self.cache_dir is None:
            return None
        model = hashlib.sha1(self.model.encode()).hexdigest()[:12]
        return Path(self.cache_dir) / "semantic" / f"{model}-items.npz"

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    def fit_catalog(self, catalog: Catalog) -> SemanticRetriever:
        """Embed every item, reusing cached vectors per item text: a rebuild that changes some
        descriptions re-encodes only those (the first run encodes everything, ~40 min on CPU)."""
        items = catalog.items
        texts = [item_text(str(t), list(th), str(x or ""))
                 for t, th, x in zip(items["title"], items["themes"], items["text"], strict=True)]
        keys = [self._text_key(t) for t in texts]
        path = self._cache_file()
        cached: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path) as z:
                cached = dict(zip(z["keys"].tolist(), z["vectors"], strict=True))
        todo = [n for n, k in enumerate(keys) if k not in cached]
        log.info("semantic: %d items, %d cached, encoding %d with %s",
                 len(texts), len(texts) - len(todo), len(todo), self.model)
        if todo:
            vecs = self._embed([texts[n] for n in todo])
            cached.update({keys[n]: v for n, v in zip(todo, vecs, strict=True)})
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, keys=np.array(list(cached)), vectors=np.stack(list(cached.values())))
        self.embeddings_ = np.stack([cached[k] for k in keys]).astype(np.float32)
        return self

    def fit(self, data: TrainData) -> SemanticRetriever:
        return self.fit_catalog(data.catalog)

    # ------------------------------------------------------------------------------------------
    def profile(self, history: History) -> np.ndarray:
        """Signed, recency-weighted mean of the history's item vectors (as ContentRetriever)."""
        if len(history) == 0:
            return np.zeros(self.embeddings_.shape[1], dtype=np.float32)
        n = len(history)
        recency = 0.5 ** ((n - 1 - np.arange(n)) / 20.0)
        v = (self.embeddings_[history.idx] * (history.weight * recency)[:, None]).sum(axis=0)
        norm = np.linalg.norm(v)
        return (v / norm if norm > 0 else v).astype(np.float32)

    def score(self, history: History) -> np.ndarray:
        return self.embeddings_ @ self.profile(history)

    def score_query(self, text: str = "", themes: list[str] | None = None) -> np.ndarray:
        """Cold start: embed the stated themes / free text. Zeros if the model can't be loaded."""
        query = " ".join([text, *(themes or [])]).strip()
        if not query:
            return np.zeros(len(self.embeddings_), dtype=np.float32)
        try:
            q = self._embed([f"Themes: {query}."])[0]
        except ImportError:
            log.warning("semantic: fastembed not installed; cold-start queries use the other retrievers")
            return np.zeros(len(self.embeddings_), dtype=np.float32)
        return self.embeddings_ @ q

    def similarity(self, i: int, j: np.ndarray) -> np.ndarray:
        return self.embeddings_[j] @ self.embeddings_[i]
