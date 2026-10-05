"""Content-only semantic retrieval: TF-IDF over metadata text -> SVD embeddings + theme vectors."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from crossverse.features.themes import THEMES, extract_themes
from crossverse.retrieval.base import Catalog, History, Retriever, TrainData, normalize_rows


def theme_matrix(themes_per_item) -> np.ndarray:
    pos = {t: i for i, t in enumerate(THEMES)}
    M = np.zeros((len(themes_per_item), len(THEMES)), dtype=np.float32)
    for r, ts in enumerate(themes_per_item):
        for rank, t in enumerate(ts):
            if t in pos:
                M[r, pos[t]] = 1.0 / (1.0 + 0.25 * rank)
    return M


class ContentRetriever(Retriever):
    """Metadata embeddings; a user profile is the signed, recency-weighted mean of item vectors.

    Theme one-hots are concatenated to the SVD text embedding so that the shared cross-domain
    vocabulary ("cyberpunk", "fantasy") carries weight even when the domains phrase things
    differently ("film" vs "gameplay").
    """

    name = "content"

    def __init__(self, dim: int = 128, theme_weight: float = 0.6, seed: int = 0):
        self.dim = dim
        self.theme_weight = theme_weight
        self.seed = seed

    def fit_catalog(self, catalog: Catalog) -> ContentRetriever:
        texts = catalog.items["text"].fillna("").astype(str).tolist()
        self.vectorizer_ = TfidfVectorizer(
            max_features=60000, min_df=2, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        try:
            tfidf = self.vectorizer_.fit_transform(texts)
        except ValueError:  # tiny catalogs: min_df too strict
            self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            tfidf = self.vectorizer_.fit_transform(texts)
        dim = max(2, min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=dim, random_state=self.seed)
        text_emb = normalize_rows(self.svd_.fit_transform(tfidf).astype(np.float32))
        self.theme_matrix_ = theme_matrix(catalog.items["themes"].tolist())
        theme_emb = normalize_rows(self.theme_matrix_)
        self.embeddings_ = normalize_rows(np.hstack([text_emb, self.theme_weight * theme_emb]))
        return self

    def fit(self, data: TrainData) -> ContentRetriever:
        return self.fit_catalog(data.catalog)

    def profile(self, history: History) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(self.embeddings_.shape[1], dtype=np.float32)
        n = len(history)
        recency = 0.5 ** ((n - 1 - np.arange(n)) / 20.0)
        w = history.weight * recency
        v = (self.embeddings_[history.idx] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def score(self, history: History) -> np.ndarray:
        # Cast the (float64) profile, not the 26k x d matrix: mixed-dtype matmul copies the matrix.
        return self.embeddings_ @ self.profile(history).astype(self.embeddings_.dtype)

    def query_vector(self, text: str = "", themes: list[str] | None = None) -> np.ndarray:
        """Embed a free-text / theme preference (cold start)."""
        themes = list(dict.fromkeys((themes or []) + extract_themes(text)))
        text_part = np.zeros(self.svd_.n_components, dtype=np.float32)
        if text or themes:
            tf = self.vectorizer_.transform([" ".join([text] + themes)])
            text_part = normalize_rows(self.svd_.transform(tf).astype(np.float32))[0]
        theme_part = normalize_rows(theme_matrix([themes]))[0]
        v = np.concatenate([text_part, self.theme_weight * theme_part])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def score_query(self, text: str = "", themes: list[str] | None = None) -> np.ndarray:
        return self.embeddings_ @ self.query_vector(text, themes).astype(self.embeddings_.dtype)

    def similarity(self, i: int, j: np.ndarray) -> np.ndarray:
        return self.embeddings_[j] @ self.embeddings_[i]
