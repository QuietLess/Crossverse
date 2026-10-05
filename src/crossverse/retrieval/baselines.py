"""Behavioural baselines: popularity, item-kNN, implicit ALS and cross-domain co-preference."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from crossverse.retrieval.base import History, Retriever, TrainData

log = logging.getLogger(__name__)


def _history_vector(history: History, n_items: int, positives_only: bool = False) -> sp.csr_matrix:
    h = history.positives() if positives_only else history
    return sp.csr_matrix((h.weight.astype(np.float32), (np.zeros(len(h), dtype=np.int64), h.idx)), shape=(1, n_items))


def _prune_top_k(m: sp.spmatrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of every row."""
    m = sp.csr_matrix(m)
    m.eliminate_zeros()
    indptr, indices, data = m.indptr, m.indices, m.data
    rows, cols, vals = [], [], []
    for r in range(m.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            continue
        d = data[a:b]
        sel = np.argpartition(-d, k - 1)[:k] if b - a > k else np.arange(b - a)
        rows.append(np.full(len(sel), r, dtype=np.int32))
        cols.append(indices[a:b][sel])
        vals.append(d[sel])
    if not rows:
        return sp.csr_matrix(m.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=m.shape
    )


class PopularityRetriever(Retriever):
    """Most-liked items (per domain once a domain mask is applied)."""

    name = "popularity"

    def fit(self, data: TrainData) -> PopularityRetriever:
        self.scores_ = np.log1p(data.item_popularity()).astype(np.float32)
        return self

    def score(self, history: History) -> np.ndarray:
        return self.scores_.copy()


class ItemKNNRetriever(Retriever):
    """Item-item cosine on the joint (movie+game) positive matrix with shrinkage."""

    name = "item_knn"

    def __init__(self, neighbors: int = 100, shrink: float = 10.0, block: int = 1000):
        self.neighbors = neighbors
        self.shrink = shrink
        self.block = block

    def fit(self, data: TrainData) -> ItemKNNRetriever:
        X = data.positive.tocsc().astype(np.float32)
        pop = np.asarray(X.sum(axis=0)).ravel()
        n = X.shape[1]
        XT = X.T.tocsr()
        sqrt_pop = np.sqrt(pop)
        blocks = []
        for start in range(0, n, self.block):
            stop = min(start + self.block, n)
            co = (XT[start:stop] @ X).tocoo()  # block_items x items co-occurrence counts
            mask = co.row + start != co.col
            r, c, v = co.row[mask], co.col[mask], co.data[mask]
            sim = v / (sqrt_pop[r + start] * sqrt_pop[c] + self.shrink)
            blocks.append(_prune_top_k(sp.csr_matrix((sim, (r, c)), shape=(stop - start, n)), self.neighbors))
        self.sim_ = sp.vstack(blocks).tocsr()
        return self

    def score(self, history: History) -> np.ndarray:
        h = _history_vector(history, self.sim_.shape[0])
        return np.asarray((h @ self.sim_).todense()).ravel()


class CoPreferenceRetriever(Retriever):
    """Interpretable cross-domain transfer baseline (blueprint 5.2).

    For bridge users (positives in both domains) count movie<->game co-likes. A target item j in
    the other domain is scored by sum_i w_i * C[i, j] / (pop_i^0.5 * pop_j^alpha), where alpha
    normalises away pure popularity. Only cross-domain pairs are scored.
    """

    name = "copref"

    def __init__(self, alpha: float = 0.5, neighbors: int = 200):
        self.alpha = alpha
        self.neighbors = neighbors

    def fit(self, data: TrainData) -> CoPreferenceRetriever:
        X = data.positive.tocsr().astype(np.float32)
        dom = data.catalog.domain
        movies = np.flatnonzero(dom == 0)
        games = np.flatnonzero(dom == 1)
        Xm, Xg = X[:, movies], X[:, games]
        bridge = (np.asarray(Xm.sum(axis=1)).ravel() > 0) & (np.asarray(Xg.sum(axis=1)).ravel() > 0)
        Xm, Xg = Xm[bridge], Xg[bridge]
        self.n_bridge_users_ = int(bridge.sum())
        C = (Xm.T @ Xg).tocoo()  # movies x games co-like counts
        pop_m = np.asarray(Xm.sum(axis=0)).ravel()
        pop_g = np.asarray(Xg.sum(axis=0)).ravel()
        n = len(dom)
        mi, gi = movies[C.row], games[C.col]
        self.counts_ = sp.csr_matrix(
            (np.concatenate([C.data, C.data]), (np.concatenate([mi, gi]), np.concatenate([gi, mi]))), shape=(n, n)
        )
        m2g = C.data / (np.sqrt(pop_m[C.row]) * np.power(pop_g[C.col], self.alpha) + 1e-9)
        g2m = C.data / (np.sqrt(pop_g[C.col]) * np.power(pop_m[C.row], self.alpha) + 1e-9)
        T = sp.csr_matrix(
            (np.concatenate([m2g, g2m]).astype(np.float32), (np.concatenate([mi, gi]), np.concatenate([gi, mi]))),
            shape=(n, n),
        )
        self.transfer_ = _prune_top_k(T, self.neighbors)
        return self

    def score(self, history: History) -> np.ndarray:
        h = _history_vector(history, self.transfer_.shape[0])
        return np.asarray((h @ self.transfer_).todense()).ravel()

    def co_likes(self, source_idx: np.ndarray, target: int) -> np.ndarray:
        """Number of bridge users who liked both each source item and the target."""
        if len(source_idx) == 0:
            return np.zeros(0)
        return np.asarray(self.counts_[source_idx, target].todense()).ravel()


class ALSRetriever(Retriever):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008) with exact fold-in for new histories.

    domain=None trains on both domains jointly (a cross-domain behavioural model); domain="movie"
    or "game" trains a single-domain model whose scores are zero outside that domain.
    """

    def __init__(self, factors: int = 64, iterations: int = 12, regularization: float = 0.05,
                 alpha: float = 10.0, domain: str | None = None, seed: int = 0, solver: str = "cg",
                 cg_steps: int = 3):
        self.factors = factors
        self.iterations = iterations
        self.reg = regularization
        self.alpha = alpha
        self.domain = domain
        self.seed = seed
        self.solver = solver  # "cg" (vectorised, default) or "exact" (per-row solve)
        self.cg_steps = cg_steps
        self.name = "als" if domain is None else f"als_{domain}"

    @staticmethod
    def _solve_side(Cui: sp.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
        k = Y.shape[1]
        YtY = Y.T @ Y
        eye = reg * np.eye(k, dtype=np.float64)
        X = np.zeros((Cui.shape[0], k), dtype=np.float32)
        indptr, indices, data = Cui.indptr, Cui.indices, Cui.data
        for u in range(Cui.shape[0]):
            a, b = indptr[u], indptr[u + 1]
            if a == b:
                continue
            idx, conf = indices[a:b], data[a:b]
            Yi = Y[idx].astype(np.float64)
            A = YtY + (Yi.T * (conf - 1.0)) @ Yi + eye
            rhs = (Yi * conf[:, None]).sum(axis=0)
            X[u] = np.linalg.solve(A, rhs)
        return X

    @staticmethod
    def _solve_side_cg(Cui: sp.csr_matrix, Y: np.ndarray, X0: np.ndarray, reg: float, steps: int = 3,
                       chunk: int = 500_000) -> np.ndarray:
        """Vectorised conjugate gradient for all rows at once (Takács et al. 2011; as in `implicit`).

        Solves (YᵀY + λI + Yᵀ(C_u − I)Y) x_u = Yᵀ C_u p_u for every row u, warm-started from X0.
        Exact solves loop over rows in Python; this does a few sparse/dense passes instead.
        """
        rows = np.repeat(np.arange(Cui.shape[0]), np.diff(Cui.indptr))
        cols, conf = Cui.indices, Cui.data.astype(np.float32)
        G = (Y.T @ Y + reg * np.eye(Y.shape[1])).astype(np.float32)
        Ym1 = sp.csr_matrix((conf - 1.0, cols, Cui.indptr), shape=Cui.shape)

        def apply_A(V: np.ndarray) -> np.ndarray:
            dots = np.empty(len(rows), dtype=np.float32)
            for s in range(0, len(rows), chunk):
                e = s + chunk
                dots[s:e] = np.einsum("ij,ij->i", V[rows[s:e]], Y[cols[s:e]])
            W = sp.csr_matrix((Ym1.data * dots, cols, Cui.indptr), shape=Cui.shape)
            return V @ G + W @ Y

        B = Cui @ Y  # Σ_i c_ui y_i
        X = X0.astype(np.float32, copy=True)
        R = B - apply_A(X)
        P = R.copy()
        rs = np.einsum("ij,ij->i", R, R)
        for _ in range(steps):
            AP = apply_A(P)
            alpha = rs / np.maximum(np.einsum("ij,ij->i", P, AP), 1e-12)
            X += alpha[:, None] * P
            R -= alpha[:, None] * AP
            rs_new = np.einsum("ij,ij->i", R, R)
            P = R + (rs_new / np.maximum(rs, 1e-12))[:, None] * P
            rs = rs_new
        return X

    def fit(self, data: TrainData) -> ALSRetriever:
        rng = np.random.default_rng(self.seed)
        P = data.positive.tocsr().astype(np.float32)
        n_items = P.shape[1]
        self.item_mask_ = data.catalog.mask(self.domain)
        if self.domain is not None:
            P = P[:, np.flatnonzero(self.item_mask_)]
        P = P[np.asarray(P.sum(axis=1)).ravel() > 0]
        C = P.copy()
        C.data = 1.0 + self.alpha * C.data
        Ct = C.T.tocsr()
        U = (rng.standard_normal((C.shape[0], self.factors)) * 0.01).astype(np.float32)
        V = (rng.standard_normal((C.shape[1], self.factors)) * 0.01).astype(np.float32)
        for it in range(self.iterations):
            if self.solver == "cg":
                U = self._solve_side_cg(C, V, U, self.reg, self.cg_steps)
                V = self._solve_side_cg(Ct, U, V, self.reg, self.cg_steps)
            else:
                U = self._solve_side(C, V, self.reg)
                V = self._solve_side(Ct, U, self.reg)
            log.debug("%s iteration %d", self.name, it)
        self.item_factors_ = np.zeros((n_items, self.factors), dtype=np.float32)
        self.item_factors_[np.flatnonzero(self.item_mask_)] = V
        self._YtY = V.T.astype(np.float64) @ V
        return self

    def user_vector(self, history: History) -> np.ndarray:
        h = history.positives()
        h = History(h.idx[self.item_mask_[h.idx]], h.weight[self.item_mask_[h.idx]])
        if len(h) == 0:
            return np.zeros(self.factors, dtype=np.float32)
        Y = self.item_factors_[h.idx].astype(np.float64)
        conf = 1.0 + self.alpha * h.weight.astype(np.float64)
        A = self._YtY + (Y.T * (conf - 1.0)) @ Y + self.reg * np.eye(self.factors)
        return np.linalg.solve(A, (Y * conf[:, None]).sum(axis=0)).astype(np.float32)

    def score(self, history: History) -> np.ndarray:
        s = self.item_factors_ @ self.user_vector(history)
        s[~self.item_mask_] = 0.0
        return s
