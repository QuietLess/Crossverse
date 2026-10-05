"""Domain-aware two-tower model trained with sampled softmax (pure NumPy, no torch required).

Item tower (one per domain, shared latent space):
    v_i = E_i + x_i @ W_{domain(i)}          x_i = content embedding (TF-IDF/SVD + themes)
User tower (shared across domains):
    u = sum_j a_j v_j / sum_j |a_j|  +  f_movie * D_movie + f_game * D_game
    a_j = signed rating weight * time decay (dislikes push away), f_* = domain mix of the history
Score: s(u, i) = u . v_i + b_i

Training examples are leave-one-out within each user's training history. With probability
`cross_prob` the encoder only sees the *other* domain than the target, which explicitly teaches
movie->game and game->movie transfer instead of letting the model lean on same-domain signal.
Negatives are drawn from the target's own domain (half uniform, half popularity-weighted), so the
model cannot cheat by learning "targets are usually games".
"""

from __future__ import annotations

import logging

import numpy as np

from crossverse.retrieval.base import History, Retriever, TrainData

log = logging.getLogger(__name__)


class TwoTowerRetriever(Retriever):
    name = "two_tower"

    def __init__(self, content_embeddings: np.ndarray, dim: int = 64, epochs: int = 8, lr: float = 0.05,
                 batch_size: int = 512, n_negatives: int = 16, cross_prob: float = 0.5, max_history: int = 50,
                 l2: float = 1e-5, half_life: float = 20.0, seed: int = 0):
        self.X = content_embeddings.astype(np.float32)
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.cross_prob = cross_prob
        self.max_history = max_history
        self.l2 = l2
        self.half_life = half_life
        self.seed = seed

    # -- parameters -------------------------------------------------------------------------
    def _init(self, n_items: int) -> None:
        rng = self.rng
        dc = self.X.shape[1]
        self.params = {
            "E": (rng.standard_normal((n_items, self.dim)) * 0.01).astype(np.float32),
            "W": (rng.standard_normal((2, dc, self.dim)) / np.sqrt(dc)).astype(np.float32),
            "D": np.zeros((2, self.dim), dtype=np.float32),
            "b": np.zeros(n_items, dtype=np.float32),
        }
        self.acc = {k: np.full_like(v, 1e-6) for k, v in self.params.items()}

    def _item_vectors(self, idx: np.ndarray) -> np.ndarray:
        p = self.params
        d = self.domain[idx]
        out = p["E"][idx].copy()
        for k in (0, 1):
            m = d == k
            if m.any():
                out[m] += self.X[idx[m]] @ p["W"][k]
        return out

    # -- batching ---------------------------------------------------------------------------
    def _sample_batch(self, users: np.ndarray):
        L = self.max_history
        B = len(users)
        H = np.zeros((B, L), dtype=np.int64)
        A = np.zeros((B, L), dtype=np.float32)
        targets = np.zeros(B, dtype=np.int64)
        keep = np.zeros(B, dtype=bool)
        for r, u in enumerate(users):
            idx, w = self.hist_idx[u], self.hist_w[u]
            pos = np.flatnonzero(w > 0)
            if len(pos) == 0 or len(idx) < 2:
                continue
            t = pos[self.rng.integers(len(pos))]
            target = idx[t]
            ctx = np.ones(len(idx), dtype=bool)
            ctx[t] = False
            if self.rng.random() < self.cross_prob:
                other = self.domain[idx] != self.domain[target]
                if (ctx & other).any():
                    ctx &= other
            ci, cw = idx[ctx], w[ctx]
            n = len(ci)
            decay = 0.5 ** ((n - 1 - np.arange(n)) / self.half_life)
            cw = cw * decay
            if n > L:
                ci, cw = ci[-L:], cw[-L:]
                n = L
            H[r, :n], A[r, :n] = ci, cw
            targets[r] = target
            keep[r] = True
        H, A, targets = H[keep], A[keep], targets[keep]
        B = len(targets)
        negs = np.empty((B, self.n_negatives), dtype=np.int64)
        tdom = self.domain[targets]
        for k in (0, 1):
            m = tdom == k
            if not m.any():
                continue
            pool, popp = self.dom_items[k], self.dom_pop_p[k]
            n_uni = self.n_negatives // 2
            negs[m, :n_uni] = pool[self.rng.integers(len(pool), size=(m.sum(), n_uni))]
            negs[m, n_uni:] = self.rng.choice(pool, size=(m.sum(), self.n_negatives - n_uni), p=popp)
        return H, A, targets, negs

    # -- forward / backward -----------------------------------------------------------------
    def _step(self, H, A, targets, negs) -> float:
        p = self.params
        B, L = H.shape
        cand = np.concatenate([targets[:, None], negs], axis=1)  # B x (1+N)
        allidx = np.concatenate([H.ravel(), cand.ravel()])
        uniq, inv = np.unique(allidx, return_inverse=True)
        V = self._item_vectors(uniq)
        invH = inv[: B * L].reshape(B, L)
        invC = inv[B * L :].reshape(cand.shape)

        absA = np.abs(A)
        norm = absA.sum(axis=1, keepdims=True) + 1e-6
        a = A / norm
        dom_mix = np.stack([((self.domain[H] == k) * absA).sum(1) for k in (0, 1)], axis=1) / norm
        u = np.einsum("bl,bld->bd", a, V[invH]) + dom_mix @ p["D"]

        Vc = V[invC]  # B x C x d
        s = np.einsum("bd,bcd->bc", u, Vc) + p["b"][cand]
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        prob = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(prob[:, 0] + 1e-12).mean())

        ds = prob
        ds[:, 0] -= 1.0
        ds /= B
        du = np.einsum("bc,bcd->bd", ds, Vc)
        gV = np.zeros_like(V)
        np.add.at(gV, invC.ravel(), (ds[:, :, None] * u[:, None, :]).reshape(-1, self.dim))
        np.add.at(gV, invH.ravel(), (a[:, :, None] * du[:, None, :]).reshape(-1, self.dim))
        gb = np.zeros_like(p["b"])
        np.add.at(gb, cand.ravel(), ds.ravel())
        gD = dom_mix.T @ du

        grads = {"b": gb, "D": gD}
        gE = np.zeros_like(p["E"])
        gE[uniq] = gV + self.l2 * p["E"][uniq]
        grads["E"] = gE
        gW = self.l2 * p["W"].copy()
        d = self.domain[uniq]
        for k in (0, 1):
            m = d == k
            if m.any():
                gW[k] += self.X[uniq[m]].T @ gV[m]
        grads["W"] = gW

        for name, g in grads.items():  # Adagrad
            if name == "E":
                rows = uniq
                self.acc[name][rows] += g[rows] ** 2
                p[name][rows] -= self.lr * g[rows] / np.sqrt(self.acc[name][rows])
            else:
                self.acc[name] += g**2
                p[name] -= self.lr * g / np.sqrt(self.acc[name])
        return loss

    def fit(self, data: TrainData) -> TwoTowerRetriever:
        self.rng = np.random.default_rng(self.seed)
        n_items = len(data.catalog)
        self.domain = data.catalog.domain.astype(np.int64)
        pop = data.item_popularity() + 1.0
        self.dom_items, self.dom_pop_p = [], []
        for k in (0, 1):
            pool = np.flatnonzero(self.domain == k)
            pp = pop[pool] ** 0.75
            self.dom_items.append(pool)
            self.dom_pop_p.append(pp / pp.sum())
        hist = data.user_histories()
        self.hist_idx = [hist[u].idx for u in range(len(data.user_ids))]
        self.hist_w = [hist[u].weight for u in range(len(data.user_ids))]
        self._init(n_items)
        users = np.arange(len(self.hist_idx))
        self.loss_history_ = []
        for epoch in range(self.epochs):
            self.rng.shuffle(users)
            losses = []
            for start in range(0, len(users), self.batch_size):
                batch = self._sample_batch(users[start : start + self.batch_size])
                if len(batch[2]):
                    losses.append(self._step(*batch))
            self.loss_history_.append(float(np.mean(losses)))
            log.info("two-tower epoch %d loss %.4f", epoch, self.loss_history_[-1])
        self.item_vectors_ = self._item_vectors(np.arange(n_items))
        self.bias_ = self.params["b"].copy()
        # Drop training-only state so the pickled bundle stays small.
        del self.hist_idx, self.hist_w, self.acc
        return self

    def user_vector(self, history: History) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        idx, w = history.idx[-self.max_history :], history.weight[-self.max_history :]
        n = len(idx)
        w = w * 0.5 ** ((n - 1 - np.arange(n)) / self.half_life)
        absw = np.abs(w)
        norm = absw.sum() + 1e-6
        mix = np.array([(absw * (self.domain[idx] == k)).sum() for k in (0, 1)]) / norm
        return (w / norm) @ self.item_vectors_[idx] + mix @ self.params["D"]

    def score(self, history: History) -> np.ndarray:
        u = self.user_vector(history).astype(self.item_vectors_.dtype)  # avoid upcasting the item matrix
        return self.item_vectors_ @ u + self.bias_
