import numpy as np

from crossverse.retrieval.base import Catalog, History, TrainData, rating_to_weight, top_k
from crossverse.retrieval.baselines import ALSRetriever, CoPreferenceRetriever, ItemKNNRetriever
from crossverse.retrieval.content import ContentRetriever
from crossverse.retrieval.two_tower import TwoTowerRetriever


def _data(fixture_data):
    interactions, items = fixture_data
    cat = Catalog(items)
    return cat, TrainData.build(cat, interactions)


def test_rating_weights_are_signed():
    assert rating_to_weight(5) > rating_to_weight(4) > rating_to_weight(3) == 0 > rating_to_weight(1)


def test_top_k_respects_mask_and_exclude():
    s = np.array([5.0, 4.0, 3.0, 2.0])
    mask = np.array([True, True, False, True])
    assert top_k(s, 2, mask, np.array([0])).tolist() == [1, 3]


def test_copref_scores_only_the_other_domain(fixture_data):
    cat, data = _data(fixture_data)
    cp = CoPreferenceRetriever().fit(data)
    movies = np.flatnonzero(cat.domain == 0)[:5]
    s = cp.score(History(movies, np.ones(5, dtype=np.float32)))
    assert np.all(s[cat.domain == 0] == 0)
    assert s[cat.domain == 1].max() > 0


def test_als_fold_in_matches_domain(fixture_data):
    cat, data = _data(fixture_data)
    als = ALSRetriever(factors=8, iterations=2, domain="game").fit(data)
    h = History(np.flatnonzero(cat.domain == 1)[:3], np.ones(3, dtype=np.float32))
    s = als.score(h)
    assert np.all(s[cat.domain == 0] == 0)
    assert np.isfinite(s).all()


def test_item_knn_neighbours_are_pruned(fixture_data):
    _, data = _data(fixture_data)
    knn = ItemKNNRetriever(neighbors=10, block=50).fit(data)
    assert (np.diff(knn.sim_.indptr) <= 10).all()
    assert knn.sim_.diagonal().sum() == 0


def test_two_tower_learns_and_negatives_match_target_domain(fixture_data):
    cat, data = _data(fixture_data)
    content = ContentRetriever(dim=16).fit(data)
    tt = TwoTowerRetriever(content.embeddings_, dim=8, epochs=4, seed=1)
    tt.fit(data)
    assert tt.loss_history_[-1] < tt.loss_history_[0]

    # negative sampling: re-create sampler state and check domain matching
    tt2 = TwoTowerRetriever(content.embeddings_, dim=8, epochs=0, seed=2)
    tt2.fit(data)
    hist = data.user_histories()
    tt2.hist_idx = [hist[u].idx for u in range(len(data.user_ids))]
    tt2.hist_w = [hist[u].weight for u in range(len(data.user_ids))]
    _, _, targets, negs = tt2._sample_batch(np.arange(200))
    assert (tt2.domain[negs] == tt2.domain[targets][:, None]).all()


def test_content_cold_start_query_prefers_matching_theme(fixture_data):
    cat, data = _data(fixture_data)
    content = ContentRetriever(dim=16).fit(data)
    s = content.score_query("", ["western"])
    top = top_k(s, 10)
    share = np.mean(["western" in cat.items["themes"].iat[i] for i in top])
    assert share >= 0.8


def test_als_cg_solver_matches_exact(fixture_data):
    cat, data = _data(fixture_data)
    exact = ALSRetriever(factors=8, iterations=4, solver="exact", seed=5).fit(data)
    cg = ALSRetriever(factors=8, iterations=4, solver="cg", cg_steps=8, seed=5).fit(data)
    h = History(np.flatnonzero(cat.domain == 0)[:4], np.ones(4, dtype=np.float32))
    a, b = exact.score(h), cg.score(h)
    assert np.corrcoef(a, b)[0, 1] > 0.98
    top_a, top_b = set(top_k(a, 20).tolist()), set(top_k(b, 20).tolist())
    assert len(top_a & top_b) >= 15


def _tiny_world():
    """Fans of game g_x: 4 also like the bestseller m_hit, 1 likes m_rare (which has no other fan)."""
    import pandas as pd

    items = pd.DataFrame({"item_id": ["g_x", "g_y", "m_hit", "m_rare"], "domain": ["game", "game", "movie", "movie"],
                          "title": ["X", "Y", "Hit", "Rare"], "text": "", "themes": [[]] * 4, "genres": [[]] * 4,
                          "year": 2000})
    rows = [(f"u{u}", "g_x") for u in range(5)] + [(f"u{u}", "m_hit") for u in range(4)] + [("u4", "m_rare")]
    # m_hit is a bestseller among bridge users (they also play g_y), which the popularity normalisation divides out
    rows += [(f"v{u}", "m_hit") for u in range(20)] + [(f"v{u}", "g_y") for u in range(20)]
    inter = pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=5.0, timestamp=range(len(rows)))
    inter["domain"] = inter["item_id"].str[0].map({"g": "game", "m": "movie"})
    catalog = Catalog(items)
    return catalog, TrainData.build(catalog, inter)


def test_min_support_drops_single_fan_pairs():
    catalog, data = _tiny_world()
    h = History(np.array([catalog.index["g_x"]]), np.array([1.0]))
    rare, hit = catalog.index["m_rare"], catalog.index["m_hit"]

    noisy = CoPreferenceRetriever(min_support=1).fit(data).score(h)
    assert noisy[rare] > noisy[hit]  # the failure mode: one shared fan of an obscure title wins
    robust = CoPreferenceRetriever(min_support=3).fit(data).score(h)
    assert robust[rare] == 0 and robust[hit] > 0

    knn = ItemKNNRetriever(min_support=2).fit(data).score(h)
    assert knn[rare] == 0 and knn[hit] > 0
