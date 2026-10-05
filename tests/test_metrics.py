import numpy as np
import pytest

from crossverse.evaluation import metrics as M


def test_recall_and_hit():
    assert M.recall_at_k([1, 2, 3], {3, 9}, 3) == pytest.approx(0.5)
    assert M.recall_at_k([1, 2, 3], {3}, 2) == 0.0
    assert M.hit_rate_at_k([1, 2, 3], {3}, 3) == 1.0
    assert M.recall_at_k([1], set(), 10) == 0.0


def test_recall_is_capped_by_k():
    # 5 relevant items, k=2, both hits -> perfect recall@2
    assert M.recall_at_k([1, 2], {1, 2, 3, 4, 5}, 2) == 1.0


def test_ndcg():
    assert M.ndcg_at_k([7, 1], {7}, 2) == pytest.approx(1.0)
    assert M.ndcg_at_k([1, 7], {7}, 2) == pytest.approx(1 / np.log2(3))
    assert M.ndcg_at_k([1, 2], {7}, 2) == 0.0


def test_map():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert M.average_precision_at_k([5, 1, 6], {5, 6}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_diversity_and_novelty():
    E = np.eye(3)
    assert M.intra_list_diversity([0, 1, 2], E) == pytest.approx(1.0)
    same = np.ones((3, 2)) / np.sqrt(2)
    assert M.intra_list_diversity([0, 1, 2], same) == pytest.approx(0.0)
    pop = np.array([100.0, 1.0])
    assert M.novelty([1], pop, 100) > M.novelty([0], pop, 100)


def test_paired_bootstrap_detects_real_lift_only():
    import pandas as pd

    from crossverse.evaluation.benchmark import paired_bootstrap

    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(400)]
    base = rng.random(400) * 0.1
    rows = []
    for u, b in zip(users, base, strict=True):
        rows += [{"task": "t", "model": "base", "user_id": u, "ndcg@10": b},
                 {"task": "t", "model": "better", "user_id": u, "ndcg@10": b + 0.02},
                 {"task": "t", "model": "same", "user_id": u, "ndcg@10": b + rng.normal(0, 0.001)}]
    cases = pd.DataFrame(rows)
    better = paired_bootstrap(cases, "better", "base").iloc[0]
    assert better["significant"] and better["ci95_low"] > 0
    same = paired_bootstrap(cases, "same", "base").iloc[0]
    assert not same["significant"]
