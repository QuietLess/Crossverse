import json
import math

import pytest

from crossverse.evaluation import judgments as J

ITEM = {"item_id": "m_1", "title": "Hostiles", "domain": "movie"}


def test_ratings_append_and_latest_answer_wins(tmp_path):
    assert J.load_ratings(tmp_path).empty
    J.add_rating("q1", ITEM, 0, tmp_path)
    J.add_rating("q1", {**ITEM, "item_id": "m_2"}, 1, tmp_path)
    J.add_rating("q1", ITEM, 2, tmp_path)  # changed my mind
    r = J.load_ratings(tmp_path)
    assert dict(zip(r.item_id, r.rating, strict=True)) == {"m_1": 2, "m_2": 1}
    with pytest.raises(ValueError):
        J.add_rating("q1", ITEM, 5, tmp_path)


def test_queries_file_in_repo_is_valid():
    queries = J.load_queries()
    assert len(queries) >= 10 and len({q.id for q in queries}) == len(queries)
    assert all(q.seed_domain != q.target_domain for q in queries)


def test_blind_order_is_stable_and_not_rank_order():
    ids = [f"i{n}" for n in range(20)]
    assert J.blind_order("q", ids) == J.blind_order("q", list(reversed(ids)))
    assert J.blind_order("q", ids) != ids


def test_graded_ndcg():
    gains = {"a": 2, "b": 1, "c": 0}
    assert J.graded_ndcg(["a", "b", "c"], gains, 3) == pytest.approx(1.0)
    assert J.graded_ndcg(["c", "x", "y"], gains, 3) == 0.0  # unjudged items count as 0
    worse = J.graded_ndcg(["b", "a"], gains, 2)
    assert worse == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))


def test_score_lists_reports_judged_share(tmp_path):
    for item_id, rating in (("a", 2), ("b", 0)):
        J.add_rating("q1", {**ITEM, "item_id": item_id}, rating, tmp_path)
    s = J.score_lists({"q1": ["a", "b", "new1", "new2"]}, J.load_ratings(tmp_path), k=4)
    assert s["good@4"] == 0.25 and s["bad@4"] == 0.25 and s["judged_share"] == 0.5


def test_custom_queries_file(tmp_path):
    q = {"id": "x", "seed_item_id": "g_1", "seed_title": "Doom", "seed_domain": "game", "target_domain": "movie"}
    (tmp_path / "queries.json").write_text(json.dumps({"queries": [q]}), encoding="utf-8")
    assert J.load_queries(tmp_path)[0].seed_title == "Doom"
