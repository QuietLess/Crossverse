from crossverse.evaluation.protocol import assert_no_leakage, make_split
from crossverse.retrieval.base import Catalog


def test_split_is_leakage_safe(fixture_data, small_settings):
    interactions, items = fixture_data
    catalog = Catalog(items)
    split = make_split(interactions, catalog, small_settings.split, max_cases=200)
    assert_no_leakage(split, interactions)

    # cross-domain holdout groups are disjoint
    groups = [set(u) for u in split.holdout_users.values()]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not groups[i] & groups[j]

    # movie_to_game test users: zero game interactions in train, inputs are movies only
    m2g = set(split.holdout_users["movie_to_game_test"])
    tr = split.train[split.train["user_id"].isin(m2g)]
    assert (tr["domain"] == "movie").all()
    for case in split.test_tasks["movie_to_game"].cases:
        assert (catalog.domain[case.history.idx] == 0).all()
        assert all(catalog.domain[i] == 1 for i in case.relevant)


def test_regular_users_are_split_chronologically(fixture_data, small_settings):
    interactions, items = fixture_data
    split = make_split(interactions, Catalog(items), small_settings.split, max_cases=50)
    holdout = {u for us in split.holdout_users.values() for u in us}
    tr_max = split.train[~split.train["user_id"].isin(holdout)].groupby("user_id")["timestamp"].max()
    full = interactions.set_index(["user_id", "item_id"])["timestamp"]
    catalog = Catalog(items)
    for case in split.test_tasks["mixed"].cases[:50]:
        for i in case.relevant:
            ts = full[(case.user_id, catalog.item_ids[i])]
            assert ts >= tr_max[case.user_id]


def test_tasks_present(fixture_data, small_settings):
    interactions, items = fixture_data
    split = make_split(interactions, Catalog(items), small_settings.split, max_cases=50)
    for name in ("within_movie", "within_game", "movie_to_game", "game_to_movie", "mixed", "cold_start"):
        assert split.test_tasks[name].cases, name
        assert split.val_tasks[name].cases, name
