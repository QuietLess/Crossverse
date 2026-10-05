"""Golden recommendation tests on the deterministic synthetic fixture (planted cross-domain taste)."""

import numpy as np


def _theme_items(engine, domain, theme, n=4):
    items = engine.catalog.items
    m = (items["domain"] == domain) & items["themes"].map(lambda t: len(t) > 0 and t[0] == theme)
    sub = items[m].assign(pop=lambda d: d["item_id"].map(lambda i: engine.popularity[engine.catalog.index[i]]))
    return sub.sort_values("pop", ascending=False)["item_id"].head(n).tolist()


def _share(result, theme):
    return np.mean([theme in r.themes for r in result.items])


def test_cyberpunk_movies_give_cyberpunk_games(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "movie", "cyberpunk")]
    res = engine.recommend(liked, target_domain="game", k=10)
    assert len(res.items) == 10
    assert all(r.domain == "game" for r in res.items)
    assert _share(res, "cyberpunk") >= 0.5


def test_fantasy_games_give_fantasy_movies(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "game", "fantasy")]
    res = engine.recommend(liked, target_domain="movie", k=10)
    assert all(r.domain == "movie" for r in res.items)
    assert _share(res, "fantasy") >= 0.5


def test_mixed_profile_spans_domains_and_excludes_history(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "movie", "horror", 3) + _theme_items(engine, "game", "horror", 3)]
    res = engine.recommend(liked, k=20)
    ids = {r.item_id for r in res.items}
    assert not ids & {i for i, _ in liked}
    assert {r.domain for r in res.items} == {"movie", "game"}


def test_dislikes_push_theme_down(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "movie", "crime", 3)]
    base = engine.recommend(liked, target_domain="game", k=10, explain=False)
    disliked = _theme_items(engine, "game", "crime", 4)
    res = engine.recommend(liked, disliked=disliked, target_domain="game", k=10, explain=False)
    assert _share(res, "crime") <= _share(base, "crime")


def test_cold_start_uses_preferences(trained):
    engine, _, _ = trained
    res = engine.recommend([], preferences=["racing"], k=10)
    assert res.cold_start
    assert _share(res, "racing") >= 0.5


def test_explanations_are_grounded(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "movie", "cyberpunk")]
    res = engine.recommend(liked, target_domain="game", k=5)
    titles = set(engine.catalog.items.set_index("item_id").loc[[i for i, _ in liked], "title"])
    for r in res.items:
        ev = r.evidence
        assert ev["reason_confidence"] in {"high", "medium", "low"}
        assert set(ev["shared_themes"]) <= set(r.themes)
        assert all(a["title"] in titles for a in ev["anchors"])  # anchors come from the real history
        assert ev["summary"].startswith("Recommended because")


def test_deterministic(trained):
    engine, _, _ = trained
    liked = [(i, 5.0) for i in _theme_items(engine, "movie", "war")]
    a = [r.item_id for r in engine.recommend(liked, target_domain="game", k=10).items]
    b = [r.item_id for r in engine.recommend(liked, target_domain="game", k=10).items]
    assert a == b


def test_full_pipeline_beats_popularity_cross_domain(trained):
    from crossverse.evaluation.benchmark import run_benchmark

    engine, split, _ = trained
    tasks = {t: split.test_tasks[t] for t in ("movie_to_game", "game_to_movie")}
    df = run_benchmark(engine, tasks, models=["popularity", "crossverse_ranker"], ks=(10, 20))
    allseg = df[df["segment"] == "all"].groupby("model")["recall@20"].mean()
    assert allseg["crossverse_ranker"] > allseg["popularity"]


def test_routed_ranker_trains_and_routes(fixture_data, small_settings):
    import dataclasses

    from crossverse.ranking.ranker import RoutedRanker
    from crossverse.training import train

    interactions, items = fixture_data
    s = dataclasses.replace(small_settings, model=dataclasses.replace(small_settings.model, routed_ranker=True))
    engine, split, _ = train(interactions, items, s, version="routed", max_eval_cases=150)
    assert isinstance(engine.ranker, RoutedRanker)
    assert engine.ranker.n_queries_["cross"] > 0 and engine.ranker.n_queries_["general"] > 0
    case = split.test_tasks["movie_to_game"].cases[0]
    recs = engine.recommend_case(case, "game", 10)
    assert len(recs) == 10 and (engine.catalog.domain[recs] == 1).all()
