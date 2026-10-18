"""Scheduled new-release refresh: AniList anime, anime tagging, the published index and live reload."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crossverse.data import new_releases as N
from crossverse.serving import new_releases as S
from crossverse.serving.api import create_app


# ---------------------------------------------------------------------------------------------
# fakes for the external APIs (no network in tests)
# ---------------------------------------------------------------------------------------------
def media(mid, english, romaji, fmt, year, popularity=50_000, prequel_tv=False,
          description="An elf <i>mage</i> travels on.<br>Years pass."):
    return {"id": mid, "title": {"english": english, "romaji": romaji}, "synonyms": [], "format": fmt,
            "startDate": {"year": year}, "description": description, "genres": ["Adventure", "Fantasy"],
            "tags": [{"name": "Elf", "rank": 80, "isMediaSpoiler": False},
                     {"name": "Tragedy Twist", "rank": 90, "isMediaSpoiler": True}],
            "popularity": popularity, "coverImage": {"large": f"https://img.example/{mid}.jpg"},
            "relations": {"edges": [{"relationType": "PREQUEL", "node": {"type": "ANIME", "format": "TV"}}]
                          if prequel_tv else []}}


class FakeAniList:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def query(self, query, variables):
        self.calls.append(variables)
        page = variables["page"]
        found = self.pages[page - 1] if page <= len(self.pages) else []
        return {"data": {"Page": {"pageInfo": {"hasNextPage": page < len(self.pages)}, "media": found}}}


class FakeTMDB:
    def __init__(self, results):
        self.results = results

    def get(self, path, **params):
        kind = path.split("/")[1]
        return {"results": [r for r in self.results if r["_kind"] == kind], "total_pages": 1}

    def details(self, kind, tmdb_id):
        return {"genres": [{"name": "Animation"}, {"name": "Drama"}], "overview": "A story.", "keywords": {}}


class FakeIGDB:
    FIELDS = "name"

    def query(self, body):
        return [{"id": 77, "name": "Brand New Game", "first_release_date": 1_700_000_000, "summary": "A game.",
                 "total_rating_count": 500, "genres": [{"name": "Adventure"}]}]


def tmdb_result(tid, kind, title, year, lang, genre_ids=(16,)):
    date = f"{year}-04-01"
    return {"_kind": kind, "id": tid, "title" if kind == "movie" else "name": title, "genre_ids": list(genre_ids),
            "original_language": lang, "release_date" if kind == "movie" else "first_air_date": date,
            "vote_count": 900, "poster_path": f"/{tid}.jpg"}


# ---------------------------------------------------------------------------------------------
# fetching and tagging
# ---------------------------------------------------------------------------------------------
def test_anilist_titles_are_films_and_series_tagged_anime():
    client = FakeAniList([
        [media(1, "Frieren: Beyond Journey's End", "Sousou no Frieren", "TV", 2023),
         media(2, None, "Kimitachi wa Dou Ikiru ka", "MOVIE", 2023),
         media(3, "Jujutsu Kaisen Season 2", "Jujutsu Kaisen 2nd Season", "TV", 2023, prequel_tv=True),
         media(4, "A Recap Special", "Recap", "SPECIAL", 2024)],
        [media(5, "Dandadan", "Dandadan", "ONA", 2024)]])
    rows = N.anilist_titles(client, since="2021-01-01")
    by = {r["item_id"]: r for r in rows}
    # a later season continues a series (one work, as in TMDB); specials and recaps aren't new works
    assert set(by) == {"n_a1", "n_a2", "n_a5"}
    assert (by["n_a1"]["kind"], by["n_a2"]["kind"], by["n_a5"]["kind"]) == ("tv", "movie", "tv")
    assert all(r["domain"] == "movie" and "Anime" in r["genres"] and r["ext_source"] == "anilist" for r in rows)
    assert by["n_a1"]["title"] == "Frieren: Beyond Journey's End"  # English title when there is one
    assert by["n_a2"]["title"] == "Kimitachi wa Dou Ikiru ka"
    assert "<" not in by["n_a1"]["text"] and "Elf" in by["n_a1"]["keywords"]
    assert "Tragedy Twist" not in by["n_a1"]["keywords"]  # no spoiler tags
    assert len(client.calls) == 2 and client.calls[0]["since"] == 20210101


def test_tmdb_japanese_animation_is_tagged_anime():
    tmdb = FakeTMDB([tmdb_result(10, "tv", "Frieren: Beyond Journey's End", 2023, "ja"),
                     tmdb_result(11, "tv", "Arcane", 2021, "en")])
    rows = {r["title"]: r for r in N.tmdb_titles(tmdb, "tv", since="2021-01-01", until="2026-10-18")}
    assert "Anime" in rows["Frieren: Beyond Journey's End"]["genres"]
    assert "Anime" not in rows["Arcane"]["genres"]  # animation, but not anime


def test_anime_found_by_both_sources_is_one_title():
    tmdb_rows = N.tmdb_titles(FakeTMDB([tmdb_result(10, "tv", "Frieren: Beyond Journey's End", 2023, "ja")]),
                              "tv", since="2021-01-01", until="2026-10-18")
    ani_rows = N.anilist_titles(FakeAniList([[
        media(1, "Frieren: Beyond Journey’s End", "Sousou no Frieren", "TV", 2023),  # curly apostrophe
        media(5, "Dandadan", "Dandadan", "ONA", 2024)]]), since="2021-01-01")
    merged = N.merge_anime(tmdb_rows, ani_rows)
    titles = sorted(r["title"] for r in merged)
    assert titles == ["Dandadan", "Frieren: Beyond Journey's End"]  # the TMDB entry wins, AniList adds Dandadan
    df = N.build(merged, pd.DataFrame({"ext_source": [], "ext_id": []}), {})
    assert all(themes[0] == "anime" for themes in df["themes"])  # shown as the first theme pill


# ---------------------------------------------------------------------------------------------
# the published index
# ---------------------------------------------------------------------------------------------
def frame(titles, kinds=None):
    kinds = kinds or ["tv"] * len(titles)
    rows = [{"item_id": f"n_t{n}", "domain": "game" if k == "game" else "movie", "kind": k, "title": t,
             "year": 2025, "text": f"{t}.", "genres": ["Drama"], "themes": ["drama"], "image": "", "collection": "",
             "keywords": [], "ext_source": "tmdb", "ext_id": n, "votes": 100 - n}
            for n, (t, k) in enumerate(zip(titles, kinds, strict=True))]
    df = pd.DataFrame(rows, columns=[*N.COLUMNS])
    df.attrs["fetched_at"] = "2026-10-18"
    return df


def index_of(titles, n_catalog=5):
    df = frame(titles)
    vecs = np.eye(len(titles), 4, dtype=np.float32)
    return S.NewReleaseIndex(df, vecs, {n_catalog: [0]}, {0: [1, 2]}, "2026-10-18")


def test_published_index_round_trips_without_pickle(tmp_path):
    index = index_of(["Show A", "Show B", "Show C"])
    version = S.publish(index, tmp_path)
    assert (tmp_path / "CURRENT").read_text().strip() == version
    files = sorted(p.name for p in (tmp_path / version).iterdir())
    assert files == ["embeddings.npy", "graph.json", "items.parquet"]  # data formats only, nothing executable
    got_version, got = S.load_published(tmp_path)
    assert got_version == version and len(got) == 3
    assert got.items["title"].tolist() == ["Show A", "Show B", "Show C"]
    assert np.array_equal(got.embeddings, index.embeddings)
    assert got.links == {5: [0]} and got.stand_ins == {0: [1, 2]} and got.fetched_at == "2026-10-18"
    assert got.item_dict(0)["themes"] == ["drama"]


def test_publish_keeps_a_few_versions_and_refuses_a_collapse(tmp_path):
    versions = [S.publish(index_of([f"Show {c}{n}" for n in range(4)]), tmp_path) for c in "ABCD"]
    assert len(set(versions)) == 4
    kept = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert kept == sorted(versions[-3:])  # older versions are pruned
    # an API outage or an expired key returns little: don't replace a good index with it
    with pytest.raises(ValueError, match="shrank"):
        S.publish(index_of(["Only One"]), tmp_path)
    assert (tmp_path / "CURRENT").read_text().strip() == versions[-1]
    assert S.publish(index_of(["Only One"]), tmp_path, force=True) != versions[-1]


def test_load_published_without_an_index(tmp_path):
    assert S.load_published(tmp_path) == (None, None)
    (tmp_path / "CURRENT").write_text("../../etc")  # a pointer outside the folder is ignored
    assert S.load_published(tmp_path) == (None, None)


def test_service_picks_up_a_newly_published_index(trained, small_settings, tmp_path, monkeypatch):
    engine, _, _ = trained
    monkeypatch.setitem(engine.__dict__, "new_releases_", None)  # restored after the test
    serving = dataclasses.replace(small_settings.serving, new_releases_dir=str(tmp_path),
                                  new_releases_check_seconds=0)
    settings = dataclasses.replace(small_settings, serving=serving)
    S.publish(index_of(["Zqx Alpha Show"], len(engine.catalog)), tmp_path)
    with TestClient(create_app(settings, engine)) as client:
        found = client.get("/items/search", params={"q": "Zqx Alpha Show"}).json()
        assert found and found[0]["title"] == "Zqx Alpha Show" and found[0]["new"] is True
        S.publish(index_of(["Zqx Beta Show"], len(engine.catalog)), tmp_path)
        found = client.get("/items/search", params={"q": "Zqx Beta Show"}).json()
        assert found and found[0]["title"] == "Zqx Beta Show"
        assert client.get("/items/search", params={"q": "Zqx Alpha Show"}).json() == []


# ---------------------------------------------------------------------------------------------
# the scheduled job
# ---------------------------------------------------------------------------------------------
def test_refresh_fetches_all_sources_and_publishes(trained, tmp_path, monkeypatch):
    engine, _, _ = trained
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(engine.catalog), 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    monkeypatch.setitem(engine.retrievers, "semantic", SimpleNamespace(embeddings_=vecs))
    monkeypatch.setitem(engine.__dict__, "new_releases_", None)

    def embed(texts):
        out = rng.normal(size=(len(texts), 8)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    clients = SimpleNamespace(
        tmdb=FakeTMDB([tmdb_result(10, "tv", "Frieren: Beyond Journey's End", 2023, "ja"),
                       tmdb_result(12, "movie", "Some New Film", 2025, "en", genre_ids=(18,))]),
        igdb=FakeIGDB(),
        anilist=FakeAniList([[media(1, "Frieren: Beyond Journey's End", "Sousou no Frieren", "TV", 2023),
                              media(5, "Dandadan", "Dandadan", "ONA", 2024)]]))
    version, counts = S.refresh(engine, clients, tmp_path, since="2021-01-01", until="2026-10-18", embed=embed)
    _, index = S.load_published(tmp_path)
    titles = set(index.items["title"])
    assert {"Frieren: Beyond Journey's End", "Some New Film", "Brand New Game", "Dandadan"} <= titles
    assert len(index.items) == 4  # Frieren once
    anime = set(index.items.loc[[t[0] == "anime" for t in index.items["themes"]], "title"])
    assert anime == {"Frieren: Beyond Journey's End", "Dandadan"}
    assert counts == {"movie": 1, "tv": 2, "game": 1}
    assert json.loads((tmp_path / version / "graph.json").read_text())["fetched_at"]
