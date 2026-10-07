import pickle
import zlib

import numpy as np
import pandas as pd

from crossverse.retrieval.base import Catalog, History
from crossverse.retrieval.semantic import SemanticRetriever, item_text

CONCEPTS = {"cowboy": 0, "outlaw": 0, "gunslinger": 0, "frontier": 0, "dragon": 1, "wizard": 1, "spaceship": 2}


def fake_embedder(texts):
    """Synonyms share a dimension (what a real sentence model learns); other words hash elsewhere."""
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in t.lower().replace(".", " ").replace(",", " ").split():
            out[r, CONCEPTS.get(w, 3 + zlib.crc32(w.encode()) % 13)] += 1
    return out


def _catalog():
    items = pd.DataFrame({
        "item_id": ["g_rdr", "m_west", "m_dragon", "m_space"], "domain": ["game", "movie", "movie", "movie"],
        "title": ["Outlaw Ride", "Gunslinger", "Dragon Keep", "Spaceship"],
        "text": ["Outlaw Ride an outlaw on the frontier", "Gunslinger a cowboy gunslinger",
                 "Dragon Keep a wizard and a dragon", "Spaceship a spaceship"],
        "themes": [["western"], ["western"], ["fantasy"], ["sci-fi"]], "genres": [[]] * 4, "year": 2000,
    })
    return Catalog(items)


def test_similar_meaning_without_shared_words(tmp_path):
    catalog = _catalog()
    r = SemanticRetriever(cache_dir=tmp_path, embedder=fake_embedder).fit_catalog(catalog)
    s = r.score(History(np.array([0]), np.array([1.0])))
    assert s[1] > s[2] and s[1] > s[3]  # "outlaw on the frontier" ~ "cowboy gunslinger"
    q = r.score_query("frontier")
    assert int(np.argmax(q[1:])) + 1 == 1


def test_embeddings_are_cached_and_the_model_is_not_pickled(tmp_path):
    catalog = _catalog()
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embedder(texts)

    a = SemanticRetriever(cache_dir=tmp_path, embedder=counting).fit_catalog(catalog)
    b = SemanticRetriever(cache_dir=tmp_path, embedder=counting).fit_catalog(catalog)
    assert calls == [4] and np.array_equal(a.embeddings_, b.embeddings_)
    changed = _catalog()
    changed.items.loc[3, "text"] = "Spaceship a spaceship crew"  # one description changes
    SemanticRetriever(cache_dir=tmp_path, embedder=counting).fit_catalog(changed)
    assert calls == [4, 1]  # only that item is re-encoded
    restored = pickle.loads(pickle.dumps(a))
    assert restored._embedder is None and np.array_equal(restored.embeddings_, a.embeddings_)


def test_item_text_puts_title_and_themes_first():
    t = item_text("Alien", ["sci-fi", "horror"], "Alien In space no one can hear you scream")
    assert t.startswith("Alien. Themes: sci-fi, horror.") and "Alien Alien" not in t
