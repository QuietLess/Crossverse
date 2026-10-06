import pandas as pd

from crossverse.data.amazon import _main_image, build_catalog

IMG = "https://m.media-amazon.com/images/I/{}.jpg"


def test_main_image_prefers_main_variant_and_large():
    rec = {"images": [
        {"variant": "PT01", "large": IMG.format("back"), "thumb": IMG.format("back_t")},
        {"variant": "MAIN", "large": IMG.format("front"), "thumb": IMG.format("front_t"), "hi_res": None},
    ]}
    assert _main_image(rec) == IMG.format("front")


def test_main_image_falls_back_and_rejects_non_https():
    assert _main_image({"images": [{"variant": "MAIN", "large": None, "thumb": IMG.format("t")}]}) == IMG.format("t")
    assert _main_image({"images": [{"variant": "MAIN", "large": "http://insecure/x.jpg"}]}) == ""
    assert _main_image({"images": None}) == ""
    assert _main_image({}) == ""


def test_catalog_image_comes_from_most_reviewed_product_with_one():
    meta = pd.DataFrame({
        "source_asin": ["A", "B", "C"], "domain": "movie",
        "title": ["Alien [Blu-ray]", "Alien (DVD)", "Alien VHS"],
        "description": "", "features": "", "genres": [["Horror"]] * 3, "creator": "", "year": 1979,
        "rating_number": [900, 500, 10], "main_category": "", "image": ["", IMG.format("dvd"), IMG.format("vhs")],
    })
    mapping = pd.DataFrame({"source_asin": ["A", "B", "C"], "canonical_item_id": "m_alien"})
    catalog = build_catalog(meta, mapping)
    assert catalog.loc[0, "image"] == IMG.format("dvd")


def test_api_returns_image_field(trained, small_settings):
    from fastapi.testclient import TestClient

    from crossverse.serving.api import create_app

    engine, _, _ = trained
    title = engine.catalog.items.sort_values("popularity").iloc[-1]["title"]
    with TestClient(create_app(small_settings, engine)) as client:
        body = client.post("/recommend", json={"liked": [{"item": title}], "k": 3}).json()
    # the synthetic catalog has no images: the field is present and null
    assert all("image" in i and i["image"] is None for i in body["items"])
    assert body["resolved_profile"][0]["image"] is None
