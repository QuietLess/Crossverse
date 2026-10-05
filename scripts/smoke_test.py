"""API smoke + light load test against a running service.

    python scripts/smoke_test.py --url http://localhost:8000 --requests 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    base = args.url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=10).json()
    assert health["status"] == "ok", health
    print("health:", health["model_version"], health["catalog_items"], "items")

    def find(domain: str) -> list[dict]:
        for q in ("the", "s", "c", "a", "n", "d", "b", "m"):  # works for real and synthetic catalogs
            hits = requests.get(f"{base}/items/search", params={"q": q, "domain": domain, "limit": 20}, timeout=10).json()
            if len(hits) >= 5:
                return hits
        return []

    movies, games = find("movie"), find("game")
    assert movies and games, "catalog search returned nothing"

    r = requests.post(f"{base}/recommend/movie-to-game", json={"liked": [{"item": m["item_id"]} for m in movies[:3]], "k": 10}, timeout=30)
    r.raise_for_status()
    body = r.json()
    assert all(i["domain"] == "game" for i in body["items"]) and body["items"]
    rid = body["items"][0]["recommendation_id"]
    assert requests.get(f"{base}/explain/{rid}", timeout=10).status_code == 200
    assert requests.post(f"{base}/feedback", json={"item_id": body["items"][0]["item_id"], "event": "click"}, timeout=10).ok
    assert "crossverse_requests_total" in requests.get(f"{base}/metrics", timeout=10).text

    def one(i: int) -> float:
        liked = [{"item": games[(i + j) % len(games)]["item_id"]} for j in range(2)]
        # A distinct exclusion per request defeats the response cache: this measures real ranking work.
        body = {"liked": liked, "k": 10, "exclude": [movies[i % len(movies)]["item_id"], f"nonce-{i}"]}
        t = time.perf_counter()
        resp = requests.post(f"{base}/recommend", json=body, timeout=30)
        resp.raise_for_status()
        return (time.perf_counter() - t) * 1000

    with ThreadPoolExecutor(args.concurrency) as ex:
        lat = sorted(ex.map(one, range(args.requests)))
    p95 = lat[int(0.95 * (len(lat) - 1))]
    print(f"load: {len(lat)} requests, p50 {statistics.median(lat):.0f} ms, p95 {p95:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
