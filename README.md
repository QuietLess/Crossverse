# CrossVerse: cross-domain entertainment recommendations

> *If you liked this movie, what game should you play next — and vice versa?*

CrossVerse is a recommender that learns taste across **movies/TV and video games** from real
shared-user behaviour: 119k Amazon users who liked items in both categories. It serves
within-domain, movie→game, game→movie, mixed-profile and cold-start recommendations through a
two-stage retrieval + LambdaRank pipeline. Every recommendation carries an evidence object
(shared themes, co-like counts, embedding similarity) instead of an invented LLM paragraph.

```
$ crossverse recommend "The Dark Knight" "Batman Begins" --to game
 1. [game]  Batman: Arkham Asylum (2009)
      Recommended because 16 people who liked The Dark Knight also liked Batman: Arkham Asylum;
      its story and setting resemble Batman Begins; it shares your taste for superhero, crime, mystery.
 2. [game]  Batman: Arkham City - Game of The Year Edition (2010)
 3. [game]  Spider-Man: Web of Shadows (2007)

$ crossverse recommend "Witcher 3: Wild Hunt" "Baldur's Gate: Enhanced Edition" --to movie
 1. [movie] Game Of Thrones: Season 6     (8 people who liked Witcher 3 also liked it; fantasy)
 2. [movie] Warcraft (2016)               (story and setting resemble Witcher 3)
 3. [movie] Kingsglaive - Final Fantasy XV
```

## Results

Production model **v7**, on held-out test users, NDCG@10. The lift column is a paired bootstrap on
the same users (2,000 resamples); "n.s." means the 95% CI includes zero. Full tables:
[reports/v7/benchmark.md](reports/v7/benchmark.md).

| task | users | popularity | best single retriever | CrossVerse | lift vs popularity (95% CI) |
|---|---:|---:|---|---:|---|
| Movies → movies | 3000 | 0.0097 | item_knn 0.0412 | 0.0454 | +371% [+0.0300, +0.0419] |
| Games → games | 3000 | 0.0203 | als_single_domain 0.0589 | 0.0624 | +207% [+0.0351, +0.0491] |
| **Movies → games** (game history fully hidden) | 2504 | 0.0290 | two_tower 0.0251 | 0.0338 | **+16%** [+0.0014, +0.0080] |
| **Games → movies** (movie history fully hidden) | 2504 | 0.0216 | two_tower 0.0188 | 0.0267 | **+24%** [+0.0022, +0.0083] |
| Mixed profile → both | 3000 | 0.0062 | item_knn 0.0380 | 0.0412 | +562% [+0.0296, +0.0402] |
| Cold start (themes only) | 3000 | 0.0067 | content 0.0003 | 0.0066 | −1% [−0.0027, +0.0025], n.s. |

What this says, plainly:

* **Cross-domain transfer works in both directions.** For users whose games (or movies) the model
  never saw, it beats popularity significantly: +16% movies→games, +24% games→movies. With the
  first, smaller dataset (36k bridge users), movies→games was only a statistical tie; more shared
  users made the difference.
* **The two-stage design pays off.** The ranker beats every single retriever on every behavioural
  task. It also significantly beats reciprocal-rank fusion of the same candidates on movies→games
  (+158%), games→movies (+30%), mixed profiles (+24%) and movies→movies (+35%).
* **Cold start ties popularity.** Stated themes alone don't predict Amazon purchases better than
  popularity, but CrossVerse honours an explicit "I want cyberpunk" at no measurable cost.
* **Not everything worked, and the gate caught it.** A routed cross-domain-specialist ranker (v5)
  and validation-tuned retriever settings (v3, v6) both lost to the plain recipe on the full
  pipeline. The promotion gate (primary metric + per-task guardrails) blocked all three. See
  [docs/evaluation.md](docs/evaluation.md#model-versions).

Serving (single worker, laptop): 12–16 ms engine time per uncached recommendation; p50 19 ms /
p95 21 ms over HTTP, with the Postgres recommendation log and the Redis cache in the loop.

The raw cross-domain signal is real and interpretable ([reports/eda.md](reports/eda.md)). Among
bridge users, *Final Fantasy VII: Advent Children* ↔ *Kingdom Hearts II* co-occur 38× more often than
chance, and *Kingsglaive* ↔ *Final Fantasy XV* 50×.

## What's inside

| blueprint area | implementation |
|---|---|
| Data (§3) | Amazon Reviews 2023 `Movies_and_TV` + `Video_Games`; every user with ≥1 liked movie and ≥1 liked game; metadata streamed; accessories, add-ons and **DVDs misfiled under Video Games** filtered out. [docs/data.md](docs/data.md) |
| Canonical items (§4) | Format/platform/edition/season stripping (incl. ordinal seasons), release-year splitting of same-titled games, ASIN→canonical map, **two hand-labelled audits** (merge precision 92% games / 80% movies). `src/crossverse/data/canonical.py` |
| Baselines (§5.1–5.2) | Popularity, item-kNN, single-domain and joint implicit ALS (vectorised conjugate gradient, exact fold-in), content (TF-IDF/SVD + themes), movie↔game **co-preference** transfer |
| Two-tower (§5.4) | Domain-aware item towers (id embedding + projected content), shared user encoder with signed/decayed history and domain mix, sampled softmax with domain-matched negatives, cross-domain context dropout. Pure NumPy. |
| Ranker (§5.5) | LightGBM LambdaRank on ~480-candidate unions, with 45 features: per-retriever score/rank/z, cross-domain, semantic, theme, quality and context. Trained on validation tasks. |
| Post-ranking | Eligibility filter (no memberships, gift cards, DLC, amiibo/cases), MMR diversity, creator repetition penalty, domain quotas for unified lists, explicit-intent tiers for cold start |
| Evaluation (§6) | Chronological per-user split **plus cross-domain holdout users whose target domain is entirely hidden from training**; Recall/NDCG/HitRate/MAP, coverage, diversity, novelty, popularity bias; segments; paired-bootstrap CIs; 100-case error analysis. Parallel (≈2.5 min). [docs/evaluation.md](docs/evaluation.md) |
| Explainability (§7) | Deterministic evidence objects + template verbalisation that only states what clears a threshold |
| Serving (§8–9) | FastAPI: `/recommend`, `/recommend/movie-to-game`, `/recommend/game-to-movie`, `/similar/{domain}/{id}`, `/feedback`, `/explain/{id}`, `/health`, `/metrics`, `/items/search`, `/admin/stats`; "did you mean" suggestions for unknown titles |
| Storage | Postgres (or SQLite locally) for the recommendation log + feedback, Redis (or in-process) cache; integration-tested against real servers |
| MLOps (§8, §12) | Versioned registry with dataset fingerprints, run tracking (MLflow if installed, JSON always), promotion gate on cross-domain NDCG + per-task guardrails that refuses cross-dataset comparisons, Prometheus metrics + provisioned Grafana dashboard |
| Demo (§13) | Streamlit app (five modes + evidence panel + admin view), CLI |
| CI (§12) | GitHub Actions with pinned `constraints.txt`: ruff, mypy, pytest (78 tests incl. Postgres/Redis integration via service containers), synthetic end-to-end train→evaluate→gate, gitleaks, Docker build + API smoke test |

What was verified, and how (including the Compose stack run natively because Docker is unavailable
on the dev machine), is in [docs/verification.md](docs/verification.md).

## Quickstart

```bash
pip install -e ".[dev,web]" -c constraints.txt

# Real data (~830 MB download; build ~6 min, training ~5 min, full benchmark ~2.5 min on 16 cores)
python pipelines/build_dataset.py --download
python pipelines/train.py --version v1
python pipelines/evaluate.py --version v1       # -> reports/v1/{benchmark,error_analysis}.md + significance
python pipelines/publish.py --version v1        # gate + guardrails -> artifacts/models/PRODUCTION

# ...or fully offline in ~30 s with the synthetic fixture (planted cross-domain taste)
python pipelines/build_dataset.py --synthetic

uvicorn apps.api.main:app --port 8000           # API docs at http://localhost:8000/docs
streamlit run apps/web/app.py                   # demo UI (falls back to in-process engine)
crossverse recommend "Blade Runner 2049" "Ex Machina" --to game

docker compose -f infra/docker-compose.yml up --build   # API + web + Postgres + Redis + Prometheus + Grafana
```

Common targets live in the [Makefile](Makefile): `make pipeline`, `make eda`, `make serve`, `make web`, `make test`, `make compose`.

### API example

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{
  "liked": [{"item": "Blade Runner 2049"}, {"item": "Mass Effect Trilogy", "rating": 5}],
  "disliked": ["Twilight"],
  "k": 10
}'
```

Each item comes back with `evidence`:

```json
{"shared_themes": ["sci-fi", "space"], "genre_overlap": 0.4, "users_who_liked_both": 12,
 "semantic_similarity": 0.52, "anchors": [{"title": "Mass Effect Trilogy", "users_who_liked_both": 12, ...}],
 "candidate_sources": ["two_tower", "copref", "content"], "reason_confidence": "high",
 "summary": "Recommended because 12 people who liked Mass Effect Trilogy also liked ...; it shares your taste for sci-fi, space."}
```

Unknown or misspelled titles come back in `unresolved` with `suggestions`, e.g. `"Witcher 3 Wild Hnt"
→ ["Witcher 3: Wild Hunt", ...]`.

Titles that exist as both a movie and a game (*Batman Begins*) resolve to the more popular one and
the other comes back in `ambiguous`. To pick one, set `"domain": "game"` on the profile item or
prefix the title (`"game: Batman Begins"`). A trailing year (`"Dune (2021)"`) is used as a hint.

## Repository layout

```
src/crossverse/
  data/         ingestion (amazon.py), canonicalisation, data contracts, synthetic fixture
  features/     shared cross-domain theme vocabulary
  retrieval/    popularity, item-kNN, ALS, co-preference, content, two-tower
  ranking/      candidate union, feature builder, LightGBM LambdaRank (+ routed option), diversity
  evaluation/   metrics, leakage-safe protocol, (parallel) benchmark, significance, error analysis
  explain/      evidence objects
  monitoring/   model registry + gate, Prometheus metrics
  serving/      engine, FastAPI app, storage adapters
apps/api, apps/web            ASGI entrypoint, Streamlit demo
pipelines/                    build_dataset → train → evaluate → publish (+ eda)
infra/                        Dockerfile, compose, Prometheus, Grafana
tests/                        unit, data-contract, leakage, golden, API, storage-backend integration
docs/                         architecture, data, evaluation, verification, portfolio, audit/
reports/                      per-version benchmarks, significance, error analysis, EDA
```

## Honest limitations

* **Amazon ratings are purchases plus reviews**, not watch/play logs. 87% are ≥4 stars, so
  "positive" mostly means "bought it", and popularity is a strong baseline. All gains are reported
  relative to it, with confidence intervals.
* **Canonicalisation** is title-based. Merge precision is 92% for games and 80% for movies (audit
  in `docs/audit/`). The remaining movie errors are different works sharing a title with no year to
  tell them apart (*Starman* film vs TV series). Fixing those needs an external ID source
  (TMDB/IMDb), as the blueprint suggests.
* The opposite error exists too: some editions of one work stay separate items (*Red Dead
  Redemption* and its GOTY edition; *Game of Thrones: Season 6* and the complete collection).
  Display titles are cleaned at load time ("Avengers 4k UHD BLURAY Digital Steelbook" →
  "Avengers"), which makes these show up as same-titled items (~590 groups). Same-titled games
  get their release year; movies don't, because Amazon's movie years are mostly DVD dates.
* A handful of misfiled non-games without any media metadata remain in the game catalog (e.g. a
  Hallmark movie listed under PC games); they show up in `reports/eda.md`.
* Disco Elysium is in the raw data but has too few likes from bridge users to pass the item filter.
  Unknown titles get "did you mean" suggestions instead of silent substitutions.
* The Docker image and `docker compose up` were not run (no virtualisation on the dev machine);
  every service in the stack was verified natively at the same versions instead.
* Vector search is brute force in-process: fine at 43k items. pgvector/Qdrant are the next step at
  catalog scale.

## Sources

* Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (Amazon Reviews 2023),
  https://amazon-reviews-2023.github.io/ ; UCSD datasets page https://cseweb.ucsd.edu/~jmcauley/datasets.html
* MovieLens 32M (not used for training: user ids do not align) https://grouplens.org/datasets/movielens/32m/
* IGDB API (optional metadata enrichment) https://api-docs.igdb.com/

Datasets are downloaded by code and never committed. Re-check each source's license before
redistributing derived data.
