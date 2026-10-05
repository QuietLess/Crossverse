# Architecture

## Offline (training) path

```
Amazon Reviews 2023 (Movies_and_TV, Video_Games)        pipelines/build_dataset.py
  rating_only CSVs + streamed metadata
        │  bridge-user selection, accessory filtering, canonical item resolution, data contracts
        ▼
data/processed/{interactions,items,asin_map}.parquet + canonical_audit_sample.csv + data_report.json
        │
        ▼                                                pipelines/train.py
make_split()  ── per-user chronological split  +  cross-domain holdout groups (target domain fully hidden)
        │
        ├── Stage A retrievers, fit on train only
        │     popularity · item-kNN (joint) · ALS (joint, movie-only, game-only)
        │     co-preference (movie↔game transfer) · content (TF-IDF/SVD + themes) · two-tower (NumPy)
        │
        ├── Stage B ranker: LightGBM LambdaRank on *validation* tasks (incl. cross-domain holdout users)
        │
        ▼
artifacts/models/<version>/{engine.pkl, split.pkl, manifest.json}
        │                                                pipelines/evaluate.py
        ▼
reports/{benchmark.md, benchmark.csv, error_analysis.md}  → manifest.test_metrics → MLflow / runs/*.json
        │                                                pipelines/publish.py
        ▼
model gate (cross_domain_ndcg@10 may not drop > tolerance) → artifacts/models/PRODUCTION
```

## Online (serving) path

```
Web UI (Streamlit)  ──HTTP──▶  FastAPI (crossverse.serving.api)
                                  │  resolve titles → catalog ids
                                  │  cache lookup (Redis | in-process TTL)
                                  ▼
                       CrossVerseEngine.recommend()
                         1. CandidateGenerator: every retriever scores the catalog; top-N per source
                            in the target domain(s) → union (≈150–500 candidates)
                         2. FeatureBuilder: 3 features per retriever (score, rank, z) + behaviour,
                            cross-domain, semantic, context, quality features
                         3. LightGBM LambdaRank score
                         4. Post-ranking rules: eligibility filter (no memberships, gift cards,
                            DLC/season passes, amiibo/cases), MMR diversity (franchise/near-duplicate
                            penalty), creator repetition penalty, domain quotas for unified lists,
                            explicit-intent tiers for cold start
                         5. Explainer: evidence object from the signals above (no free generation)
                                  │
                                  ├──▶ Postgres | SQLite: recommendations log, feedback
                                  └──▶ Prometheus metrics → Grafana dashboard
```

All models implement one contract — `score(history) -> scores for every catalog item` — so the
same code serves offline fold-in evaluation, anonymous API profiles, and cold start.

## Retrievers

| name | what it captures | cross-domain? |
|---|---|---|
| `popularity` | most-liked items | – |
| `item_knn` | item-item cosine (shrunk) on the joint movie+game matrix | yes, via shared users |
| `als` | implicit ALS on the joint matrix (vectorised conjugate-gradient solver, 8× faster than per-user solves at equal quality), exact fold-in | yes |
| `als_movie`, `als_game` | single-domain ALS (blueprint baseline) | no |
| `copref` | movie↔game co-like counts among bridge users, popularity-normalised (§5.2) | only |
| `content` | TF-IDF/SVD metadata embeddings + shared theme vocabulary | yes, semantic |
| `two_tower` | domain-aware item towers (learned id embedding + projected content) and a shared user encoder, sampled softmax with domain-matched negatives and cross-domain-only context dropout | yes |

## Vector index

Item embeddings live in an in-process dense matrix (numpy) and are scored by matrix multiplication:
at ~26k items this takes about a millisecond and needs no extra service. `/health` reports the index
type. The retriever contract hides the index, so pgvector/Qdrant can be added behind it when the
catalog outgrows brute force.

## Explainability

`crossverse.explain.evidence.Explainer` packages the signals that already drove the ranking:
shared themes, theme overlap (Jaccard), number of bridge users who liked both the anchor and the
candidate, embedding similarity, nominating retrievers, overlap with disliked themes, and a
confidence label. The text summary comes from a template over that evidence. An LLM could
verbalise the same object later, but the evidence object stays the source of truth.

## Throughput engineering

* **Parallel evaluation.** `pipelines/evaluate.py --jobs N` fans test cases out over processes. Each
  worker loads the engine once, and per-case rows are aggregated in the parent, so the numbers are
  identical to a sequential run (verified to 0.0 difference). A full benchmark (≈17k cases × 10
  models) takes about 2.5 min instead of about 30.
* **Parallel ranker data.** `pipelines/train.py --jobs N` builds the LambdaRank training set in
  worker processes with the same shuffled order and per-task caps, producing an identical ranker.
* **Engine latency.** Profile vectors are cast to float32 before scoring (mixed-dtype matmul used to
  copy the 40k × d item matrix on every call), candidate ranks use partial sorts, and title search
  uses a token-prefix index (< 1 ms). Uncached recommendations take 12–15 ms of engine time.

## Unknown titles

`CrossVerseEngine.suggest` offers "did you mean" candidates (token-prefix candidates ranked by
string similarity). The API returns them in `suggestions`, and in the 422 error when nothing in a
profile resolves.

## Options that were tried and are off by default

* `routed_ranker` trains a separate LambdaRank model for pure cross-domain queries. It lost to the
  joint ranker (v5 vs v4): alone it early-stops after about 11 trees on ~3k queries, while the joint
  model shares structure across query types.
* Validation-optimal retriever knobs (co-preference α = 0, kNN shrinkage 500) improve the retrievers
  standalone but made the full pipeline worse on two datasets (v3, v6).
