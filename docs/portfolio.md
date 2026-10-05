# Portfolio notes (blueprint §15 and M8)

## CV entry

**CrossVerse: Cross-Domain Recommendation Platform** · Python, LightGBM, NumPy/SciPy, FastAPI, PostgreSQL/Redis, Prometheus/Grafana, Docker, GitHub Actions

- Built a hybrid recommender that learns movie/TV ↔ video-game taste from 1.6M Amazon interactions
  of 119k shared users, after resolving 260k product listings (DVD/Blu-ray/platform editions,
  misfiled DVDs, accessories) into canonical titles, with hand-audited merge precision of 92% for games.
- Designed a two-stage pipeline: 8 retrievers (implicit ALS with a vectorised CG solver, item-kNN,
  cross-domain co-preference, content embeddings, NumPy two-tower model) feed a LightGBM
  LambdaRank ranker with 45 features. It beats every single retriever and beats rank fusion of the
  same candidates by up to +158%.
- Built a leakage-safe evaluation in which the target domain of held-out users is entirely hidden
  from training, with paired-bootstrap confidence intervals. Cross-domain lift over popularity is
  **+16% movies→games and +24% games→movies (both significant)**; within-domain and mixed-profile
  NDCG@10 improves **+207% to +562%**.
- Shipped FastAPI recommendation + explanation APIs (evidence objects, not LLM text; 12–16 ms
  engine latency), a model registry with dataset fingerprints, a promotion gate plus per-task
  guardrails (which blocked three of my own regressing candidates), Prometheus/Grafana monitoring
  and CI with pinned dependencies and Postgres/Redis integration tests.

## 3-minute recruiter walkthrough

1. The problem fits in one sentence: "If you liked this movie, what game should you play next?"
2. The data is real shared-user behaviour, not two datasets stitched together (README › Results, `reports/eda.md`).
3. The honest evaluation: held-out users with their target domain fully hidden, plus significance tests.
4. Demo: Streamlit, "The Dark Knight" → Batman: Arkham Asylum, with "16 people who liked…" evidence.
5. Engineering: the registry, gate and guardrails blocked v3/v5/v6, with a Grafana dashboard and CI.

## Medium draft outline: "Can your movie taste pick your next video game?"

1. Hook: Final Fantasy VII: Advent Children ↔ Kingdom Hearts II co-occur 38× more often than chance.
2. Why cross-domain is hard: no shared catalog, product ≠ entity (canonicalisation), sparse bridge users,
   and sellers who list DVDs under "Video Games".
3. The trap of easy evaluation: random splits and seen users. Our protocol hides the target domain.
4. Architecture: retrieval union → LambdaRank → business rules → evidence. Why each stage earns its place
   (ranker vs reciprocal-rank fusion: +158% on movies→games, significant).
5. Results with error bars, including what did *not* work: a routed specialist ranker, standalone-tuned
   retrievers, and theme-only cold start. Also how 3.4× more shared users turned movies→games from a
   tie into a significant win.
6. MLOps: a promotion gate that said "no" to three of our own models, and refused to compare models
   trained on different datasets.
7. What's next: external entity IDs (TMDB/IGDB) for canonicalisation, sequence models, an online A/B
   feedback loop.
