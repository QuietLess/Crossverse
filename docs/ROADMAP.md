# Roadmap

## Done

| when | milestone |
|---|---|
| 2026-10-05 | Data pipeline (Amazon Reviews 2023, bridge users, canonical titles with audits), retrievers, two-tower, LambdaRank ranker, leakage-safe evaluation, registry + promotion gate, FastAPI, Streamlit, CI, Compose stack |
| 2026-10-06 – 07 | Larger dataset (ds3), cover images, minimum-support co-likes, taste setting, judged evaluation set, semantic retriever, TMDB/IGDB metadata (v7–v12 benchmarks) |
| 2026-10-08 | Public web UI, Cloud Run + Firebase Hosting deployment, TV as its own target |
| 2026-10-09 – 10 | "Same universe" adaptations and tie-ins; new releases (post-2023 titles from TMDB/IGDB) |
| 2026-10-12 – 13 | Slider Balanced → Same DNA, public scores on cards, Letterboxd import |
| 2026-10-18 | API hardening: bounded request sizes, one metrics label for unmatched paths; SPEC/ROADMAP |

## Next (proposed, not yet agreed)

* **Scheduled new-release refresh.** Fetch new films, series, anime and games on a schedule and update
  the "New releases" row without a full retrain. Source research in progress.
* **Durable feedback log** (optional, costs money): a hosted Postgres for the public deployment.

## Later ideas

* External IDs (TMDB/IMDb) to fix same-title movie merges (80% merge precision today).
* Retraining on fresh behaviour data, if a legally usable source of cross-medium likes exists.
* pgvector/Qdrant once the catalog outgrows in-process brute-force search.
