# Roadmap

## Done

| when | milestone |
|---|---|
| 2026-10-05 | Data pipeline (Amazon Reviews 2023, bridge users, canonical titles with audits), retrievers, two-tower, LambdaRank ranker, leakage-safe evaluation, registry + promotion gate, FastAPI, Streamlit, CI, Compose stack |
| 2026-10-06 – 07 | Larger dataset (ds3), cover images, minimum-support co-likes, taste setting, judged evaluation set, semantic retriever, TMDB/IGDB metadata (v7–v12 benchmarks) |
| 2026-10-08 | Public web UI, Cloud Run + Firebase Hosting deployment, TV as its own target |
| 2026-10-09 – 10 | "Same universe" adaptations and tie-ins; new releases (post-2023 titles from TMDB/IGDB) |
| 2026-10-12 – 13 | Slider Balanced → Same DNA, public scores on cards, Letterboxd import |
| 2026-10-18 | TMDB logo in the footer (TMDB terms §3); scheduled new-release refresh with AniList anime as a tag, published index reloaded by the service (code and dry-run deploy script; cloud resources not yet created) |
| 2026-10-18 | API hardening: bounded request sizes and title-lookup work, one metrics label for unmatched paths, per-request fields on cache hits; SPEC/ROADMAP |

## Open decision: TMDB terms

TMDB's API terms (updated 2023-10-20) prohibit using TMDB content in, "including for training", a
machine-learning or AI application (§1.C), count AI training as commercial use (§2.A), cap caching at
6 months (§1.C) and require the TMDB logo (§3; the app shows the text notice but no logo). Models v12+
use TMDB metadata in training, and new releases are matched by embeddings of TMDB overviews. Options:
ask TMDB for permission, keep TMDB for display only (scores, listings, links) and retrain without it,
or replace it with sources whose terms allow this use. Owner to decide; not legal advice.

## Next (proposed, not yet agreed)

* **Turn on the scheduled refresh**: `python scripts/deploy_refresh_job.py --apply` (needs gcloud on
  the deploy machine, the production model to redeploy the service image with the new code, and the
  owner's go-ahead for the small recurring cost). Design notes below.
* **Scheduled new-release refresh (2×/week)**, built 2026-10-18. Fetch titles released since the last run,
  embed them, rebuild the new-releases index and publish it without retraining the ranker. Retraining on a
  schedule would change nothing: the behaviour data (Amazon 2023) is frozen, and no source with fresh,
  legally usable movie + game likes was found (2026-10-18 research). Candidate sources: AniList
  (anime: seasons and airing schedule, no key; no bulk mirroring), TVmaze (`/updates/shows`, streaming
  schedule; CC BY-SA credit), IGDB `release_dates` (current), TMDB `upcoming`/`on_the_air`/`/changes`
  (subject to the decision above), RAWG as a games cross-check (backlink required), Wikidata (CC0) for
  cross-source IDs and franchises. Not usable: Letterboxd (no API access for recommendation projects,
  scraping banned), Metacritic and Rotten Tomatoes (paid licences only), Jikan (scrapes MyAnimeList).
* **Refresh follow-ups** (security review, LOW): read the new-release index once per request so a reload
  mid-request can't mix two versions (scores are re-keyed just after the swap, too); alert on failed job runs; stop serving a published index older than 6 months (TMDB terms) if
  the job keeps failing; accept cover images only over https from known hosts and send
  `Referrer-Policy: no-referrer`; let `deploy_refresh_job.py` update an existing schedule; keep TMDB/IGDB
  titles when only AniList is down; fetch public scores for new titles in the job; confirm the site stays
  non-commercial (AniList and TMDB terms) and drop the permanent local AniList cache of
  `pipelines/new_releases.py`.
* **Durable feedback log** (optional, costs money): a hosted Postgres for the public deployment.
* **Rate limiting** for the public API (per-client token bucket, or Cloud Armor) and a cap on the
  local log's size; reject deeply nested feedback `context`.
* **Type-ahead cost**: a one-letter query ("a") ranks every title with a word starting with it; cap
  the shortlist like "did you mean". The new-release search is a linear scan (~7 ms per call at 4,000
  titles; up to 200 per `/recommend`, ~1.4 s): give it the catalog's prefix index. Among new releases with
  the same title, prefer the requested year before votes in imports (the catalog already does).

## Later ideas

* External IDs (TMDB/IMDb) to fix same-title movie merges (80% merge precision today).
* Retraining on fresh behaviour data, if a legally usable source of cross-medium likes exists.
* pgvector/Qdrant once the catalog outgrows in-process brute-force search.
