# Verification log

What has been checked, how, and what could not be checked on the development machine (Windows 11,
no Docker). Dates: 2026-10-05.

## New-release refresh, 2026-10-18

| check | result |
|---|---|
| `pytest` | 313 passed, 3 skipped |
| Security review of the refresh | 4 MEDIUM findings fixed with regression tests: an index built for another model was served (500s), public scores showed on the wrong titles after a reload, a Twitch auth error put the IGDB secret in the error text (and so the job's logs), the job and the public service shared one over-privileged account. A re-review confirmed the fixes and found 3 LOW issues (a concurrent run's unfinished version could be pruned, a transient read error wasn't retried, `Retry-After: nan` crashed the job), fixed with tests. Other LOW items are follow-ups in ROADMAP |
| AniList query against the live API (one page, `since=2025-07-01`) | 25 of 50 titles kept (later seasons and specials dropped), e.g. *Chainsaw Man – The Movie: Reze Arc* as a film, *Gachiakuta* as a series, each with "anime" as first theme |
| Service with a published index (synthetic engine, those AniList titles, headless Chrome) | `/health` shows the published version; under Movies the New releases row shows the anime films with Film + Anime + New chips |
| TMDB and IGDB against the live APIs | not run: no keys on this machine |
| `scripts/deploy_refresh_job.py` | dry run only (unit test with a fake gcloud); nothing created in Google Cloud |

## Re-run 2026-10-18 (macOS, Python 3.12, `constraints.txt` versions)

| check | result |
|---|---|
| `ruff check src tests pipelines apps scripts` | All checks passed |
| `mypy src/crossverse --ignore-missing-imports` | no issues in 42 source files |
| `pytest` (no Postgres/Redis) | 294 passed, 3 skipped (the storage-backend integration tests) |

Found and fixed: requests that match no API route were labelled with their raw URL in Prometheus, so
50 probes of random paths created 50 time series (reproduced against the app with the synthetic
fixture); they now share `<unmatched>`. Request lists, titles and feedback context had no size limits
(a 2 MB `context` was stored); they are now bounded and rejected with 422.

A security review of that change found that bounded input still allowed unbounded CPU. Measured with
the engine's own search/suggest code on a synthetic 42.6k-title catalog (the size of ds3), worst case
before → after:

| path | before | after |
|---|---:|---:|
| "did you mean" for one unknown 300-character title of common words | 8.1 s | 0.08 s |
| one `/recommend` with 200 such titles (now: suggestions for 5 distinct ones) | ~1,600 s | 0.4 s |
| one Letterboxd name of a repeated common word (now an exact-title lookup) | 0.28 s | 0.0001 s |
| a 5,000-name Letterboxd import of such names | ~1,400 s | ~1 s |
| a 5,000-name import with a 3,000-title new-release index attached (as in production) | 18–30 s | 0.1–1.2 s |

A first version of the fix was timed without a new-release index and missed that every name was also
scanned against all new releases; the security re-review caught it. One new-release search (type-ahead,
title resolution) still scans the index: ~7 ms per call for 4,000 titles.

The same review found that a cached `/recommend` response returned the first requester's unknown
titles and inputs; cache hits now carry the current request's own.

The tables below are the original Windows run (2026-10-05).

## Code quality and CI

| check | how | result |
|---|---|---|
| Unit, data-contract, leakage, golden, API and storage-backend tests | `pytest` with Postgres + Redis running | 78 passed |
| Lint / types | `ruff check`, `mypy src/crossverse` | clean |
| CI workflow, emulated | fresh virtualenv, `pip install -e ".[dev]" -c constraints.txt`, then the exact commands of `.github/workflows/ci.yml` incl. the synthetic build → train → evaluate → gate job | passed |
| Secret scan | gitleaks 8.30.1 `gitleaks dir` over the repository source | no leaks found |
| Dockerfile lint | hadolint 2.12.0 | clean (DL3008 apt-pin warning documented and ignored on purpose) |
| Config syntax | PyYAML / JSON parse of CI, Compose, Prometheus and Grafana files | valid |

The clean-venv run found a real problem: an unpinned install pulled numpy 2.5.3, whose type stubs
crash mypy 2.4.0, so CI would have gone red on an arbitrary day. `constraints.txt` now pins the
tested versions for CI and the Docker image.

## Storage backends (real servers)

PostgreSQL 16.4 (official portable binaries) and Redis 5.0.14 ran locally.
`tests/test_storage_backends.py` runs when `CROSSVERSE_TEST_PG_URL` / `CROSSVERSE_TEST_REDIS_URL`
are set; CI provides both as service containers.

* 3/3 passed: Postgres recommendation log + feedback round trip, Redis cache round trip, and the
  full API with Postgres + Redis (cache hit returns identical results; `/explain` reads back from
  Postgres).
* Found and fixed: redis-py ≥ 8 negotiates RESP3 (`HELLO 3`), which Redis < 6 rejects, so the
  health check failed against older or managed Redis. The client now pins RESP2.

## The Compose stack, without Docker

Docker Desktop cannot run on the dev machine: hardware virtualisation is disabled in firmware,
WSL is not installed, and the session has no administrator rights. Each service of
`infra/docker-compose.yml` was therefore run natively, at the same versions, with the project's
config files. The only change was hostnames (`api` → `127.0.0.1`, `prometheus` → `127.0.0.1`),
because there is no Docker DNS.

| service | result |
|---|---|
| API (uvicorn, Compose env: `CROSSVERSE_DATABASE_URL`, `CROSSVERSE_REDIS_URL`) | `/health` ok, backends `postgres` + `redis`; smoke/load test passed, **p50 32 ms / p95 45 ms** uncached, sequential |
| Prometheus 2.53.0 with `infra/prometheus/prometheus.yml` | target `up`; per-endpoint counters scraped |
| Grafana 11.1.0 with `infra/grafana/provisioning` | datasource and dashboard provisioned (folder *CrossVerse*, 9 panels); every panel query executed: all valid. The error-rate panel showed "No data" when there were zero errors; fixed with `or vector(0)` |
| Streamlit (`apps/web/app.py`) | server boots and is healthy; all 6 tabs render and the movie→game and cold-start flows run without exceptions (Streamlit `AppTest`) |

**Not verified:** building the container image and running `docker compose up`. To verify on a
machine with Docker:

```bash
make train publish                      # or copy artifacts/ from a trained machine
docker compose -f infra/docker-compose.yml up --build
python scripts/smoke_test.py --url http://localhost:8000
open http://localhost:3000              # Grafana dashboard "CrossVerse — service & recommendation quality"
```

On this machine that requires enabling virtualisation (Intel VT-x / AMD-V) in BIOS/UEFI, then
`wsl --install` and Docker Desktop, all of which need administrator rights.

## Evaluation tooling

* Parallel benchmark (`--jobs`) reproduces the sequential numbers exactly (max abs. difference 0.0).
* Parallel ranker-data generation produces an identical ranker (same rows, same feature importances).
* Vectorised conjugate-gradient ALS matches exact ALS quality (validation NDCG within ±0.0007) and
  fits 8× faster.
