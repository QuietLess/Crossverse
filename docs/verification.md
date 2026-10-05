# Verification log

What has been checked, how, and what could not be checked on the development machine (Windows 11,
no Docker). Dates: 2026-10-05.

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
