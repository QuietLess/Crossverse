PY ?= python
export PYTHONPATH := src$(if $(filter Windows_NT,$(OS)),;,:).

.PHONY: install data data-download synthetic eda train evaluate publish pipeline serve web test lint smoke compose clean

install:            ## install package + dev/web extras
	$(PY) -m pip install -e ".[dev,web]"

data-download:      ## download Amazon Reviews 2023 (Movies_and_TV + Video_Games, ~830MB) and build
	$(PY) pipelines/build_dataset.py --download

data:               ## build data/processed from already-downloaded raw files
	$(PY) pipelines/build_dataset.py

synthetic:          ## offline synthetic fixture (no download)
	$(PY) pipelines/build_dataset.py --synthetic

eda:                ## data audit (M1) -> reports/eda.md
	$(PY) pipelines/eda.py

train:              ## split + retrievers + ranker -> artifacts/models/<version>
	$(PY) pipelines/train.py

evaluate:           ## benchmark latest version on held-out test tasks -> reports/
	$(PY) pipelines/evaluate.py

publish:            ## promote latest version through the cross-domain NDCG gate
	$(PY) pipelines/publish.py

pipeline: data train evaluate publish

serve:              ## FastAPI on :8000
	$(PY) -m uvicorn apps.api.main:app --port 8000

web:                ## Streamlit demo on :8501
	$(PY) -m streamlit run apps/web/app.py

test:
	$(PY) -m pytest

lint:
	ruff check src tests pipelines apps scripts

smoke:              ## smoke + light load test against a running API
	$(PY) scripts/smoke_test.py --url http://localhost:8000

compose:            ## full stack: API, web, Postgres, Redis, Prometheus, Grafana
	docker compose -f infra/docker-compose.yml up --build

clean:
	rm -rf artifacts/models/*/split.pkl .pytest_cache .ruff_cache
