# Notebooks

Exploration only; no production logic lives here (blueprint §10).

The reproducible data audit is a script rather than a notebook, so it runs in CI and its output can be diffed:

```bash
python pipelines/eda.py        # -> reports/eda.md (funnel, sparsity, popularity concentration, top movie<->game pairs)
```

For ad-hoc exploration, load the processed tables and a trained engine:

```python
import pandas as pd
from crossverse.serving.engine import CrossVerseEngine
inter = pd.read_parquet("data/processed/interactions.parquet")
items = pd.read_parquet("data/processed/items.parquet")
engine = CrossVerseEngine.load("artifacts/models/v3/engine.pkl")
engine.recommend([(engine.resolve("The Dark Knight"), 5.0)], target_domain="game", k=10)
```
