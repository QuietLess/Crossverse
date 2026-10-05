"""Evaluate any `recommend(case, target_domain, k) -> ranked item indices` function on tasks.

Scoring is split into a per-case step (`score_cases`, embarrassingly parallel) and an aggregation
step (`aggregate`), so the benchmark can fan cases out over processes and still produce exactly the
same numbers as a sequential run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from crossverse.evaluation import metrics as M
from crossverse.evaluation.protocol import EvalCase, EvalTask

RecommendFn = Callable[[EvalCase, "str | None", int], np.ndarray]


def popularity_percentiles(popularity: np.ndarray) -> np.ndarray:
    order = popularity.argsort().argsort()
    return order / max(len(order) - 1, 1)


def score_cases(recommend: RecommendFn, target_domain: str | None, cases: Sequence[EvalCase],
                ks: tuple[int, ...] = (10, 20, 50), embeddings: np.ndarray | None = None,
                popularity: np.ndarray | None = None, n_users: int = 1) -> list[dict]:
    """One row per case: metrics, segments and the top-10 (for coverage)."""
    kmax = max(ks)
    pop_pct = popularity_percentiles(popularity) if popularity is not None else None
    rows = []
    for case in cases:
        ranked = list(recommend(case, target_domain, kmax))
        row: dict = {"user_id": case.user_id, "segments": case.segments or ("all",), "top10": ranked[:10]}
        for k in ks:
            row[f"recall@{k}"] = M.recall_at_k(ranked, case.relevant, k)
            row[f"ndcg@{k}"] = M.ndcg_at_k(ranked, case.relevant, k)
            row[f"hit@{k}"] = M.hit_rate_at_k(ranked, case.relevant, k)
        row["map@10"] = M.average_precision_at_k(ranked, case.relevant, 10)
        if embeddings is not None:
            row["diversity@10"] = M.intra_list_diversity(ranked[:10], embeddings)
        if popularity is not None and pop_pct is not None:
            row["novelty@10"] = M.novelty(ranked[:10], popularity, n_users)
            row["pop_pct@10"] = M.popularity_percentile(ranked[:10], pop_pct)
        rows.append(row)
    return rows


def aggregate(rows: list[dict], n_items: int | None = None) -> dict[str, dict[str, float]]:
    """{segment: {metric: mean}} plus n per segment and catalog coverage@10 for "all"."""
    per_seg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    recommended: set[int] = set()
    for row in rows:
        recommended.update(int(i) for i in row["top10"])
        for seg in row["segments"]:
            for m, v in row.items():
                if m not in ("user_id", "segments", "top10"):
                    per_seg[seg][m].append(v)
    out = {seg: {m: float(np.mean(v)) for m, v in ms.items()} for seg, ms in per_seg.items()}
    for seg in out:
        out[seg]["n"] = len(per_seg[seg]["map@10"])
    if "all" in out and n_items:
        out["all"]["coverage@10"] = len(recommended) / n_items
    return out


def case_record(task: str, model: str, row: dict) -> dict:
    return {"task": task, "model": model, "user_id": row["user_id"],
            "ndcg@10": row.get("ndcg@10", np.nan), "recall@20": row.get("recall@20", np.nan)}


def evaluate_task(recommend: RecommendFn, task: EvalTask, ks: tuple[int, ...] = (10, 20, 50),
                  embeddings: np.ndarray | None = None, popularity: np.ndarray | None = None,
                  n_users: int = 1, case_rows: list[dict] | None = None,
                  model_name: str = "") -> dict[str, dict[str, float]]:
    """Return {segment: {metric: value}} (segment "all" always present).

    If `case_rows` is given, per-case ndcg@10 / recall@20 are appended to it (for paired tests).
    """
    rows = score_cases(recommend, task.target_domain, task.cases, ks, embeddings, popularity, n_users)
    if case_rows is not None:
        case_rows.extend(case_record(task.name, model_name, r) for r in rows)
    return aggregate(rows, len(popularity) if popularity is not None else None)


def benchmark(models: dict[str, RecommendFn], tasks: dict[str, EvalTask], case_rows: list[dict] | None = None,
              **kwargs) -> pd.DataFrame:
    rows = []
    for task_name, task in tasks.items():
        if not task.cases:
            continue
        for model_name, fn in models.items():
            res = evaluate_task(fn, task, case_rows=case_rows, model_name=model_name, **kwargs)
            for seg, ms in res.items():
                rows.append({"task": task_name, "model": model_name, "segment": seg, **ms})
    return pd.DataFrame(rows)
