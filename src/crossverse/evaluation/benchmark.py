"""Benchmark every baseline and the full pipeline on the same tasks; qualitative error analysis."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from crossverse.evaluation.protocol import EvalCase, EvalTask
from crossverse.evaluation.runner import aggregate, benchmark, case_record, score_cases
from crossverse.retrieval.base import top_k
from crossverse.retrieval.content import ContentRetriever
from crossverse.serving.engine import CrossVerseEngine

CROSS_TASKS = ("movie_to_game", "game_to_movie")


def model_suite(engine: CrossVerseEngine) -> dict[str, Callable[[str], Callable | None]]:
    """name -> factory(task_name) returning a recommend fn, or None if the model does not apply."""
    cat = engine.catalog
    R = engine.retrievers

    def from_retriever(name: str):
        r = R[name]

        def fn(case: EvalCase, target: str | None, k: int) -> np.ndarray:
            return top_k(r.score(case.history), k, cat.mask(target), case.exclude)

        return fn

    def content_fn(case: EvalCase, target: str | None, k: int) -> np.ndarray:
        c = R["content"]
        assert isinstance(c, ContentRetriever)
        s = c.score(case.history) if len(case.history) else c.score_query("", case.preferences)
        return top_k(s, k, cat.mask(target), case.exclude)

    def als_domain(case: EvalCase, target: str | None, k: int) -> np.ndarray:
        return top_k(R[f"als_{target}"].score(case.history), k, cat.mask(target), case.exclude)

    def fusion(case: EvalCase, target: str | None, k: int) -> np.ndarray:
        return engine.recommend_case(case, target, k, use_ranker=False, diversify_results=False)

    def full(case: EvalCase, target: str | None, k: int) -> np.ndarray:
        return engine.recommend_case(case, target, k, diversify_results=False)

    def full_div(case: EvalCase, target: str | None, k: int) -> np.ndarray:
        return engine.recommend_case(case, target, k)

    behavioural = lambda name: (lambda task: None if task == "cold_start" else from_retriever(name))  # noqa: E731
    return {
        "popularity": lambda task: from_retriever("popularity"),
        "item_knn": behavioural("item_knn"),
        "als_single_domain": lambda task: als_domain if task in ("within_movie", "within_game") else None,
        "als_joint": behavioural("als"),
        "content": lambda task: content_fn,
        "copref": lambda task: from_retriever("copref") if task in CROSS_TASKS else None,
        "two_tower": behavioural("two_tower"),
        "rrf_fusion": lambda task: fusion,
        "crossverse_ranker": lambda task: full,
        "crossverse_ranker+diversity": lambda task: full_div,
    }


def run_benchmark(engine: CrossVerseEngine, tasks: dict[str, EvalTask], models: list[str] | None = None,
                  ks: tuple[int, ...] = (10, 20, 50), n_users: int = 1,
                  case_rows: list[dict] | None = None) -> pd.DataFrame:
    suite = model_suite(engine)
    frames = []
    for task_name, task in tasks.items():
        fns = {}
        for name, factory in suite.items():
            if models and name not in models:
                continue
            fn = factory(task_name)
            if fn is not None:
                fns[name] = fn
        frames.append(
            benchmark(fns, {task_name: task}, case_rows=case_rows, ks=ks, embeddings=engine.content.embeddings_,
                      popularity=engine.popularity, n_users=n_users)
        )
    return pd.concat(frames, ignore_index=True)


# --- parallel benchmark ------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(engine_path: str, split_path: str) -> None:
    import pickle

    from threadpoolctl import threadpool_limits

    threadpool_limits(1)  # one BLAS thread per process; parallelism comes from processes
    _WORKER["engine"] = CrossVerseEngine.load(Path(engine_path))
    with open(split_path, "rb") as fh:
        _WORKER["tasks"] = pickle.load(fh).test_tasks
    _WORKER["suite"] = model_suite(_WORKER["engine"])


def _score_chunk(job: tuple[str, str, int, int, tuple[int, ...], int]) -> tuple[str, str, list[dict]]:
    task_name, model, start, stop, ks, n_users = job
    engine, task = _WORKER["engine"], _WORKER["tasks"][task_name]
    fn = _WORKER["suite"][model](task_name)
    rows = score_cases(fn, task.target_domain, task.cases[start:stop], ks, engine.content.embeddings_,
                       engine.popularity, n_users)
    return task_name, model, rows


def run_benchmark_parallel(engine: CrossVerseEngine, engine_path: Path, split_path: Path,
                           tasks: dict[str, EvalTask], models: list[str] | None = None,
                           ks: tuple[int, ...] = (10, 20, 50), n_users: int = 1,
                           case_rows: list[dict] | None = None, n_jobs: int | None = None,
                           chunk: int = 250) -> pd.DataFrame:
    """Same output as `run_benchmark`, with cases fanned out over processes (spawn-safe).

    `tasks` must be the test tasks stored in `split_path` (workers re-load them by name).
    """
    from concurrent.futures import ProcessPoolExecutor

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 2)
    suite = model_suite(engine)
    jobs = []
    for task_name, task in tasks.items():
        for model, factory in suite.items():
            if (models and model not in models) or factory(task_name) is None:
                continue
            for start in range(0, len(task.cases), chunk):
                jobs.append((task_name, model, start, min(start + chunk, len(task.cases)), ks, n_users))
    collected: dict[tuple[str, str], list[dict]] = {}
    with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(str(engine_path), str(split_path))) as ex:
        for task_name, model, rows in ex.map(_score_chunk, jobs):
            collected.setdefault((task_name, model), []).extend(rows)
    out = []
    for (task_name, model), rows in collected.items():  # insertion order follows `jobs`
        if case_rows is not None:
            case_rows.extend(case_record(task_name, model, r) for r in rows)
        for seg, ms in aggregate(rows, len(engine.catalog)).items():
            out.append({"task": task_name, "model": model, "segment": seg, **ms})
    return pd.DataFrame(out)


def paired_bootstrap(cases: pd.DataFrame, model: str, baseline: str, metric: str = "ndcg@10",
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per task: mean lift of `model` over `baseline` on the *same* users, with a 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for task, g in cases.groupby("task", sort=False):
        piv = g.pivot_table(index="user_id", columns="model", values=metric)
        if model not in piv or baseline not in piv:
            continue
        a, b = piv[model].to_numpy(), piv[baseline].to_numpy()
        diff = a - b
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"task": task, "model": model, "baseline": baseline, "n": len(diff),
                     f"{metric} model": a.mean(), f"{metric} baseline": b.mean(),
                     "lift_abs": diff.mean(), "lift_rel": diff.mean() / b.mean() if b.mean() else np.nan,
                     "ci95_low": lo, "ci95_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def significance_markdown(table: pd.DataFrame, metric: str = "ndcg@10") -> str:
    lines = [f"| task | vs baseline | n | {metric} model | {metric} baseline | lift | 95% CI (abs) | significant |",
             "|---|---|---:|---:|---:|---:|---|:---:|"]
    for _, r in table.iterrows():
        lift = f"{r['lift_rel']:+.1%}" if pd.notna(r["lift_rel"]) else "–"
        lines.append(f"| {r['task']} | {r['baseline']} | {r['n']} | {r[f'{metric} model']:.4f} | "
                     f"{r[f'{metric} baseline']:.4f} | {lift} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | "
                     f"{'yes' if r['significant'] else 'no'} |")
    return "\n".join(lines)


def headline_metrics(df: pd.DataFrame, model: str = "crossverse_ranker+diversity") -> dict[str, float]:
    allseg = df[(df["segment"] == "all") & (df["model"] == model)].set_index("task")
    out = {}
    for task, row in allseg.iterrows():
        for m in ("ndcg@10", "recall@10", "recall@20", "hit@10", "coverage@10", "diversity@10"):
            if m in row and pd.notna(row[m]):
                out[f"{task}_{m}"] = float(row[m])
    cross = [t for t in CROSS_TASKS if t in allseg.index]
    if cross:
        out["cross_domain_ndcg@10"] = float(np.mean([allseg.loc[t, "ndcg@10"] for t in cross]))
        out["cross_domain_recall@20"] = float(np.mean([allseg.loc[t, "recall@20"] for t in cross]))
    return out


def markdown_table(df: pd.DataFrame, segment: str = "all",
                   metrics: tuple[str, ...] = ("ndcg@10", "recall@10", "recall@20", "hit@10", "coverage@10",
                                               "diversity@10", "pop_pct@10")) -> str:
    sub = df[df["segment"] == segment]
    lines = []
    for task, g in sub.groupby("task", sort=False):
        best = {m: g[m].max() for m in metrics if m in g}
        lines.append(f"\n### {task} (n={int(g['n'].iloc[0])})\n")
        lines.append("| model | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---:|" * len(metrics))
        for _, r in g.iterrows():
            cells = []
            for m in metrics:
                v = r.get(m)
                if pd.isna(v):
                    cells.append("–")
                else:
                    s = f"{v:.4f}"
                    cells.append(f"**{s}**" if m not in ("pop_pct@10",) and np.isclose(v, best[m]) else s)
            lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def segment_table(df: pd.DataFrame, model: str = "crossverse_ranker+diversity", metric: str = "ndcg@10") -> str:
    sub = df[df["model"] == model].pivot_table(index="task", columns="segment", values=metric)
    cnt = df[df["model"] == model].pivot_table(index="task", columns="segment", values="n")
    cols = [c for c in ("all", "heavy", "medium", "sparse", "bridge") if c in sub.columns]
    lines = ["| task | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for task in sub.index:
        cells = [f"{sub.loc[task, c]:.4f} (n={int(cnt.loc[task, c])})" if pd.notna(sub.loc[task, c]) else "–" for c in cols]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def error_analysis(engine: CrossVerseEngine, task: EvalTask, n: int = 50, k: int = 10, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    cases = [task.cases[i] for i in rng.permutation(len(task.cases))[:n]]
    titles = engine.catalog.items["title"].to_numpy()
    lines = [f"## {task.name}: {len(cases)} sampled users\n"]
    n_hit = 0
    for c in cases:
        recs = engine.recommend_case(c, task.target_domain, k)
        hits = [int(i) for i in recs if int(i) in c.relevant]
        n_hit += bool(hits)
        liked = [titles[i] for i in c.history.positives().idx[-6:]]
        disliked = [titles[i] for i in c.history.idx[c.history.weight < 0][-3:]]
        lines.append(f"**{c.user_id}** — liked: {', '.join(liked)}" + (f"; disliked: {', '.join(disliked)}" if disliked else ""))
        rec_str = ", ".join(("✅ " if int(i) in c.relevant else "") + titles[i] for i in recs)
        lines.append(f"- recommended: {rec_str}")
        lines.append(f"- actually liked (held out): {', '.join(titles[i] for i in list(c.relevant)[:6])}\n")
    lines.insert(1, f"Hit@{k} on this sample: {n_hit}/{len(cases)}\n")
    return "\n".join(lines)
