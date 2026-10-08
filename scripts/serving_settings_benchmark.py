"""Benchmark serving settings (taste, compilation filter) on one trained model.

Every setting is evaluated on the same test users, so differences are paired. Besides NDCG@10 it
reports the mean popularity percentile of the recommendations (1 = the most popular item) and the
share of recommendations that carry a specific (non-generic) theme from the user's history: a
descriptive measure of "similar taste", not a quality metric.

    python scripts/serving_settings_benchmark.py v10      # ~15 min on 6 processes
"""

from __future__ import annotations

import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

SETTINGS = {  # name -> CrossVerseEngine.rank() keyword arguments
    "as_before": {"taste": 0.0, "drop_compilations": False},
    "no_compilations": {"taste": 0.0},
    "taste_0.3": {"taste": 0.3},
    "taste_0.5": {"taste": 0.5},
    "taste_1.0": {"taste": 1.0},
}
GENERIC = {"drama", "action", "adventure", "comedy", "family", "thriller", "mystery"}
ENGINE = SPLIT = None


def _init(version: str) -> None:
    global ENGINE, SPLIT
    from crossverse.serving.engine import CrossVerseEngine

    ENGINE = CrossVerseEngine.load(f"artifacts/models/{version}/engine.pkl")
    with open(f"artifacts/models/{version}/split.pkl", "rb") as fh:
        SPLIT = pickle.load(fh)


def _run(job: tuple[str, int, int]) -> list[tuple]:
    from crossverse.evaluation.metrics import ndcg_at_k

    task_name, lo, hi = job
    task = SPLIT.test_tasks[task_name]
    themes = ENGINE.catalog.items["themes"].to_numpy()
    pop_pct = 1 - np.argsort(np.argsort(-ENGINE.popularity)) / len(ENGINE.popularity)
    rows = []
    for case in task.cases[lo:hi]:
        liked = {t for i in case.history.positives().idx for t in themes[i]} - GENERIC
        for name, kw in SETTINGS.items():
            cands, _, order = ENGINE.rank(case.history, task.target_domain, 10, case.exclude, case.preferences, **kw)
            recs = cands.idx[order]
            shares = [bool(liked & (set(themes[r]) - GENERIC)) for r in recs]
            rows.append((task_name, name, ndcg_at_k(list(recs), case.relevant, 10),
                         float(np.mean(pop_pct[recs])) if len(recs) else np.nan,
                         float(np.mean(shares)) if len(recs) and liked else np.nan))
    return rows


CR = chr(13)


class Progress:
    """Progress bar without dependencies. In a terminal it redraws one line; redirected to a file
    (a log you follow with `Get-Content -Wait` / `tail -f`) it prints a new line every 5%."""

    def __init__(self, total: int, unit: str = "users", width: int = 30):
        self.total, self.unit, self.width = total, unit, width
        self.done, self.t0, self.last_step = 0, time.time(), -1
        self.tty = sys.stdout.isatty()

    def update(self, n: int) -> None:
        self.done += n
        frac = self.done / self.total
        step = int(frac * 20)  # 5% steps for log files
        if not self.tty and step == self.last_step and self.done < self.total:
            return
        self.last_step = step
        elapsed = time.time() - self.t0
        left = elapsed / frac * (1 - frac) if frac else 0
        filled = int(self.width * frac)
        line = (f"[{'#' * filled}{'-' * (self.width - filled)}] {frac:6.1%}  {self.done}/{self.total} {self.unit}"
                f"  elapsed {elapsed // 60:.0f}m{elapsed % 60:02.0f}s  ~{left // 60:.0f}m{left % 60:02.0f}s left")
        if self.tty:  # carriage return: redraw the same terminal line
            print(CR + line, end="", flush=True)
        else:
            print(line, flush=True)
        if self.tty and self.done >= self.total:
            print()


def main(version: str, processes: int = 6, chunk: int = 100) -> None:
    print(f"loading model {version} ...", flush=True)
    _init(version)
    jobs = [(t, lo, min(lo + chunk, len(SPLIT.test_tasks[t].cases)))
            for t in SPLIT.test_tasks for lo in range(0, len(SPLIT.test_tasks[t].cases), chunk)]
    total = sum(hi - lo for _, lo, hi in jobs)
    print(f"{total} test users x {len(SETTINGS)} settings on {processes} processes "
          f"(each loads the model first, ~1 min)", flush=True)
    t0 = time.time()
    progress = Progress(total)
    rows = []
    with Pool(processes, initializer=_init, initargs=(version,)) as pool:
        for part in pool.imap_unordered(_run, jobs):
            rows += part
            progress.update(len(part) // len(SETTINGS))
    df = pd.DataFrame(rows, columns=["task", "setting", "ndcg@10", "popularity", "shares_specific_theme"])
    df.to_csv(f"reports/{version}/serving_settings_cases.csv.gz", index=False)
    summary = df.groupby(["task", "setting"]).mean().unstack("setting")
    pd.set_option("display.width", 250)
    for col in ["ndcg@10", "popularity", "shares_specific_theme"]:
        print(f"\n{col}:\n" + summary[col][list(SETTINGS)].round(4).to_string())
    print(f"\n{len(df)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
