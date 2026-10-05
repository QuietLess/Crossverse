"""Data audit (milestone M1): bridge-user size, sparsity, popularity concentration, top cross-domain pairs.

    python pipelines/eda.py    -> reports/eda.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from crossverse.config import PROJECT_ROOT, get_settings


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else 0.0


def main() -> None:
    s = get_settings()
    d = s.data.processed_dir
    inter = pd.read_parquet(d / "interactions.parquet")
    items = pd.read_parquet(d / "items.parquet")
    report = json.loads((d / "data_report.json").read_text()) if (d / "data_report.json").exists() else {}
    pos = inter[inter["rating"] >= s.data.positive_threshold]

    out = ["# Data audit (M1)\n"]
    out.append("## Funnel\n")
    out.append("| metric | value |\n|---|---:|")
    for k in ("raw_interactions", "users_in_both_domains", "bridge_users_selected", "single_domain_users_selected",
              "products_with_metadata", "canonical_items", "products_merged_away", "final_interactions",
              "final_users", "final_bridge_users"):
        if k in report:
            out.append(f"| {k} | {report[k]:,} |")

    out.append("\n## Shape\n")
    per_user = inter.groupby("user_id").size()
    dom = pos.groupby(["user_id", "domain"]).size().unstack(fill_value=0)
    out.append(f"* interactions per user: median {per_user.median():.0f}, p90 {per_user.quantile(.9):.0f}, max {per_user.max()}")
    out.append("* rating distribution: " + ", ".join(f"{int(r)}★ {p:.1%}" for r, p in inter['rating'].value_counts(normalize=True).sort_index().items()))
    density = len(inter) / (inter["user_id"].nunique() * len(items))
    out.append(f"* matrix density: {density:.5%}")
    for dname in ("movie", "game"):
        p = pos[pos["domain"] == dname].groupby("item_id").size()
        top1 = p.sort_values(ascending=False)
        share = top1.head(max(1, len(top1) // 100)).sum() / top1.sum()
        out.append(f"* {dname}s: {len(p):,} items with positives, Gini {gini(p.to_numpy()):.3f}, top 1% of items = {share:.1%} of positives")
    both = dom[(dom.get("movie", 0) > 0) & (dom.get("game", 0) > 0)]
    out.append(f"* bridge users (≥1 positive each domain): {len(both):,}; median positives movie {both['movie'].median():.0f} / game {both['game'].median():.0f}")
    out.append(f"* items with ≥1 theme: {(items['themes'].map(len) > 0).mean():.1%}; canonical items merging >1 product: {(items.get('n_products', pd.Series([1])) > 1).mean():.1%}")

    # Top movie<->game co-liked pairs among bridge users (lift-ranked, min support).
    users = both.index
    bp = pos[pos["user_id"].isin(users)]
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(items["item_id"])}
    X = sp.csr_matrix((np.ones(len(bp)), (bp["user_id"].map(uidx), bp["item_id"].map(iidx))), shape=(len(users), len(items)))
    m = (items["domain"] == "movie").to_numpy()
    Xm, Xg = X[:, m], X[:, ~m]
    C = (Xm.T @ Xg).tocoo()
    pm, pg = np.asarray(Xm.sum(0)).ravel(), np.asarray(Xg.sum(0)).ravel()
    n = len(users)
    lift = C.data * n / (pm[C.row] * pg[C.col])
    df = pd.DataFrame({"movie": items["title"].to_numpy()[m][C.row], "game": items["title"].to_numpy()[~m][C.col],
                       "co_likes": C.data.astype(int), "lift": lift})
    top = df[df["co_likes"] >= 15].sort_values("lift", ascending=False).head(25)
    out.append("\n## Strongest movie ↔ game affinities (bridge users, ≥15 co-likes, ranked by lift)\n")
    out.append("| movie / series | game | co-likes | lift |\n|---|---|---:|---:|")
    for _, r in top.iterrows():
        out.append(f"| {r['movie'][:60]} | {r['game'][:60]} | {r['co_likes']} | {r['lift']:.1f} |")
    out.append("\nLift = P(both) / (P(movie)·P(game)): how much more often the pair co-occurs than chance. These "
               "pairs are the raw cross-domain signal that the co-preference and two-tower models learn from.")

    path = PROJECT_ROOT / "reports" / "eda.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
