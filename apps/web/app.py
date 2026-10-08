"""CrossVerse demo UI (Streamlit). Talks to the FastAPI service; falls back to an in-process engine.

    streamlit run apps/web/app.py            # CROSSVERSE_API_URL defaults to http://localhost:8000
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from crossverse.evaluation import judgments as J

API = os.environ.get("CROSSVERSE_API_URL", "http://localhost:8000")
DOMAIN_ICON = {"movie": "🎬", "game": "🎮"}
TASTE_DEFAULT = 0.8  # best on the judged set with v11 (docs/evaluation.md#judged-evaluation)
POOL_TASTES, POOL_K = (0.0, 0.5, 1.0), 5  # what the Rate tab asks you to judge, per query

st.set_page_config(page_title="CrossVerse", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------------------------
# Backend: HTTP API, or the engine in-process when the API is not running.
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def _local_client():
    from fastapi.testclient import TestClient

    from crossverse.serving.api import create_app

    client = TestClient(create_app())
    client.__enter__()
    return client


def _api_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource
def backend_mode() -> str:
    return "api" if _api_up() else "local"


def call(method: str, path: str, **kw) -> Any:
    if backend_mode() == "api":
        r = requests.request(method, f"{API}{path}", timeout=30, **kw)
    else:
        r = _local_client().request(method, path, **kw)
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.json().get('detail', r.text) if r.headers.get('content-type','').startswith('application/json') else r.text}")
        return None
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def search(q: str, domain: str | None) -> list[dict]:
    if not q or len(q) < 2:
        return []
    return call("GET", "/items/search", params={"q": q, "domain": domain, "limit": 8}) or []


def picker(label: str, domain: str | None, key: str, default: list[str] | None = None) -> list[dict]:
    """Multi-select fed by catalog search (type to add titles)."""
    if key not in st.session_state:
        st.session_state[key] = []
        for d in default or []:
            hits = search(d, domain)
            if hits:
                st.session_state[key].append(hits[0])
    q = st.text_input(f"{label} — search", key=f"{key}_q", placeholder="type a title…")
    hits = search(q, domain)
    if hits:
        options = {f"{DOMAIN_ICON[h['domain']]} {h['title']}" + (f" ({h['year']})" if h.get("year") else ""): h for h in hits}
        choice = st.selectbox("matches", list(options), key=f"{key}_sel", label_visibility="collapsed")
        if st.button("Add", key=f"{key}_add") and options[choice]["item_id"] not in {x["item_id"] for x in st.session_state[key]}:
            st.session_state[key].append(options[choice])
    chosen = st.session_state[key]
    if chosen:
        cols = st.columns(min(len(chosen), 4))
        for n, item in enumerate(list(chosen)):
            with cols[n % len(cols)]:
                if st.button(f"✕ {DOMAIN_ICON[item['domain']]} {item['title'][:40]}", key=f"{key}_rm_{item['item_id']}"):
                    st.session_state[key] = [x for x in chosen if x["item_id"] != item["item_id"]]
                    st.rerun()
    return st.session_state[key]


def cover(item: dict, width: int) -> None:
    """Cover image, or the domain icon when the catalog has none (models built before images existed)."""
    if item.get("image"):
        st.image(item["image"], width=width)
    else:
        st.markdown(f"<div style='font-size:{width // 2}px;text-align:center'>{DOMAIN_ICON[item['domain']]}</div>",
                    unsafe_allow_html=True)


def render_results(body: dict | None) -> None:
    if not body:
        return
    st.caption(
        f"model **{body['model_version']}** · mode `{body['mode']}` · {body['candidate_count']} candidates · "
        f"{body['latency_ms']:.0f} ms" + (" · cold start" if body.get("cold_start") else "")
    )
    if body.get("unresolved"):
        st.warning("Not found in catalog: " + ", ".join(body["unresolved"]))
    picked = {p["input"]: p for p in body.get("resolved_profile", [])}
    for title, alts in body.get("ambiguous", {}).items():
        used = picked.get(title)
        if used:
            other = ", ".join(f"{DOMAIN_ICON[a['domain']]} {a['title']}" for a in alts)
            st.info(f"“{title}” was read as {DOMAIN_ICON[used['domain']]} {used['domain']} — "
                    f"also exists as {other}. Prefix it with `{alts[0]['domain']}:` to use that instead.")
    for item in body["items"]:
        ev = item["evidence"]
        conf = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(ev.get("reason_confidence"), "")
        with st.container(border=True):
            c0, c1, c2 = st.columns([1, 5, 1])
            with c0:
                cover(item, 90)
            with c1:
                year = f" ({item['year']})" if item.get("year") and str(item["year"]) not in item["title"] else ""
                st.markdown(f"**{item['rank']}. {DOMAIN_ICON[item['domain']]} {item['title']}**{year}")
                st.caption(" · ".join(item["themes"][:5]))
                st.write(ev.get("summary", ""))
            with c2:
                st.write(f"{conf} {ev.get('reason_confidence', '')}")
                b1, b2 = st.columns(2)
                if b1.button("👍", key=f"l_{item['recommendation_id']}"):
                    call("POST", "/feedback", json={"item_id": item["item_id"], "event": "like",
                                                    "recommendation_id": item["recommendation_id"]})
                    st.toast("Thanks — feedback recorded")
                if b2.button("👎", key=f"d_{item['recommendation_id']}"):
                    call("POST", "/feedback", json={"item_id": item["item_id"], "event": "dislike",
                                                    "recommendation_id": item["recommendation_id"]})
                    st.toast("Noted")
            with st.expander("Why? (evidence)"):
                st.json({k: v for k, v in ev.items() if k != "summary"})


# ---------------------------------------------------------------------------------------------
st.title("CrossVerse")
st.markdown("*If you liked this movie, what game should you play next — and vice versa?*")
st.sidebar.markdown(f"Backend: **{backend_mode()}**" + (f" (`{API}`)" if backend_mode() == "api" else " (in-process engine)"))
k = st.sidebar.slider("How many recommendations", 5, 30, 10)
taste = st.sidebar.slider(
    "Recommendation style", 0.0, 1.0, TASTE_DEFAULT, 0.1,
    help="0 = what fans with a similar history liked (often popular titles). "
         "1 = most similar story, setting and themes among well-liked titles. In between: a blend.",
)
st.sidebar.caption("◀ popular with similar fans · similar story & setting ▶")

tabs = st.tabs(["🎬 → 🎮 Movie to Game", "🎮 → 🎬 Game to Movie", "🔀 Mixed profile", "🔎 Similar items",
                "✨ Cold start", "📊 Admin", "⭐ Rate"])

with tabs[0]:
    liked = picker("Movies / series you love", "movie", "m2g", ["Blade Runner 2049", "Ex Machina"])
    if st.button("Recommend games", type="primary", disabled=not liked):
        render_results(call("POST", "/recommend/movie-to-game",
                            json={"liked": [{"item": x["item_id"]} for x in liked], "k": k, "taste": taste}))

with tabs[1]:
    liked = picker("Games you love", "game", "g2m", ["Witcher 3: Wild Hunt", "Baldur's Gate"])
    if st.button("Recommend movies & series", type="primary", disabled=not liked):
        render_results(call("POST", "/recommend/game-to-movie",
                            json={"liked": [{"item": x["item_id"]} for x in liked], "k": k, "taste": taste}))

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        liked = picker("Likes (movies and games)", None, "mix_like")
    with c2:
        disliked = picker("Dislikes", None, "mix_dislike")
    target = st.radio("Recommend", ["both", "movie", "game"], horizontal=True)
    if st.button("Build my profile", type="primary", disabled=not liked):
        render_results(call("POST", "/recommend", json={
            "liked": [{"item": x["item_id"]} for x in liked], "disliked": [x["item_id"] for x in disliked],
            "target_domain": None if target == "both" else target, "k": k, "taste": taste}))

with tabs[3]:
    dom = st.radio("Domain", ["movie", "game"], horizontal=True, key="sim_dom")
    item = picker("Item", dom, f"sim_{dom}")
    if item and st.button("Find similar", type="primary"):
        res = call("GET", f"/similar/{dom}/{item[-1]['item_id']}", params={"k": k})
        if res:
            c1, c2 = st.columns(2)
            for col, key, label in ((c1, "same_domain", "Same domain"), (c2, "cross_domain", "Cross-domain analogues")):
                with col:
                    st.subheader(label)
                    for n, r in enumerate(res[key], 1):
                        img, text = st.columns([1, 5])
                        with img:
                            cover(r, 60)
                        with text:
                            st.markdown(f"{n}. {DOMAIN_ICON[r['domain']]} **{r['title']}** — {', '.join(r['themes'][:3])}")
                            st.caption(r["evidence"]["summary"])

with tabs[4]:
    themes = call("GET", "/themes") or []
    prefs = st.multiselect("Pick themes you enjoy", themes, default=[t for t in ("cyberpunk", "philosophical") if t in themes])
    text = st.text_input("…or describe a mood", placeholder="slow-burn sci-fi about memory and identity")
    target = st.radio("Recommend", ["both", "movie", "game"], horizontal=True, key="cs_t")
    if st.button("Get starter picks", type="primary", disabled=not (prefs or text)):
        render_results(call("POST", "/recommend", json={
            "preferences": prefs, "free_text": text, "k": k, "target_domain": None if target == "both" else target}))

with tabs[5]:
    stats = call("GET", "/admin/stats")
    health = call("GET", "/health")
    if stats and health:
        c = st.columns(4)
        c[0].metric("Model version", stats["model_version"])
        c[1].metric("Catalog items", f"{health['catalog_items']:,}")
        live = stats["live"]
        c[2].metric("Catalog coverage (live)", f"{live['catalog_coverage']:.2%}")
        c[3].metric("p95 latency", f"{live['latency_ms']['p95'] or 0:.0f} ms")
        st.subheader("Offline test metrics (cross-domain)")
        tm = stats.get("test_metrics", {})
        st.dataframe({k: [round(v, 4)] for k, v in tm.items() if "ndcg@10" in k or "recall@20" in k})
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Candidate-source mix (served)")
            st.bar_chart(live["candidate_source_mix"])
        with c2:
            st.subheader("Requests by mode")
            st.bar_chart(live["requests_by_mode"])
        st.caption(f"Mean popularity percentile of served items: {live['mean_popularity_percentile']} "
                   "(drift indicator: rising = more popularity-biased)")
        st.json({"health": health, "feedback": stats["feedback"]})


@st.cache_data(show_spinner="Collecting suggestions to rate…")
def rating_pool(model_version: str) -> dict[str, list[dict]]:
    """Top POOL_K for every query at each POOL_TASTES setting, merged and in blind order."""
    pool = {}
    for q in J.load_queries():
        path = "/recommend/movie-to-game" if q.seed_domain == "movie" else "/recommend/game-to-movie"
        items: dict[str, dict] = {}
        for t in POOL_TASTES:
            body = call("POST", path, json={"liked": [{"item": q.seed_item_id}], "k": POOL_K, "taste": t,
                                            "explain": False}) or {}
            for it in body.get("items", []):
                items.setdefault(it["item_id"], it)
        pool[q.id] = [items[i] for i in J.blind_order(q.id, list(items))]
    return pool


@st.cache_data(show_spinner=False)
def seed_item(title: str, domain: str) -> dict:
    hits = call("GET", "/items/search", params={"q": title, "domain": domain, "limit": 1}) or []
    return hits[0] if hits else {"title": title, "domain": domain}


@st.fragment
def rate_panel() -> None:
    """Runs as a fragment: a rating click refreshes only this panel, not all tabs (each full rerun calls
    the API for every tab, which made saving a rating take seconds)."""
    health = call("GET", "/health") or {}
    queries = {q.id: q for q in J.load_queries()}
    pool = rating_pool(health.get("model_version", "?"))
    mine = J.load_ratings(rater=J.HUMAN)
    done = set(zip(mine["query_id"], mine["item_id"], strict=True))
    todo = [(qid, it) for qid, items in pool.items() for it in items if (qid, it["item_id"]) not in done]
    total = sum(len(v) for v in pool.values())
    skipped = int((mine["rating"] < 0).sum())
    st.progress((total - len(todo)) / total if total else 1.0,
                text=f"{total - len(todo)} / {total} answered ({skipped} skipped as unknown)")
    agree = J.agreement(J.load_ratings())
    if agree.get("n"):
        st.caption(f"Agreement with Claude's ratings on {agree['n']} items you both rated: "
                   f"{agree['exact']:.0%} identical, {agree['within_one']:.0%} within one step.")
    if not todo:
        st.success("Nothing left to rate for this model. Thank you! Run `python scripts/judged_eval.py`.")
        return
    qid, item = todo[0]
    q = queries[qid]
    left, right = st.columns(2)
    with left:
        st.caption("Someone who liked")
        seed = seed_item(q.seed_title, q.seed_domain)
        cover(seed, 160)
        st.markdown(f"### {DOMAIN_ICON[q.seed_domain]} {q.seed_title}")
        st.caption(" · ".join(seed.get("themes", [])[:5]))
    with right:
        st.caption(f"…would they enjoy this {item['domain']}?")
        cover(item, 160)
        year = f" ({item['year']})" if item.get("year") and str(item["year"]) not in item["title"] else ""
        st.markdown(f"### {DOMAIN_ICON[item['domain']]} {item['title']}{year}")
        st.caption(" · ".join(item["themes"][:5]))
    choices = ((2, "👍 Good fit"), (1, "🤷 Okay"), (0, "👎 Bad fit"), (-1, "🙈 Don't know it"))
    for col, (score, label) in zip(st.columns(4), choices, strict=True):
        if col.button(label, key=f"rate_{score}_{qid}_{item['item_id']}", use_container_width=True):
            J.add_rating(qid, item, score)
            st.rerun(scope="fragment")


with tabs[6]:
    st.markdown("Would someone who liked the title on the left enjoy the one on the right? Only rate what you "
                "know: **🙈 Don't know it** skips a title for good. Suggestions from different settings are mixed "
                "and shuffled, so you can't tell which setting proposed what. Answers are saved to "
                "`docs/judgments/ratings.csv` immediately: stop any time.")
    rate_panel()
