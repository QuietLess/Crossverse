"""FastAPI recommendation service (blueprint section 9)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from crossverse import __version__
from crossverse.config import PROJECT_ROOT, Settings, get_settings
from crossverse.features.themes import THEMES
from crossverse.monitoring import metrics as M
from crossverse.monitoring.registry import ModelRegistry
from crossverse.serving import new_releases
from crossverse.serving.engine import CrossVerseEngine, EngineResult
from crossverse.serving.storage import SQLStore, make_cache

log = logging.getLogger(__name__)

Domain = Literal["movie", "game"]
Target = Literal["movie", "game", "film", "tv"]
# Request size limits. The service is public, so every list and string is bounded before title matching
# (fuzzy "did you mean" per unknown title) or storage happens. The web UI sends far less: hand-picked
# titles plus at most 10 imported favourites.
MAX_PROFILE = 100
MAX_EXCLUDE = 1000
MAX_CONTEXT_BYTES = 4096
MAX_SUGGESTED = 5  # unknown titles that get "did you mean" suggestions per request
Title = Annotated[str, Field(max_length=300)]
ShortText = Annotated[str, Field(max_length=100)]
UserId = Annotated[str, Field(max_length=128)]
# Web UI files. The default fits a source checkout; installed (non-editable) packages, as in the container
# images, set CROSSVERSE_UI_DIR because PROJECT_ROOT then points into site-packages.
UI_DIR = Path(os.environ.get("CROSSVERSE_UI_DIR") or PROJECT_ROOT / "apps" / "ui")


# ---------------------------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------------------------
class ProfileItem(BaseModel):
    item: Title = Field(..., description="catalog item_id or a title to resolve", examples=["Blade Runner 2049"])
    rating: float = Field(5.0, ge=1, le=5)
    domain: Domain | None = Field(None, description="disambiguate titles that exist as both a movie and a game "
                                                    "(or prefix the title: 'game: Batman Begins')")


class RecommendRequest(BaseModel):
    liked: list[ProfileItem] = Field(default_factory=list, max_length=MAX_PROFILE)
    disliked: list[Title] = Field(default_factory=list, max_length=MAX_PROFILE, description="item_ids or titles")
    target_domain: Target | None = Field(None, description="game, film (movies only), tv (series only), "
                                                     "movie (films and TV), or null for everything")
    k: int = Field(10, ge=1, le=100)
    preferences: list[ShortText] = Field(default_factory=list, max_length=len(THEMES), description=f"cold-start themes, e.g. {list(THEMES[:5])}")
    free_text: str = Field("", max_length=500, description="cold-start free text, e.g. 'dark sci-fi about AI'")
    exclude: list[Title] = Field(default_factory=list, max_length=MAX_EXCLUDE)
    explain: bool = True
    diversify: bool = True
    user_id: UserId | None = None
    taste: float = Field(0.0, ge=0, le=1, description="0 = what fans with similar histories liked (trained "
                         "ranker), 1 = similar story & setting among well-liked titles; in between, a blend")


class DomainRequest(BaseModel):
    liked: list[ProfileItem] = Field(..., max_length=MAX_PROFILE)
    disliked: list[Title] = Field(default_factory=list, max_length=MAX_PROFILE)
    k: int = Field(10, ge=1, le=100)
    explain: bool = True
    taste: float = Field(0.0, ge=0, le=1, description="0 = what fans with similar histories liked (trained "
                         "ranker), 1 = similar story & setting among well-liked titles; in between, a blend")


class LetterboxdFilm(BaseModel):
    name: str = Field(..., max_length=300)
    year: int | None = Field(None, ge=1870, le=2100)
    rating: float | None = Field(None, ge=0, le=5, description="Letterboxd stars, 0.5-5")


class LetterboxdImport(BaseModel):
    films: list[LetterboxdFilm] = Field(..., max_length=5000, description="rows of ratings.csv from a Letterboxd export")


class FeedbackRequest(BaseModel):
    item_id: Title
    event: Literal["like", "dislike", "click", "save", "skip"]
    recommendation_id: ShortText | None = None
    user_id: UserId | None = None
    context: dict[ShortText, Any] = Field(default_factory=dict, max_length=50)

    @field_validator("context")
    @classmethod
    def _small_context(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v, default=str)) > MAX_CONTEXT_BYTES:
            raise ValueError(f"context is larger than {MAX_CONTEXT_BYTES} bytes as JSON")
        return v


class RecItem(BaseModel):
    recommendation_id: str
    item_id: str
    title: str
    domain: str
    rank: int
    score: float
    sources: list[str]
    themes: list[str]
    year: int | None
    evidence: dict[str, Any]
    image: str | None = Field(None, description="cover image URL (Amazon product image), if known")
    kind: str = Field("", description="movie, tv or game")
    rating: dict[str, Any] | None = Field(None, description="public scores: TMDB audience (0-10) or IGDB critics / "
                                                            "players (0-100), with vote counts")


class RecommendResponse(BaseModel):
    model_version: str
    mode: str
    cold_start: bool
    resolved_profile: list[dict[str, Any]]
    unresolved: list[str]
    suggestions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # input title -> same-titled items in the other domain that were *not* picked
    ambiguous: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    candidate_count: int
    latency_ms: float
    items: list[RecItem]
    # films / series / games of the same franchise as the liked titles (adaptations, tie-ins); not ranked
    same_universe: list[dict[str, Any]] = Field(default_factory=list)
    # titles newer than the review data, closest in story to the liked titles; not ranked
    new_releases: list[dict[str, Any]] = Field(default_factory=list)
    # picked new release -> the catalog titles standing in for it in the ranked list
    stand_ins: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------------------------
class State:
    engine: CrossVerseEngine | None = None
    settings: Settings
    store: SQLStore
    cache: Any
    quality: M.QualityTracker
    pop_pct: np.ndarray
    loaded_at: float = 0.0
    fresh_version: str | None = None  # published new-release index in use (None: the model's own)
    fresh_seen: str | None = None  # last published version examined (loaded or rejected)
    fresh_checked: float = 0.0


state = State()
_fresh_lock = threading.Lock()


def reload_new_releases(force: bool = False) -> None:
    """Swap in a newly published new-release index (CROSSVERSE_NEW_RELEASES_DIR), checking at most every
    new_releases_check_seconds. A missing or unreadable version keeps the index in use."""
    serving = state.settings.serving
    if not serving.new_releases_dir or state.engine is None:
        return
    now = time.monotonic()
    if not force and now - state.fresh_checked < serving.new_releases_check_seconds:
        return
    if not _fresh_lock.acquire(blocking=False):  # another request is already checking
        return
    try:
        state.fresh_checked = now
        root = Path(serving.new_releases_dir)
        version = new_releases.current_version(root)
        if version is None or version == state.fresh_seen:
            return
        loaded, index = new_releases.load_published(root, engine=state.engine)
        state.fresh_seen = version  # read (loaded or rejected); a read error above is retried next check
        if index is not None:
            version = loaded
            state.engine.attach_new_releases(index)
            state.fresh_version = version
            log.info("new releases %s: %d titles", version, len(index))
    except Exception:
        log.exception("new releases not reloaded; keeping %s", state.fresh_version or "the model's own")
    finally:
        _fresh_lock.release()


def load_engine(settings: Settings, engine: CrossVerseEngine | None = None) -> None:
    state.settings = settings
    if engine is None:
        registry = ModelRegistry(settings.serving.artifacts_dir)
        engine = CrossVerseEngine.load(registry.engine_path(settings.serving.model_version))
    state.engine = engine
    state.fresh_version, state.fresh_seen, state.fresh_checked = None, None, 0.0
    reload_new_releases(force=True)
    engine.search("warmup")  # build the title prefix index before the first request
    engine.match_title("warmup")  # and the exact-title table (Letterboxd import)
    # Build the other lazy pieces too (eligibility masks, the sentence model for mood queries) so the
    # first visitor of a fresh container (Cloud Run cold start) doesn't wait seconds for them.
    engine.compilations()
    if len(engine.catalog):
        engine.recommend([(str(engine.catalog.item_ids[0]), 5.0)], k=3, explain=False, taste=0.5)
        engine.recommend([], k=3, explain=False, free_text="warm up")
    order = engine.popularity.argsort().argsort()
    state.pop_pct = order / max(len(order) - 1, 1)
    state.quality = M.QualityTracker(len(engine.catalog))
    M.MODEL_INFO.labels(version=engine.version).set(1)
    state.loaded_at = time.time()


def create_app(settings: Settings | None = None, engine: CrossVerseEngine | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.store = SQLStore(settings.serving.database_url,
                               settings.serving.artifacts_dir / "crossverse.db")
        state.cache = make_cache(settings.serving.redis_url)
        try:
            load_engine(settings, engine)
        except FileNotFoundError as e:  # service still starts; /health reports not ready
            log.error("model not loaded: %s", e)
        yield

    app = FastAPI(title="CrossVerse", version=__version__, lifespan=lifespan,
                  description="Cross-domain movie <-> game recommendations with evidence-based explanations.")

    @app.middleware("http")
    async def metrics_mw(request: Request, call_next):
        t = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            # API routes are labelled by their template (/similar/{domain}/{item_id}). Anything else (the
            # static UI, 404s from bots probing random URLs) shares one label: a raw URL would add a new
            # time series per distinct path, forever.
            route = request.scope.get("route")
            path = getattr(route, "path", "<unmatched>")
            if path != "/metrics":
                M.LATENCY.labels(path).observe(time.perf_counter() - t)
                M.REQUESTS.labels(path, status).inc()

    def engine_or_503() -> CrossVerseEngine:
        if state.engine is None:
            raise HTTPException(503, "model not loaded")
        reload_new_releases()
        return state.engine

    def resolve_profile(eng: CrossVerseEngine, liked: list[ProfileItem], disliked: list[str],
                        source_domain: str | None = None):
        resolved, unresolved, pairs, dis = [], [], [], []
        ambiguous: dict[str, list[dict[str, Any]]] = {}
        for p in liked:
            iid, alternatives = eng.resolve_detail(p.item, p.domain or source_domain)
            if alternatives:
                ambiguous[p.item] = alternatives
            if iid is None:
                unresolved.append(p.item)
                M.UNKNOWN_ITEMS.inc()
                continue
            pairs.append((iid, p.rating))
            resolved.append({**(eng.describe(iid) or {}), "rating": p.rating, "input": p.item})
        for d in disliked:
            iid, alternatives = eng.resolve_detail(d, source_domain)
            if alternatives:
                ambiguous[d] = alternatives
            if iid is None:
                unresolved.append(d)
                M.UNKNOWN_ITEMS.inc()
                continue
            dis.append(iid)
            resolved.append({**(eng.describe(iid) or {}), "rating": 1.0, "input": d})
        return pairs, dis, resolved, unresolved, ambiguous

    def suggestions(eng: CrossVerseEngine, unresolved: list[str], domain: str | None = None
                    ) -> dict[str, list[dict[str, Any]]]:
        """'Did you mean' for the first few distinct unknown titles (each lookup costs real CPU)."""
        return {u: eng.suggest(u, domain) for u in list(dict.fromkeys(unresolved))[:MAX_SUGGESTED]}

    def respond(eng: CrossVerseEngine, mode: str, pairs, dis, resolved, unresolved, target, k, prefs=None,
                free_text="", exclude=None, explain=True, diversify=True,
                ambiguous=None, taste=0.0) -> RecommendResponse:
        key = "rec:" + hashlib.sha1(json.dumps(
            [eng.version, state.fresh_version, mode, pairs, dis, target, k, prefs, free_text, exclude, explain, diversify, taste],
            sort_keys=True, default=str).encode()).hexdigest()
        # The key covers what resolved; what this request typed (unknown titles, inputs) is its own.
        per_request = {"resolved_profile": resolved, "unresolved": unresolved,
                       "suggestions": suggestions(eng, unresolved), "ambiguous": ambiguous or {}}
        cached = state.cache.get(key)
        if cached:
            return RecommendResponse(**{**cached, **per_request})
        # New releases aren't in the trained model: their catalog stand-ins take their place in the ranking.
        ranked_pairs, stand_ins = eng.expand_profile(pairs)
        res: EngineResult = eng.recommend(ranked_pairs, dis, target, k, prefs, free_text, exclude, explain, diversify,
                                          taste)
        items = [RecItem(**r.__dict__) for r in res.items]
        body = RecommendResponse(model_version=res.model_version, mode=mode, cold_start=res.cold_start,
                                 **per_request, candidate_count=res.candidate_count, latency_ms=round(res.latency_ms, 2), items=items,
                                 same_universe=eng.same_universe([i for i, _ in pairs], target, exclude_ids=exclude),
                                 new_releases=eng.new_releases([i for i, _ in pairs], target,
                                                               exclude_ids=[*(exclude or []), *dis]),
                                 stand_ins=stand_ins)
        state.store.log_recommendations([{**i.model_dump(), "model_version": res.model_version, "request_mode": mode}
                                         for i in items])
        for i in items:
            M.RECS_SERVED.labels(mode, i.domain).inc()
        for s, n in res.source_mix.items():
            M.CANDIDATE_SOURCE.labels(s).inc(n)
        if res.cold_start:
            M.COLD_START.inc()
        state.quality.record(mode, [i.item_id for i in items],
                             [float(state.pop_pct[eng.catalog.index[i.item_id]]) for i in items],
                             res.source_mix, res.latency_ms)
        state.cache.set(key, body.model_dump(), settings.serving.cache_ttl_seconds)
        return body

    # -----------------------------------------------------------------------------------------
    @app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
    def recommend(req: RecommendRequest) -> RecommendResponse:
        """Mixed profile (movies + games + dislikes) or cold-start preferences -> ranked items."""
        eng = engine_or_503()
        pairs, dis, resolved, unresolved, ambiguous = resolve_profile(eng, req.liked, req.disliked)
        mode = "cold_start" if not pairs and not dis else "mixed"
        return respond(eng, mode, pairs, dis, resolved, unresolved, req.target_domain, req.k, req.preferences,
                       req.free_text, req.exclude, req.explain, req.diversify, ambiguous, taste=req.taste)

    @app.post("/recommend/movie-to-game", response_model=RecommendResponse, tags=["recommend"])
    def movie_to_game(req: DomainRequest) -> RecommendResponse:
        """Movie history only -> games."""
        eng = engine_or_503()
        pairs, dis, resolved, unresolved, ambiguous = resolve_profile(eng, req.liked, req.disliked, "movie")
        if not pairs:
            raise HTTPException(422, {"error": "none of the liked movies were found",
                                      "suggestions": suggestions(eng, unresolved, "movie")})
        return respond(eng, "movie_to_game", pairs, dis, resolved, unresolved, "game", req.k, explain=req.explain, ambiguous=ambiguous,
                       taste=req.taste)

    @app.post("/recommend/game-to-movie", response_model=RecommendResponse, tags=["recommend"])
    def game_to_movie(req: DomainRequest) -> RecommendResponse:
        """Game history only -> movies / series."""
        eng = engine_or_503()
        pairs, dis, resolved, unresolved, ambiguous = resolve_profile(eng, req.liked, req.disliked, "game")
        if not pairs:
            raise HTTPException(422, {"error": "none of the liked games were found",
                                      "suggestions": suggestions(eng, unresolved, "game")})
        return respond(eng, "game_to_movie", pairs, dis, resolved, unresolved, "movie", req.k, explain=req.explain, ambiguous=ambiguous,
                       taste=req.taste)

    @app.get("/similar/{domain}/{item_id}", tags=["items"])
    def similar(domain: Domain, item_id: str = PathParam(..., max_length=300), k: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
        """Same-domain nearest neighbours plus cross-domain analogues."""
        eng = engine_or_503()
        iid = eng.resolve(item_id, domain)
        if iid is None:
            raise HTTPException(404, f"unknown {domain}: {item_id}")
        return eng.similar(iid, k)

    @app.get("/items/search", tags=["items"])
    def search(q: str = Query(..., min_length=1, max_length=300), domain: Domain | None = None,
               limit: int = Query(10, ge=1, le=50)) -> list[dict[str, Any]]:
        return engine_or_503().search(q, domain, limit)

    @app.get("/items/{item_id}", tags=["items"])
    def item(item_id: str = PathParam(..., max_length=300)) -> dict[str, Any]:
        found = engine_or_503().describe(item_id)
        if found is None:
            raise HTTPException(404, "unknown item")
        return found

    @app.post("/import/letterboxd", tags=["items"])
    def import_letterboxd(req: LetterboxdImport) -> dict[str, Any]:
        """Match a Letterboxd export (ratings.csv rows) to catalog films: same title, year within one.
        Returns the matches with the member's rating, best first, and the names that found nothing."""
        eng = engine_or_503()
        matched, unmatched, seen = [], [], set()
        for f in req.films:
            hit = eng.match_title(f.name, f.year)
            if hit is None:
                unmatched.append(f.name)
            elif hit["item_id"] not in seen:
                seen.add(hit["item_id"])
                matched.append({**hit, "member_rating": f.rating, "input": f.name})
        matched.sort(key=lambda m: -(m["member_rating"] or 0))
        return {"matched": matched, "unmatched": unmatched, "total": len(req.films)}

    @app.get("/themes", tags=["items"])
    def themes() -> list[str]:
        return list(THEMES)

    @app.post("/feedback", tags=["feedback"])
    def feedback(req: FeedbackRequest) -> dict[str, Any]:
        eng = engine_or_503()
        if req.item_id not in eng.catalog.index:
            raise HTTPException(404, "unknown item")
        fid = state.store.add_feedback({**req.model_dump(), "model_version": eng.version})
        M.FEEDBACK.labels(req.event).inc()
        return {"status": "recorded", "id": fid}

    @app.get("/explain/{recommendation_id}", tags=["recommend"])
    def explain(recommendation_id: str) -> dict[str, Any]:
        rec = state.store.get_recommendation(recommendation_id)
        if rec is None:
            raise HTTPException(404, "unknown recommendation_id")
        eng = engine_or_503()
        if rec["item_id"] in eng.catalog.index:
            rec["item"] = eng.item_dict(eng.catalog.index[rec["item_id"]])
        return rec

    @app.get("/health", tags=["ops"])
    def health(response: Response) -> dict[str, Any]:
        eng = state.engine
        ok = eng is not None
        db_ok = state.store.ping()
        cache_ok = state.cache.ping()
        if not (ok and db_ok):
            response.status_code = 503
        return {
            "status": "ok" if ok and db_ok else "degraded",
            "model_version": eng.version if eng else None,
            "model_loaded_at": state.loaded_at or None,
            "catalog_items": len(eng.catalog) if eng else 0,
            "retrievers": list(eng.retrievers) if eng else [],
            "index": {"type": "in-process dense (numpy)", "dim": int(eng.content.embeddings_.shape[1])} if eng else None,
            "database": {"backend": state.store.backend, "ok": db_ok},
            "cache": {"backend": getattr(state.cache, "backend", "?"), "ok": cache_ok},
            "new_releases": {"version": state.fresh_version, "titles": len(eng.fresh) if eng and eng.fresh else 0,
                             "fetched_at": getattr(eng.fresh, "fetched_at", None) if eng else None},
        }

    @app.get("/admin/stats", tags=["ops"])
    def admin_stats() -> dict[str, Any]:
        eng = engine_or_503()
        meta = eng.metadata or {}
        registry = ModelRegistry(settings.serving.artifacts_dir)
        manifest = registry.manifest(eng.version)
        return {
            "model_version": eng.version,
            "production_version": registry.production_version(),
            "test_metrics": manifest.get("test_metrics", {}),
            "train_report": {k: v for k, v in meta.items() if k in ("split", "fit_seconds", "ranker_queries")},
            "live": state.quality.snapshot(),
            "feedback": state.store.feedback_counts(),
        }

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # Web UI (apps/ui: static HTML/JS that calls this API from the same origin).
    # Mounted after every API route, so "/" serves the app (index.html, app.js, ...) and API paths still
    # win; /ui/ keeps older shared links working.
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui-root")

    return app
