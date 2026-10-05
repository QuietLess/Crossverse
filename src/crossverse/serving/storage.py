"""Storage adapters: feedback/recommendation log (Postgres or SQLite) and cache (Redis or in-process).

Postgres and Redis are used when DATABASE_URL / REDIS_URL style settings are present (see
infra/docker-compose.yml); local development and tests fall back to SQLite + an in-process TTL cache
with identical behaviour.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS feedback (
        id {pk}, created_at DOUBLE PRECISION NOT NULL, user_id TEXT, item_id TEXT NOT NULL,
        event TEXT NOT NULL, recommendation_id TEXT, model_version TEXT, context TEXT)""",
    """CREATE TABLE IF NOT EXISTS recommendations (
        recommendation_id TEXT PRIMARY KEY, created_at DOUBLE PRECISION NOT NULL, item_id TEXT NOT NULL,
        domain TEXT, rank INTEGER, model_version TEXT, request_mode TEXT, evidence TEXT)""",
]


class Store(Protocol):
    def log_recommendations(self, rows: list[dict[str, Any]]) -> None: ...
    def get_recommendation(self, recommendation_id: str) -> dict[str, Any] | None: ...
    def add_feedback(self, row: dict[str, Any]) -> int: ...
    def feedback_counts(self) -> dict[str, int]: ...
    def ping(self) -> bool: ...


class SQLStore:
    """Works with sqlite3 and psycopg (Postgres) connections."""

    def __init__(self, url: str = "", sqlite_path: Path | None = None):
        self.lock = threading.Lock()
        self.conn: Any  # sqlite3.Connection or psycopg.Connection; both expose cursor()/execute()
        if url.startswith(("postgres://", "postgresql://")):
            import psycopg  # optional dependency: pip install crossverse[storage]

            self.conn = psycopg.connect(url, autocommit=True)
            self.ph, pk = "%s", "BIGSERIAL PRIMARY KEY"
            self.backend = "postgres"
        else:
            path = sqlite_path or Path("crossverse.db")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
            self.ph, pk = "?", "INTEGER PRIMARY KEY AUTOINCREMENT"
            self.backend = "sqlite"
        with self.lock:
            cur = self.conn.cursor()
            for stmt in SCHEMA:
                cur.execute(stmt.format(pk=pk))

    def _exec(self, sql: str, params: tuple = (), fetch: str | None = None):
        sql = sql.replace("?", self.ph)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return cur

    def log_recommendations(self, rows: list[dict[str, Any]]) -> None:
        """One transaction per request (per-row autocommit costs an fsync per item)."""
        if not rows:
            return
        now = time.time()
        sql = ("INSERT INTO recommendations (recommendation_id, created_at, item_id, domain, rank, model_version, "
               "request_mode, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)").replace("?", self.ph)
        params = [(r["recommendation_id"], now, r["item_id"], r["domain"], r["rank"], r["model_version"],
                   r["request_mode"], json.dumps(r["evidence"])) for r in rows]
        with self.lock:
            cur = self.conn.cursor()
            if self.backend == "sqlite":
                cur.execute("BEGIN")
                cur.executemany(sql, params)
                cur.execute("COMMIT")
            else:
                with self.conn.transaction():
                    cur.executemany(sql, params)

    def get_recommendation(self, recommendation_id: str) -> dict[str, Any] | None:
        row = self._exec(
            "SELECT recommendation_id, item_id, domain, rank, model_version, request_mode, evidence "
            "FROM recommendations WHERE recommendation_id = ?", (recommendation_id,), fetch="one",
        )
        if not row:
            return None
        keys = ["recommendation_id", "item_id", "domain", "rank", "model_version", "request_mode", "evidence"]
        out = dict(zip(keys, row, strict=True))
        out["evidence"] = json.loads(out["evidence"])
        return out

    def add_feedback(self, row: dict[str, Any]) -> int:
        cur = self._exec(
            "INSERT INTO feedback (created_at, user_id, item_id, event, recommendation_id, model_version, context) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), row.get("user_id"), row["item_id"], row["event"], row.get("recommendation_id"),
             row.get("model_version"), json.dumps(row.get("context") or {})),
        )
        return int(cur.lastrowid or 0) if self.backend == "sqlite" else 0

    def feedback_counts(self) -> dict[str, int]:
        rows = self._exec("SELECT event, COUNT(*) FROM feedback GROUP BY event", fetch="all")
        return {e: int(n) for e, n in rows}

    def ping(self) -> bool:
        try:
            self._exec("SELECT 1", fetch="one")
            return True
        except Exception:
            return False


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int) -> None: ...
    def ping(self) -> bool: ...


class MemoryCache:
    def __init__(self, max_items: int = 5000):
        self.data: dict[str, tuple[float, Any]] = {}
        self.max_items = max_items
        self.lock = threading.Lock()
        self.backend = "memory"

    def get(self, key: str) -> Any | None:
        with self.lock:
            hit = self.data.get(key)
            if not hit:
                return None
            exp, value = hit
            if exp < time.time():
                self.data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self.lock:
            if len(self.data) >= self.max_items:
                self.data.pop(next(iter(self.data)))
            self.data[key] = (time.time() + ttl, value)

    def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, url: str):
        import redis  # optional dependency

        # RESP2 works on every Redis version; redis-py >= 8 defaults to RESP3 (HELLO 3), which Redis < 6 rejects.
        self.client = redis.Redis.from_url(url, protocol=2)
        self.backend = "redis"

    def get(self, key: str) -> Any | None:
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Any, ttl: int) -> None:
        self.client.setex(key, ttl, json.dumps(value))

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def make_cache(url: str) -> Cache:
    return RedisCache(url) if url else MemoryCache()
