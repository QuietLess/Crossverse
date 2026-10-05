"""Prometheus metrics for the recommendation service + in-process quality counters for /admin."""

from __future__ import annotations

import threading
from collections import Counter, deque

from prometheus_client import CollectorRegistry, Gauge, Histogram
from prometheus_client import Counter as PCounter

REGISTRY = CollectorRegistry()

REQUESTS = PCounter("crossverse_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram(
    "crossverse_request_latency_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5), registry=REGISTRY,
)
RECS_SERVED = PCounter("crossverse_recommendations_total", "Items recommended", ["mode", "domain"], registry=REGISTRY)
CANDIDATE_SOURCE = PCounter("crossverse_candidate_source_total", "Retriever that nominated a served item",
                            ["source"], registry=REGISTRY)
COLD_START = PCounter("crossverse_cold_start_requests_total", "Requests without any history", registry=REGISTRY)
FEEDBACK = PCounter("crossverse_feedback_total", "Feedback events", ["event"], registry=REGISTRY)
CATALOG_COVERAGE = Gauge("crossverse_catalog_coverage_ratio", "Distinct items served / catalog size (rolling)",
                         registry=REGISTRY)
MODEL_INFO = Gauge("crossverse_model_info", "Loaded model version", ["version"], registry=REGISTRY)
UNKNOWN_ITEMS = PCounter("crossverse_unknown_items_total", "Request items not found in catalog", registry=REGISTRY)


class QualityTracker:
    """Rolling window of served items for coverage / popularity-drift diagnostics."""

    def __init__(self, catalog_size: int, window: int = 20000):
        self.catalog_size = max(catalog_size, 1)
        self.served: deque[str] = deque(maxlen=window)
        self.pop_pct: deque[float] = deque(maxlen=window)
        self.sources: Counter[str] = Counter()
        self.modes: Counter[str] = Counter()
        self.latencies: deque[float] = deque(maxlen=2000)
        self.lock = threading.Lock()

    def record(self, mode: str, items: list[str], pop_pct: list[float], sources: dict[str, int],
               latency_ms: float) -> None:
        with self.lock:
            self.served.extend(items)
            self.pop_pct.extend(pop_pct)
            self.sources.update(sources)
            self.modes[mode] += 1
            self.latencies.append(latency_ms)
            CATALOG_COVERAGE.set(len(set(self.served)) / self.catalog_size)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            lat = sorted(self.latencies)

            def pct(p: float) -> float | None:
                return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else None

            total_src = sum(self.sources.values()) or 1
            return {
                "requests_by_mode": dict(self.modes),
                "served_items_window": len(self.served),
                "catalog_coverage": round(len(set(self.served)) / self.catalog_size, 4),
                "mean_popularity_percentile": round(sum(self.pop_pct) / len(self.pop_pct), 4) if self.pop_pct else None,
                "candidate_source_mix": {k: round(v / total_src, 4) for k, v in self.sources.most_common()},
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
            }
