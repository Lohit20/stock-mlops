"""
Prediction cache for the FastAPI serving layer.

Priority:
  1. Redis  (REDIS_URL env var — e.g. redis://localhost:6379/0)
  2. In-process dict with TTL (single-worker fallback, no extra dependency)

Cache key schema:
  stock:pred:{symbol}:{model_type}:{YYYY-MM-DD}

TTL: CACHE_TTL_SECONDS env var (default 3600 — 1 hour).
     Predictions from the same day are reused; keys auto-expire at midnight
     if you set TTL ≤ seconds remaining in the trading day.
"""

import json
import os
import time
from loguru import logger

REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL        = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


# ── In-process fallback ───────────────────────────────────────────────────────

class _DictCache:
    """Thread-unsafe in-memory TTL cache (single-worker dev fallback)."""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # key → (value, expiry)
        self._hits   = 0
        self._misses = 0

    def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expiry = entry
        if expiry and time.time() > expiry:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value.encode() if isinstance(value, str) else value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (value, time.time() + ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def keys(self, pattern: str = "*") -> list:
        import fnmatch
        pat = pattern.replace("*", "**")
        return [k for k in self._store if fnmatch.fnmatch(k, pat)]

    def flushdb(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend":    "dict",
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self._hits / total, 4) if total else 0.0,
            "size":       len(self._store),
        }


# ── Redis client ──────────────────────────────────────────────────────────────

_client_singleton = None


def get_client():
    """
    Return a cache client (Redis or in-process dict).
    Result is cached at module level — same instance across requests.
    """
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    try:
        import redis as redis_lib
        r = redis_lib.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
        r.ping()
        logger.info(f"Cache: connected to Redis at {REDIS_URL}")
        _client_singleton = r
    except Exception as exc:
        logger.warning(f"Redis unavailable ({exc}) — using in-process dict cache")
        _client_singleton = _DictCache()

    return _client_singleton


def _reset_client():
    """Force re-initialisation (used in tests)."""
    global _client_singleton
    _client_singleton = None


# ── Public API ────────────────────────────────────────────────────────────────

def cache_key(symbol: str, model_type: str) -> str:
    from datetime import date
    return f"stock:pred:{symbol.upper()}:{model_type.lower()}:{date.today()}"


def get_cached(key: str) -> dict | None:
    """Return cached prediction dict, or None on miss / decode error."""
    try:
        raw = get_client().get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug(f"Cache get error for {key}: {exc}")
        return None


def set_cached(key: str, value: dict, ttl: int = CACHE_TTL) -> None:
    """Serialise and store a prediction dict."""
    try:
        get_client().setex(key, ttl, json.dumps(value))
    except Exception as exc:
        logger.debug(f"Cache set error for {key}: {exc}")


def invalidate(symbol: str | None = None, model_type: str | None = None) -> int:
    """
    Invalidate cached predictions matching symbol and/or model_type.
    If both are None, flush the entire cache.
    Returns the number of keys deleted.
    """
    client = get_client()
    if symbol is None and model_type is None:
        if hasattr(client, "flushdb"):
            client.flushdb()
            return -1   # unknown count
        return 0

    parts = ["stock:pred"]
    parts.append(symbol.upper() if symbol else "*")
    parts.append(model_type.lower() if model_type else "*")
    parts.append("*")
    pattern = ":".join(parts)

    try:
        keys = list(client.keys(pattern))
        if keys:
            if hasattr(client, "delete"):
                client.delete(*keys)
            return len(keys)
    except Exception as exc:
        logger.debug(f"Cache invalidate error: {exc}")
    return 0


def cache_stats() -> dict:
    """Return hit/miss stats (Redis INFO or DictCache.stats())."""
    client = get_client()
    if isinstance(client, _DictCache):
        return client.stats()
    try:
        info = client.info("stats")
        return {
            "backend":  "redis",
            "hits":     info.get("keyspace_hits",   0),
            "misses":   info.get("keyspace_misses", 0),
            "hit_rate": round(
                info.get("keyspace_hits", 0) /
                max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
                4
            ),
        }
    except Exception:
        return {"backend": "redis", "hits": 0, "misses": 0, "hit_rate": 0.0}
