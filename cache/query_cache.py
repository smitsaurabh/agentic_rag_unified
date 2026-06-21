"""
In-memory TTL result cache for query responses.

• Thread-safe (uses a lock).
• Evicts expired entries on access.
• Configurable TTL and max size via environment settings.
• Only caches Medium/High-confidence answers.
"""

from __future__ import annotations

import hashlib
import time
import threading
import logging
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── Internal cache store ───────────────────────────────────────────────────────

_cache: dict[str, tuple[Any, float]] = {}      # key → (value, inserted_at)
_lock  = threading.Lock()

_TTL     = getattr(settings, "query_cache_ttl", 300)   # seconds (default 5 min)
_MAXSIZE = getattr(settings, "query_cache_maxsize", 100)


# ── Public API ─────────────────────────────────────────────────────────────────

def _cache_key(query: str, domain: str | None, company: str | None) -> str:
    """Stable, normalised cache key from query parameters."""
    raw = f"{query.lower().strip()}|{domain or ''}|{company or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    query: str,
    domain: Optional[str] = None,
    company: Optional[str] = None,
) -> Optional[Any]:
    """Return cached result or None if missing/expired."""
    key = _cache_key(query, domain, company)
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        value, inserted_at = entry
        if time.time() - inserted_at > _TTL:
            del _cache[key]
            return None
        logger.info("Cache HIT — key=%s", key)
        return value


def set_cached(
    query: str,
    result: Any,
    domain: Optional[str] = None,
    company: Optional[str] = None,
    confidence: str = "Medium",
) -> None:
    """Store result. Only caches Medium or High confidence answers."""
    if confidence.lower() == "low":
        logger.debug("Cache SKIP — Low confidence answer not cached")
        return

    key = _cache_key(query, domain, company)
    with _lock:
        # Evict oldest entry if at capacity
        if len(_cache) >= _MAXSIZE:
            oldest_key = min(_cache, key=lambda k: _cache[k][1])
            del _cache[oldest_key]
            logger.debug("Cache EVICT — key=%s", oldest_key)
        _cache[key] = (result, time.time())
        logger.info("Cache SET — key=%s ttl=%ss", key, _TTL)


def cache_stats() -> dict[str, Any]:
    """Return current cache statistics."""
    with _lock:
        now = time.time()
        alive = sum(1 for _, (_, ts) in _cache.items() if now - ts <= _TTL)
        return {
            "total_entries": len(_cache),
            "live_entries":  alive,
            "ttl_seconds":   _TTL,
            "max_size":      _MAXSIZE,
        }


def clear_cache() -> int:
    """Flush the entire cache. Returns number of entries cleared."""
    with _lock:
        n = len(_cache)
        _cache.clear()
        logger.info("Cache CLEARED — %d entries removed", n)
        return n
