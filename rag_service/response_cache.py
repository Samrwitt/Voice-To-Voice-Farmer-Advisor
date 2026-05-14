"""Short TTL cache for stable /rag/answer payloads (KB-only, no dynamic block)."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any

_lock = threading.Lock()
_store: dict[str, tuple[float, dict[str, Any]]] = {}


def cache_ttl_seconds() -> float:
    raw = os.getenv("RAG_RESPONSE_CACHE_TTL_SEC", "0").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 0.0
    return max(0.0, v)


def cache_max_entries() -> int:
    raw = os.getenv("RAG_RESPONSE_CACHE_MAX", "512").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 512
    return max(16, min(n, 5000))


def make_rag_cache_key(*, query_text: str, phone_number: str, user_region: str | None) -> str:
    p = (phone_number or "").strip()
    r = (user_region or "").strip()
    q = (query_text or "").strip().lower()
    h = hashlib.sha256(f"{q}|{p}|{r}".encode("utf-8")).hexdigest()
    return f"rag:v1:{h}"


def get(key: str) -> dict[str, Any] | None:
    if cache_ttl_seconds() <= 0:
        return None
    now = time.monotonic()
    with _lock:
        ent = _store.get(key)
        if not ent:
            return None
        exp, payload = ent
        if now > exp:
            del _store[key]
            return None
        return dict(payload)


def set(key: str, payload: dict[str, Any], ttl_sec: float | None = None) -> None:
    ttl = ttl_sec if ttl_sec is not None else cache_ttl_seconds()
    if ttl <= 0:
        return
    now = time.monotonic()
    with _lock:
        _store[key] = (now + ttl, dict(payload))
        max_e = cache_max_entries()
        while len(_store) > max_e:
            # Drop oldest few entries by expiry time (cheap trim).
            sorted_keys = sorted(_store.keys(), key=lambda k: _store[k][0])
            for k in sorted_keys[: max(1, len(_store) // 4)]:
                _store.pop(k, None)
