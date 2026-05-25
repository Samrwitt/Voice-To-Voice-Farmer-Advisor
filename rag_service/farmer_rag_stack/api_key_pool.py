"""
Round-robin API key pools for Groq / Gemini (team keys in one .env).

Configure multiple keys (one per teammate)::

  GROQ_API_KEYS=key1,key2,key3,key4,key5
  GEMINI_API_KEYS=key1,key2,key3,key4,key5

Legacy single-key vars still work: ``GROQ_API_KEY``, ``GEMINI_API_KEY``.

On HTTP 429/503 the key is cooled down briefly and the next key is tried.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_SPLIT_RE = re.compile(r"[,;\n|]+")


def parse_api_keys(*env_names: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in env_names:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        for part in _SPLIT_RE.split(raw):
            k = part.strip().strip('"').strip("'")
            if k and k not in seen:
                seen.add(k)
                out.append(k)
    return out


def groq_api_keys() -> list[str]:
    return parse_api_keys("GROQ_API_KEYS", "GROQ_API_KEY")


def gemini_api_keys() -> list[str]:
    keys = parse_api_keys("GEMINI_API_KEYS")
    if keys:
        return keys
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return [v]
    return []


def _cooldown_sec() -> float:
    try:
        return max(15.0, float(os.environ.get("RAG_API_KEY_COOLDOWN_SEC", "90")))
    except ValueError:
        return 90.0


class ApiKeyPool:
    def __init__(self, keys: list[str], *, label: str) -> None:
        self._keys = list(keys)
        self._label = label
        self._lock = threading.Lock()
        self._rr = 0
        self._cooldown_until: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def empty(self) -> bool:
        return not self._keys

    def _available_indices(self, now: float) -> list[int]:
        avail = [i for i in range(len(self._keys)) if self._cooldown_until.get(i, 0) <= now]
        if avail:
            return avail
        return list(range(len(self._keys)))

    def ordered_indices(self) -> list[int]:
        """Round-robin order: start index rotates; cooled keys tried last."""
        now = time.monotonic()
        with self._lock:
            if not self._keys:
                return []
            start = self._rr % len(self._keys)
            self._rr += 1
            avail = set(self._available_indices(now))
            order = [(start + o) % len(self._keys) for o in range(len(self._keys))]
            warm = [i for i in order if i in avail]
            rest = [i for i in order if i not in avail]
            return warm + rest

    def mark_rate_limited(self, index: int) -> None:
        until = time.monotonic() + _cooldown_sec()
        with self._lock:
            self._cooldown_until[index] = until

    def key_at(self, index: int) -> str:
        return self._keys[index]


_groq_pool: ApiKeyPool | None = None
_gemini_pool: ApiKeyPool | None = None
_pool_lock = threading.Lock()


def groq_pool() -> ApiKeyPool:
    global _groq_pool
    with _pool_lock:
        if _groq_pool is None:
            _groq_pool = ApiKeyPool(groq_api_keys(), label="groq")
        return _groq_pool


def gemini_pool() -> ApiKeyPool:
    global _gemini_pool
    with _pool_lock:
        if _gemini_pool is None:
            _gemini_pool = ApiKeyPool(gemini_api_keys(), label="gemini")
        return _gemini_pool


def reset_pools_for_tests() -> None:
    global _groq_pool, _gemini_pool
    with _pool_lock:
        _groq_pool = None
        _gemini_pool = None


def run_with_key_pool(
    pool: ApiKeyPool,
    fn: Callable[[str, int], T],
    *,
    is_rate_limit: Callable[[BaseException], bool],
) -> T:
    """Call ``fn(api_key, key_index)`` trying each key in round-robin order."""
    if pool.empty():
        raise RuntimeError(f"{pool._label}: no API keys configured")
    last_exc: BaseException | None = None
    for idx in pool.ordered_indices():
        key = pool.key_at(idx)
        try:
            return fn(key, idx)
        except BaseException as exc:
            last_exc = exc
            if is_rate_limit(exc):
                pool.mark_rate_limited(idx)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{pool._label}: all keys failed")