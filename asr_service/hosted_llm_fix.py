"""
Post-ASR typo / grammar fix via Groq or Gemini (same API keys as RAG).

Env (shared with rag-service):
  GROQ_API_KEYS / GROQ_API_KEY
  GEMINI_API_KEYS / GEMINI_API_KEY / GOOGLE_API_KEY
  ASR_HOSTED_LLM_FIX=auto|1|0
  ASR_LLM_FIX_BACKEND=groq_then_gemini|groq|gemini
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger("asr-hosted-llm-fix")

T = TypeVar("T")
_SPLIT_RE = re.compile(r"[,;\n|]+")


def _parse_keys(*env_names: str) -> list[str]:
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


def groq_keys() -> list[str]:
    return _parse_keys("GROQ_API_KEYS", "GROQ_API_KEY")


def gemini_keys() -> list[str]:
    keys = _parse_keys("GEMINI_API_KEYS")
    if keys:
        return keys
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return [v]
    return []


def hosted_fix_enabled() -> bool:
    raw = os.getenv("ASR_HOSTED_LLM_FIX", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return bool(groq_keys() or gemini_keys())
    # auto
    return bool(groq_keys() or gemini_keys())


def _backend_mode() -> str:
    return os.getenv("ASR_LLM_FIX_BACKEND", "groq_then_gemini").strip().lower()


class _KeyPool:
    def __init__(self, keys: list[str], label: str) -> None:
        self._keys = list(keys)
        self._label = label
        self._lock = threading.Lock()
        self._rr = 0
        self._cooldown: dict[int, float] = {}

    def empty(self) -> bool:
        return not self._keys

    def ordered_indices(self) -> list[int]:
        now = time.monotonic()
        with self._lock:
            if not self._keys:
                return []
            start = self._rr % len(self._keys)
            self._rr += 1
            avail = {i for i in range(len(self._keys)) if self._cooldown.get(i, 0) <= now}
            order = [(start + o) % len(self._keys) for o in range(len(self._keys))]
            warm = [i for i in order if i in avail]
            rest = [i for i in order if i not in avail]
            return warm + rest

    def mark_rate_limited(self, index: int) -> None:
        with self._lock:
            self._cooldown[index] = time.monotonic() + float(
                os.getenv("ASR_LLM_FIX_COOLDOWN_SEC", "90") or "90"
            )

    def key_at(self, index: int) -> str:
        return self._keys[index]


_groq_pool: _KeyPool | None = None
_gemini_pool: _KeyPool | None = None


def _groq_pool() -> _KeyPool:
    global _groq_pool
    if _groq_pool is None:
        _groq_pool = _KeyPool(groq_keys(), "groq")
    return _groq_pool


def _gemini_pool() -> _KeyPool:
    global _gemini_pool
    if _gemini_pool is None:
        _gemini_pool = _KeyPool(gemini_keys(), "gemini")
    return _gemini_pool


def _is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    s = str(exc)
    return "429" in s or "503" in s or "Too Many Requests" in s


def _run_pool(pool: _KeyPool, fn: Callable[[str], T]) -> T:
    last: BaseException | None = None
    for idx in pool.ordered_indices():
        try:
            return fn(pool.key_at(idx))
        except BaseException as exc:
            last = exc
            if _is_rate_limit(exc):
                pool.mark_rate_limited(idx)
                continue
            raise
    if last is not None:
        raise last
    raise RuntimeError(f"{pool._label}: no API keys")


def _groq_model() -> str:
    return (os.getenv("ASR_GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()


def _gemini_model() -> str:
    return (os.getenv("ASR_GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()


def _fix_prompt(raw: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You fix Amharic speech-to-text transcripts for Ethiopian farmers. "
                "Correct spelling, word boundaries, and obvious ASR mistakes. "
                "Keep the same meaning and agricultural terms. "
                "Do not translate. Do not add sentences. "
                "Return ONLY the corrected Amharic text, nothing else."
            ),
        },
        {"role": "user", "content": f"Raw ASR transcript:\n{raw}\n\nCorrected:"},
    ]


def _chat_groq(key: str, messages: list[dict]) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": _groq_model(),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": int(os.getenv("ASR_LLM_FIX_MAX_TOKENS", "256")),
    }
    timeout = float(os.getenv("ASR_LLM_FIX_TIMEOUT_SEC", "25"))
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code in (429, 503):
            r.raise_for_status()
        r.raise_for_status()
        data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _chat_gemini(key: str, messages: list[dict]) -> str:
    system = ""
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if role == "system":
            system = text
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
    model = _gemini_model()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(os.getenv("ASR_LLM_FIX_MAX_TOKENS", "256")),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    timeout = float(os.getenv("ASR_LLM_FIX_TIMEOUT_SEC", "25"))
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=body)
        if r.status_code in (429, 503):
            r.raise_for_status()
        r.raise_for_status()
        data = r.json()
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _sanitize_llm_output(text: str, fallback: str) -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    # Drop common English preambles
    t = re.sub(r"^(corrected|here is|the corrected).*?:\s*", "", t, flags=re.I).strip()
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1].strip()
    # Must still look like Amharic farmer text
    if len(t) < 2:
        return fallback
    return t


def semantic_correction_hosted(raw_text: str) -> tuple[str, str]:
    """
    Returns ``(corrected_text, backend)`` where backend is ``groq`` or ``gemini``.
  """
    text = (raw_text or "").strip()
    if not text:
        return text, "none"

    messages = _fix_prompt(text)
    mode = _backend_mode()
    errors: list[str] = []

    def try_groq() -> str:
        pool = _groq_pool()
        if pool.empty():
            raise RuntimeError("no groq keys")
        return _run_pool(pool, lambda k: _chat_groq(k, messages))

    def try_gemini() -> str:
        pool = _gemini_pool()
        if pool.empty():
            raise RuntimeError("no gemini keys")
        return _run_pool(pool, lambda k: _chat_gemini(k, messages))

    order: list[tuple[str, Callable[[], str]]] = []
    if mode == "gemini":
        order = [("gemini", try_gemini)]
    elif mode == "groq":
        order = [("groq", try_groq)]
    else:
        order = [("groq", try_groq), ("gemini", try_gemini)]

    for name, fn in order:
        try:
            out = _sanitize_llm_output(fn(), text)
            logger.info("ASR hosted fix via %s: %r -> %r", name, text[:80], out[:80])
            return out, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("ASR hosted fix %s failed: %s", name, exc)

    logger.warning("ASR hosted fix exhausted: %s", "; ".join(errors))
    return text, "none"
