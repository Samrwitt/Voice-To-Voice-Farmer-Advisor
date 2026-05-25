"""Hosted LLM backends (Groq, Gemini) + helpers. Ollama stays in query.py.

**Multiple team keys:** set ``GROQ_API_KEYS`` and/or ``GEMINI_API_KEYS`` (comma-separated) so each
teammate's free-tier key is rotated round-robin; on 429/503 the next key is tried automatically.

Groq → Gemini: when ``RAG_LLM_BACKEND`` is groq and all Groq keys fail, ``groq_*_with_gemini_fallback``
calls Gemini (also pooled via ``GEMINI_API_KEYS`` / ``GEMINI_API_KEY``). Disable with ``GROQ_GEMINI_FALLBACK=0``.

When both hosted calls fail, ``query.py`` can fall back to local Ollama (``RAG_HOSTED_FALLBACK_OLLAMA``).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import httpx

from .api_key_pool import (
    gemini_api_keys,
    gemini_pool,
    groq_api_keys,
    groq_pool,
    run_with_key_pool,
)


def load_dotenv_if_present() -> None:
    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # Minimal parser without python-dotenv
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def gemini_api_key() -> str:
    """First configured Gemini key (pool or legacy single-key env)."""
    keys = gemini_api_keys()
    return keys[0] if keys else ""


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    if isinstance(exc, RuntimeError):
        s = str(exc)
        if "429" in s or "503" in s or "ወሰን" in s or "Too Many Requests" in s:
            return True
    return False


def effective_llm_backend() -> str:
    """groq | gemini | ollama | openai — see RAG_LLM_BACKEND, auto if unset."""
    b = os.environ.get("RAG_LLM_BACKEND", "").strip().lower()
    if b in ("groq", "gemini", "ollama", "openai"):
        return b
    # Use local Ollama first when keys exist but you want zero API unless Ollama fails (no auto-up to API).
    if os.environ.get("RAG_LOCAL_FIRST", "").strip().lower() in ("1", "true", "yes", "ollama"):
        if os.environ.get("USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes"):
            return "ollama"
    if gemini_api_keys():
        return "gemini"
    if groq_api_keys():
        return "groq"
    if os.environ.get("USE_OLLAMA", "1").strip() in ("1", "true", "yes"):
        return "ollama"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    return "ollama"


def groq_model(fast: bool) -> str:
    if os.environ.get("GROQ_MODEL", "").strip():
        return os.environ["GROQ_MODEL"].strip()
    return "llama-3.1-8b-instant" if fast else "llama-3.3-70b-versatile"


def gemini_model(fast: bool) -> str:
    if os.environ.get("GEMINI_MODEL", "").strip():
        return os.environ["GEMINI_MODEL"].strip()
    return "gemini-2.0-flash" if fast else "gemini-2.0-flash"


def _groq_retry_attempts() -> int:
    if os.environ.get("GROQ_RETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return 1
    return max(1, int(os.environ.get("GROQ_RETRY_ATTEMPTS", "6")))


def _groq_backoff_sec(attempt_index: int) -> float:
    base = float(os.environ.get("GROQ_RETRY_BACKOFF_BASE", "1.6"))
    cap = float(os.environ.get("GROQ_RETRY_BACKOFF_MAX", "45"))
    return min(cap, base * (2**attempt_index) + random.uniform(0.0, 0.35))


def _groq_rate_limit_hint() -> str:
    return (
        "Groq የጥያቄ ወሰን (429) — ከአንድ ወደ ሁለት ደቂቃ በኋላ ይሞክሩ፣ ወይም "
        "`.env` ውስጥ `RAG_LLM_BACKEND=gemini` ወይም `ollama` ያዘጋጁ።"
    )


def openai_style_chat(
    messages: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: float = 120.0,
    max_attempts: int | None = None,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.12 if "llama" in model.lower() else 0.15,
        "max_tokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
    }
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    is_groq = "api.groq.com" in base_url
    if max_attempts is not None:
        attempts = max(1, max_attempts)
    else:
        attempts = _groq_retry_attempts() if is_groq else 1
    data: dict | None = None
    for attempt in range(attempts):
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code in (429, 503) and attempt + 1 < attempts:
            time.sleep(_groq_backoff_sec(attempt))
            continue
        if r.status_code >= 400:
            if is_groq and r.status_code == 429 and attempt + 1 >= attempts:
                raise RuntimeError(_groq_rate_limit_hint()) from None
            if is_groq and r.status_code == 503 and attempt + 1 >= attempts:
                raise RuntimeError(
                    "Groq ሰርቨር ለአፊት ተወሰን (503) — ከጥቂት ጊዜ በኋላ ይሞክሩ ወይም RAG_LLM_BACKEND=gemini ይሞክሩ።"
                ) from None
            r.raise_for_status()
        data = r.json()
        break
    if data is None:
        raise RuntimeError("Groq: no response body after retries.")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def groq_chat_messages(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> str:
    pool = groq_pool()
    if pool.empty():
        raise RuntimeError("GROQ_API_KEY / GROQ_API_KEYS missing")

    def _one(key: str, _idx: int) -> str:
        return openai_style_chat(
            messages,
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
            model=groq_model(fast),
            timeout_sec=timeout_sec,
            max_attempts=2,
        )

    return run_with_key_pool(pool, _one, is_rate_limit=_is_rate_limit_error)


def _iter_groq_stream_with_key(
    messages: list[dict],
    *,
    key: str,
    fast: bool,
    timeout_sec: float,
):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": groq_model(fast),
        "messages": messages,
        "temperature": 0.12,
        "max_tokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
        "stream": True,
    }
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    with httpx.Client(timeout=tmo) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code in (429, 503):
                response.read()
                raise RuntimeError(f"Groq HTTP {response.status_code}")
            if response.status_code >= 400:
                response.read()
                response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_s = line[6:].strip()
                if data_s == "[DONE]":
                    break
                try:
                    data = json.loads(data_s)
                except json.JSONDecodeError:
                    continue
                delta = (data.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield piece


def iter_groq_chat(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
):
    pool = groq_pool()
    if pool.empty():
        yield "[groq] GROQ_API_KEY / GROQ_API_KEYS missing"
        return

    last_exc: BaseException | None = None
    for idx in pool.ordered_indices():
        key = pool.key_at(idx)
        try:
            yield from _iter_groq_stream_with_key(
                messages, key=key, fast=fast, timeout_sec=timeout_sec
            )
            return
        except BaseException as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                pool.mark_rate_limited(idx)
                continue
            raise
    if last_exc is not None:
        raise RuntimeError(_groq_rate_limit_hint()) from last_exc
    yield "[groq] all API keys rate-limited"


def _messages_to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return "\n\n".join(system_parts).strip(), contents


def _gemini_retry_attempts() -> int:
    if os.environ.get("GEMINI_RETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return 1
    return max(1, min(6, int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "3"))))


def _gemini_backoff_sec(attempt_index: int, response_text: str) -> float:
    """Parse retry hint from error JSON if present; else exponential backoff."""
    m = re.search(r"Please retry in ([0-9.]+)\s*s", response_text, re.I)
    if m:
        return min(60.0, float(m.group(1)) + random.uniform(0.15, 0.9))
    base = float(os.environ.get("GEMINI_RETRY_BACKOFF_BASE", "2.2"))
    cap = float(os.environ.get("GEMINI_RETRY_BACKOFF_MAX", "38"))
    return min(cap, base * (2**attempt_index) + random.uniform(0.0, 0.5))


def _gemini_chat_with_key(
    messages: list[dict],
    *,
    key: str,
    fast: bool,
    timeout_sec: float,
) -> str:
    system_text, contents = _messages_to_gemini_contents(messages)
    if not contents:
        raise RuntimeError("No user/model messages for Gemini")
    model = gemini_model(fast)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    per_key_attempts = min(2, _gemini_retry_attempts())
    last_detail = ""
    for attempt in range(per_key_attempts):
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=body)
        if r.status_code in (429, 503) and attempt + 1 < per_key_attempts:
            last_detail = (r.text or "")[:2000]
            time.sleep(_gemini_backoff_sec(attempt, last_detail))
            continue
        if r.status_code in (429, 503):
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {(r.text or '')[:500]}")
        if r.status_code >= 400:
            detail = (r.text or "")[:2000]
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {detail}")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"Gemini empty response: {data!r}"[:1500])
        parts = (cands[0].get("content") or {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
    raise RuntimeError(f"Gemini: exhausted retries ({last_detail})")


def gemini_chat_messages(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> str:
    pool = gemini_pool()
    if pool.empty():
        raise RuntimeError("No Gemini API key (set GEMINI_API_KEYS or GEMINI_API_KEY)")

    def _one(key: str, _idx: int) -> str:
        return _gemini_chat_with_key(messages, key=key, fast=fast, timeout_sec=timeout_sec)

    return run_with_key_pool(pool, _one, is_rate_limit=_is_rate_limit_error)


def gemini_groq_fallback_enabled() -> bool:
    if os.environ.get("GEMINI_GROQ_FALLBACK", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return bool(groq_api_keys())


def should_fallback_gemini_to_groq(exc: BaseException) -> bool:
    if not gemini_groq_fallback_enabled():
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503, 400)
    if isinstance(exc, RuntimeError):
        s = str(exc)
        if "429" in s or "503" in s or "400" in s:
            return True
        if "Too Many Requests" in s or "Quota" in s or "Exceeded" in s:
            return True
    return False


def gemini_chat_messages_with_groq_fallback(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> tuple[str, str]:
    """Returns (reply_text, backend_used): ``gemini`` or ``groq`` when Gemini hits 429/503/400."""
    try:
        return (
            gemini_chat_messages(messages, fast=fast, timeout_sec=timeout_sec),
            "gemini",
        )
    except (RuntimeError, httpx.HTTPStatusError) as e:
        if should_fallback_gemini_to_groq(e):
            return (
                groq_chat_messages(messages, fast=fast, timeout_sec=timeout_sec),
                "groq",
            )
        raise


def iter_gemini_chat_with_groq_fallback(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
):
    """Stream Gemini completion, or fallback to Groq if Gemini fails with 429/503/400."""
    try:
        # Since Gemini implementation doesn't stream here (it returns full), yield full
        yield gemini_chat_messages(messages, fast=fast, timeout_sec=timeout_sec)
    except (RuntimeError, httpx.HTTPStatusError) as e:
        if should_fallback_gemini_to_groq(e):
            yield from iter_groq_chat(messages, fast=fast, timeout_sec=timeout_sec)
            return
        raise

load_dotenv_if_present()