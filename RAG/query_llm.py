"""LLM routing for RAG: message shaping, hosted APIs, Ollama, and quota fallbacks."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator

import httpx

from llm_providers import (
    gemini_chat_messages,
    groq_chat_messages_with_gemini_fallback,
    iter_groq_chat_with_gemini_fallback,
    openai_style_chat,
)


def default_chat_model(fast: bool) -> str:
    if os.environ.get("OLLAMA_MODEL", "").strip():
        return os.environ["OLLAMA_MODEL"].strip()
    return "qwen2.5:3b" if fast else "qwen3:4b-instruct"


def ollama_options(fast: bool) -> dict:
    raw = os.environ.get("OLLAMA_OPTIONS_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("Warning: OLLAMA_OPTIONS_JSON invalid JSON, using preset.", file=sys.stderr)
    if fast:
        return {
            "temperature": 0.06,
            "top_p": 0.82,
            "repeat_penalty": 1.12,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "3072")),
            "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "240")),
        }
    return {
        "temperature": 0.05,
        "top_p": 0.9,
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "512")),
    }


def _hosted_chat_rounds_limit() -> int:
    v = os.environ.get("RAG_HOSTED_CHAT_ROUNDS", "3").strip().lower()
    if not v or v in ("0", "off", "unlimited", "no", "false"):
        return 0
    try:
        return max(0, int(v))
    except ValueError:
        return 3


def trim_hosted_conversation_messages(messages: list[dict]) -> list[dict]:
    if len(messages) <= 2:
        return messages
    if (messages[0].get("role") or "") != "system":
        return messages
    limit = _hosted_chat_rounds_limit()
    if limit == 0:
        return messages
    system = messages[0]
    tail = messages[-1]
    mid = messages[1:-1]
    max_mid = limit * 2
    if len(mid) <= max_mid:
        return messages
    return [system] + mid[-max_mid:] + [tail]


def hosted_ollama_fallback_enabled() -> bool:
    if os.environ.get("RAG_HOSTED_FALLBACK_OLLAMA", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return os.environ.get("USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes")


def shrink_messages_for_hosted_api(messages: list[dict]) -> list[dict]:
    cap = max(12_000, int(os.environ.get("RAG_HOSTED_MESSAGES_MAX_CHARS", "26000")))
    mark = "\n\n…(ለ API መጠን ተሰነዘለ)…"
    out: list[dict] = [{"role": m["role"], "content": str(m.get("content") or "")} for m in messages]

    def total() -> int:
        return sum(len(x["content"]) for x in out)

    for _ in range(32):
        if total() <= cap:
            return out
        over = total() - cap + len(mark) + 40
        cut_idx: int | None = None
        for idx in range(len(out) - 1, -1, -1):
            if out[idx]["role"] not in ("user", "assistant"):
                continue
            c = out[idx]["content"]
            if len(c) < 2800:
                continue
            cut_idx = idx
            break
        if cut_idx is not None:
            c = out[cut_idx]["content"]
            new_len = max(2500, len(c) - max(over, int(0.12 * len(c))))
            out[cut_idx]["content"] = c[:new_len].rstrip() + mark
            continue
        li = max(range(len(out)), key=lambda i: len(out[i]["content"]))
        c = out[li]["content"]
        if len(c) <= 1800:
            break
        new_len = max(1500, len(c) - over)
        out[li]["content"] = c[:new_len].rstrip() + mark
    return out


def ollama_chat_messages(
    messages: list[dict],
    model: str,
    base_url: str,
    *,
    options: dict,
    timeout_sec: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    tmo = httpx.Timeout(connect=20.0, read=timeout_sec, write=120.0, pool=10.0)
    try:
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=payload)
    except httpx.ReadTimeout as e:
        raise RuntimeError(
            f"Ollama read timed out after {timeout_sec:g}s (model may be loading on CPU/GPU). "
            f"Try: export OLLAMA_HTTP_TIMEOUT=180  or  ollama run {model}  once to warm the model."
        ) from e
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url!r}. Is `ollama serve` running?"
        ) from e
    if r.status_code >= 400:
        detail = (r.text or "").strip()[:2500]
        raise RuntimeError(
            f"Ollama HTTP {r.status_code} for model {model!r}. "
            f"Often: out of memory — try export OLLAMA_NUM_CTX=2048 or a smaller model. Body:\n{detail}"
        )
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def iter_ollama_chat(
    messages: list[dict],
    model: str,
    base_url: str,
    *,
    options: dict,
    timeout_sec: float,
):
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    tmo = httpx.Timeout(connect=20.0, read=timeout_sec, write=120.0, pool=10.0)
    with httpx.Client(timeout=tmo) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                piece = (data.get("message") or {}).get("content") or ""
                if piece:
                    yield piece


def _ollama_messages(
    system: str,
    conversation: list[dict] | None,
    user_block: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system}]
    if conversation:
        for m in conversation[-12:]:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            c = (m.get("content") or "").strip()
            if not c:
                continue
            messages.append({"role": role, "content": c[:12000]})
    messages.append({"role": "user", "content": user_block})
    return messages


def prepare_rag_llm_messages(
    system: str,
    conversation: list[dict] | None,
    user_block: str,
    backend: str,
) -> list[dict]:
    msgs = _ollama_messages(system, conversation, user_block)
    if backend in ("groq", "gemini", "openai"):
        msgs = trim_hosted_conversation_messages(msgs)
        msgs = shrink_messages_for_hosted_api(msgs)
    return msgs


def _ollama_runtime(fast: bool) -> tuple[str, str, dict, float]:
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = default_chat_model(fast)
    opts = ollama_options(fast)
    timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
    return base, model, opts, timeout


def ollama_failover_answer(msgs: list[dict], fast: bool) -> str:
    base, model, opts, timeout = _ollama_runtime(fast)
    return ollama_chat_messages(msgs, model, base, options=opts, timeout_sec=timeout)


def hosted_llm_timeout(fast: bool) -> float:
    return float(os.environ.get("RAG_HOSTED_HTTP_TIMEOUT", "120" if fast else "240"))


def run_sync_llm(backend: str, msgs: list[dict], fast: bool) -> tuple[str, str]:
    """Single non-streaming LLM call. Returns (answer_text, llm_used)."""
    t = hosted_llm_timeout(fast)
    if backend == "groq":
        return groq_chat_messages_with_gemini_fallback(msgs, fast=fast, timeout_sec=t)
    if backend == "gemini":
        return gemini_chat_messages(msgs, fast=fast, timeout_sec=t), "gemini"
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not key:
            return (
                "OPENAI_API_KEY አልተዋቀረም። RAG_LLM_BACKEND=groq ወይም gemini ይሞክሩ።",
                "openai",
            )
        return (
            openai_style_chat(
                msgs,
                base_url=base,
                api_key=key,
                model=model,
                timeout_sec=t,
            ),
            "openai",
        )
    if os.environ.get("USE_OLLAMA", "1").strip() not in ("1", "true", "yes"):
        return (
            "USE_OLLAMA=0 ነው። RAG_LLM_BACKEND=groq ወይም gemini ይመርጡ ወይም USE_OLLAMA=1 ያድርጉ።",
            backend,
        )
    base, model, opts, timeout = _ollama_runtime(fast)
    try:
        return ollama_chat_messages(msgs, model, base, options=opts, timeout_sec=timeout), "ollama"
    except RuntimeError as e:
        return f"[Ollama]\n{e}", "ollama"


def iter_primary_llm(backend: str, msgs: list[dict], fast: bool) -> Iterator[str]:
    t = hosted_llm_timeout(fast)
    if backend == "groq":
        yield from iter_groq_chat_with_gemini_fallback(msgs, fast=fast, timeout_sec=t)
        return
    if backend == "gemini":
        yield gemini_chat_messages(msgs, fast=fast, timeout_sec=t)
        return
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not key:
            yield "OPENAI_API_KEY አልተዋቀረም።"
            return
        yield openai_style_chat(
            msgs,
            base_url=base,
            api_key=key,
            model=model,
            timeout_sec=t,
        )
        return
    if os.environ.get("USE_OLLAMA", "1").strip() not in ("1", "true", "yes"):
        yield "USE_OLLAMA=0 — RAG_LLM_BACKEND=groq ወይም gemini ይጠቀሙ።"
        return
    base, model, opts, timeout = _ollama_runtime(fast)
    try:
        yield from iter_ollama_chat(msgs, model, base, options=opts, timeout_sec=timeout)
    except (RuntimeError, httpx.HTTPError, httpx.RequestError) as e:
        yield f"[Ollama]\n{e}"


def iter_ollama_failover(msgs: list[dict], fast: bool) -> Iterator[str]:
    base, model, opts, timeout = _ollama_runtime(fast)
    yield from iter_ollama_chat(msgs, model, base, options=opts, timeout_sec=timeout)
