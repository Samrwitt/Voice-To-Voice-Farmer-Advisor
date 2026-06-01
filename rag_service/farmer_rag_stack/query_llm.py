"""LLM routing for RAG: message shaping, hosted APIs, and quota fallbacks."""

from __future__ import annotations

import os
from collections.abc import Iterator

from .llm_providers import (
    gemini_chat_messages,
    gemini_chat_messages_with_groq_fallback,
    iter_gemini_chat_with_groq_fallback,
    openai_style_chat,
)


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


def _base_messages(
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
    msgs = _base_messages(system, conversation, user_block)
    if backend in ("groq", "gemini", "openai"):
        msgs = trim_hosted_conversation_messages(msgs)
        msgs = shrink_messages_for_hosted_api(msgs)
    return msgs


def hosted_llm_timeout(fast: bool) -> float:
    return float(os.environ.get("RAG_HOSTED_HTTP_TIMEOUT", "120" if fast else "240"))


def run_sync_llm(backend: str, msgs: list[dict], fast: bool) -> tuple[str, str]:
    """Single non-streaming LLM call. Returns (answer_text, llm_used)."""
    t = hosted_llm_timeout(fast)
    if backend == "gemini":
        return gemini_chat_messages_with_groq_fallback(msgs, fast=fast, timeout_sec=t)
    if backend == "groq":
        return gemini_chat_messages_with_groq_fallback(msgs, fast=fast, timeout_sec=t)
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
    return (
        f"Unsupported backend {backend!r}. RAG_LLM_BACKEND=groq ወይም gemini ይጠቀሙ።",
        backend,
    )


def iter_primary_llm(backend: str, msgs: list[dict], fast: bool) -> Iterator[str]:
    t = hosted_llm_timeout(fast)
    if backend == "gemini":
        yield from iter_gemini_chat_with_groq_fallback(msgs, fast=fast, timeout_sec=t)
        return
    if backend == "groq":
        yield from iter_gemini_chat_with_groq_fallback(msgs, fast=fast, timeout_sec=t)
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
    yield f"Unsupported backend {backend!r}. RAG_LLM_BACKEND=groq ወይም gemini ይጠቀሙ።"
