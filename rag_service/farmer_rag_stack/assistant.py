"""
GPT-style grounded answers: RAG folder prompts + LLM routing over Postgres KB hits.
Optional web/weather tools (``rag_tools.augment``). Falls back to None for chunk-only path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .context_utils import build_context, sanitize_chat_answer
from .llm_providers import effective_llm_backend, load_dotenv_if_present
from .nlu_farmer import parse_farmer_nlu
from .query_llm import (
    hosted_ollama_fallback_enabled,
    ollama_failover_answer,
    prepare_rag_llm_messages,
    run_sync_llm,
)
from .rag_prompts import system_for_rag
from .rag_tools.augment import augment_kb_context
from .retrieval_ranking import pg_hits_to_rank_rows

logger = logging.getLogger("rag_service.assistant")


def _fast_mode() -> bool:
    return os.environ.get("RAG_MODE", "fast").strip().lower() not in (
        "quality",
        "slow",
        "accurate",
    )


def _looks_like_llm_error(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t.startswith("[Ollama]"):
        return True
    if t.startswith("[groq]") or t.startswith("[gemini]") or t.startswith("[openai]"):
        return True
    if "አልተዋቀረም" in t and ("API" in t or "KEY" in t):
        return True
    return False


def try_llm_assistant_response(
    *,
    query_text: str,
    session_id: str,
    hits: list[dict[str, Any]],
    user_context: str,
    alerts_text: str,
    dynamic_block: str,
    history_pairs: list[tuple[str, str]],
) -> str | None:
    """
    Returns assistant answer in Amharic, or None to use non-LLM composition.

    Env:
      RAG_ASSISTANT_LLM — default ``1``; set ``0`` to disable.
      RAG_CONTEXT_CHARS — max KB chars in prompt (fast default 4200, else 9000).
      RAG_TOOLS / RAG_WEB_MODE / RAG_WEATHER_TOOL — see ``rag_tools/augment.py``.
      RAG_VOICE_MAX_CHARS — cap final answer (0 = no cap). Default 900.
    """
    del session_id  # reserved for future logging / tracing
    if os.environ.get("RAG_ASSISTANT_LLM", "1").strip().lower() in ("0", "false", "no", "off"):
        return None

    load_dotenv_if_present()

    rows = pg_hits_to_rank_rows(hits)
    if not rows or not any((r.get("text") or "").strip() for r in rows):
        return None

    fast = _fast_mode()
    try:
        extra_ctx, _tool_trace = augment_kb_context(query_text, rows, fast=fast)
    except Exception as exc:
        logger.warning("augment_kb_context failed: %s", exc)
        extra_ctx, _tool_trace = "", []

    base_max = int(os.environ.get("RAG_CONTEXT_CHARS", "4200" if fast else "9000"))
    reserved = min(len(extra_ctx) + 400, 3600) if (extra_ctx or "").strip() else 0
    max_chars = max(900, base_max - reserved)
    ctx = build_context(rows, max_chars=max_chars, compact=fast)

    conv = [{"role": r, "content": (m or "").strip()} for r, m in history_pairs if (m or "").strip()]
    conv = [m for m in conv if m["role"] in ("user", "assistant")]

    preamble_parts: list[str] = []
    if (user_context or "").strip():
        preamble_parts.append((user_context or "").strip())
    if (alerts_text or "").strip():
        preamble_parts.append((alerts_text or "").strip())
    if (dynamic_block or "").strip():
        preamble_parts.append((dynamic_block or "").strip())
    preamble = "\n".join(preamble_parts).strip()
    pre_block = f"ቅድመ መረጃ፦\n{preamble}\n\n" if preamble else ""

    has_aux = bool((extra_ctx or "").strip())
    aux_block = (
        f"ተጨማሪ (ድር / መሳሪያ — ከመመሪያ ቤት ይለያል፤ [W1] …)፦\n{extra_ctx}\n\n"
        if has_aux
        else ""
    )

    user_block = (
        f"{pre_block}"
        f"ጥያቄ፦ {query_text.strip()}\n\n"
        f"መረጃ (ከመመሪያ ቤት)፦\n{ctx}\n\n"
        f"{aux_block}"
    )

    farmer_nlu = parse_farmer_nlu(query_text)
    system = system_for_rag(
        fast,
        has_aux_context=has_aux,
        follow_up=bool(conv),
        nlu=farmer_nlu,
    )
    backend = effective_llm_backend()
    msgs = prepare_rag_llm_messages(system, conv, user_block, backend)

    try:
        answer, llm_used = run_sync_llm(backend, msgs, fast)
    except Exception as exc:
        logger.warning("LLM primary failed: %s", exc)
        answer = ""
        llm_used = backend
        if backend in ("groq", "gemini", "openai") and hosted_ollama_fallback_enabled():
            try:
                answer = ollama_failover_answer(msgs, fast)
                llm_used = "ollama"
            except Exception as fe:
                logger.warning("Ollama fallback failed: %s", fe)
                answer = f"[{backend}]\n{exc}\n[Ollama]\n{fe}"

    answer = sanitize_chat_answer(answer)
    if _looks_like_llm_error(answer):
        logger.info("LLM path unusable (%s); falling back to chunk composition.", llm_used)
        return None

    cap_raw = os.environ.get("RAG_VOICE_MAX_CHARS", "900").strip()
    if cap_raw and cap_raw not in ("0", "off", "false", "no"):
        try:
            cap = int(cap_raw)
            if cap > 0 and len(answer) > cap:
                answer = answer[: max(0, cap - 3)].rstrip() + "..."
        except ValueError:
            pass

    return answer
