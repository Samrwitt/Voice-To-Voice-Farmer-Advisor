"""Transparency metadata for voice / API responses (grounding, SLA hints)."""

from __future__ import annotations

import os
from typing import Any

# Short Amharic footer when ``RAG_TRUST_FOOTER=1`` (informational-only disclaimer).
TRUST_FOOTER_AM = (
    "ማሳሰቢያ፦ እነዚህ ምክሮች ከመርሃ መመሪያ ሰነዶች የተገነዘቡ አጠቃላይ መረጃዎች ብቻ ናቸው። "
    "ለመርቢያ እርሻ ጉዳዮች በአካል ባለሙያ ይጠይቁ።"
)


def trust_footer_enabled() -> bool:
    return os.getenv("RAG_TRUST_FOOTER", "0").strip().lower() in ("1", "true", "yes", "on")


def maybe_append_trust_footer(final_text: str, *, sources: list[str]) -> str:
    if not trust_footer_enabled():
        return final_text
    if not (final_text or "").strip():
        return final_text
    if "escalation" in sources or "kb" not in sources:
        return final_text
    if TRUST_FOOTER_AM in final_text:
        return final_text
    return f"{final_text.rstrip()}\n\n{TRUST_FOOTER_AM}"


def build_voice_trust_meta(
    *,
    hits: list[dict[str, Any]],
    used_llm_assistant: bool,
    used_chunk_compose: bool,
    sources: list[str],
    escalated_empty: bool,
    latency_ms: float,
    sla_target_hours: int,
) -> dict[str, Any]:
    n = len(hits or [])
    base = {
        "sources": list(sources),
        "kb_chunks_used": n,
        "sources_in_prompt": min(n, 3),
        "latency_ms": round(latency_ms, 1),
        "escalation_sla_target_hours": sla_target_hours,
        "human_review": escalated_empty,
    }
    if escalated_empty:
        base["grounding"] = "escalation"
        return base
    if "kb" not in sources:
        base["grounding"] = "dynamic_only" if "dynamic" in sources else "none"
        return base
    if used_llm_assistant:
        base["grounding"] = "kb_llm"
    elif used_chunk_compose:
        base["grounding"] = "kb_compose"
    else:
        base["grounding"] = "kb_unknown"
    return base
