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


def maybe_append_trust_footer(final_text: str, *, grounding: str) -> str:
    if not trust_footer_enabled():
        return final_text
    if not (final_text or "").strip():
        return final_text
    if grounding in ("escalation", "none", "dynamic_only"):
        return final_text
    if TRUST_FOOTER_AM in final_text:
        return final_text
    return f"{final_text.rstrip()}\n\n{TRUST_FOOTER_AM}"


def build_voice_trust_meta(
    *,
    hits: list[dict[str, Any]],
    used_llm_assistant: bool,
    used_chunk_compose: bool,
    dynamic_prefixed: bool,
    escalated_empty: bool,
    latency_ms: float,
    sla_target_hours: int,
) -> dict[str, Any]:
    n = len(hits or [])
    if escalated_empty:
        return {
            "grounding": "escalation",
            "kb_chunks_used": n,
            "sources_in_prompt": n,
            "human_review": True,
            "latency_ms": round(latency_ms, 1),
            "escalation_sla_target_hours": sla_target_hours,
        }
    if n == 0:
        return {
            "grounding": "none",
            "kb_chunks_used": 0,
            "sources_in_prompt": 0,
            "human_review": False,
            "latency_ms": round(latency_ms, 1),
            "escalation_sla_target_hours": sla_target_hours,
        }
    if dynamic_prefixed and not (used_llm_assistant or used_chunk_compose):
        g = "dynamic_only"
    elif used_llm_assistant:
        g = "kb_llm"
    elif used_chunk_compose:
        g = "kb_compose"
    else:
        g = "unknown"
    return {
        "grounding": g,
        "kb_chunks_used": n,
        "sources_in_prompt": min(n, 3),
        "human_review": False,
        "latency_ms": round(latency_ms, 1),
        "escalation_sla_target_hours": sla_target_hours,
    }
