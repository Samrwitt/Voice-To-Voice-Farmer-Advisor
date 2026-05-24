"""Voice ``/rag/answer`` grounding checks and escalation copy."""

from __future__ import annotations

import os


def voice_low_conf_escalation_enabled() -> bool:
    return os.getenv("RAG_VOICE_LOW_CONF_ESCALATE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def kb_grounded_for_voice(hits: list, best_distance: float, max_l2: float) -> bool:
    """
    True when retrieval has at least one in-threshold chunk **and** the pool
  ``best_distance`` is within threshold.

    Chroma mirror can add low-distance junk while Postgres ``best`` stays high;
    in that case we treat the turn as not KB-grounded.
    """
    if not hits:
        return False
    try:
        best = float(best_distance)
    except (TypeError, ValueError):
        best = 999.0
    if best > max_l2:
        return False
    return any(float(h.get("distance", 999)) <= max_l2 for h in hits)


def confident_kb_hits(hits: list, max_l2: float) -> list:
    return [h for h in (hits or []) if float(h.get("distance", 999)) <= max_l2]


GENERIC_LOW_CONFIDENCE_ESCALATION_AM = (
    "ይቅርታ፣ ይህንን ጥያቄ ሙሉ በሙሉ ልመልስ አልቻልኩም። "
    "ጥያቄዎን ለግብርና ባለሙያ አስተላልፌዋለሁ፤ በቅርቡ መልስ ያገኛሉ።"
)
