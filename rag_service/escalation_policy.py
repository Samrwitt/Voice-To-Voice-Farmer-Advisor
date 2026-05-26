from __future__ import annotations

import os
from typing import Any


USER_REQUESTED_ESCALATION_AM = (
    "እሺ፣ ጥያቄዎን ለግብርና ባለሙያ አስተላልፌዋለሁ። በቅርቡ መልስ ያገኛሉ።"
)

OUT_OF_DOMAIN_ESCALATION_AM = (
    "ይህ ጥያቄ ከግብርና ምክር ወሰን ውጭ ሊሆን ይችላል። "
    "እንዳልሳሳት ጥያቄዎን ለባለሙያ አስተላልፌዋለሁ፤ በቅርቡ መልስ ያገኛሉ።"
)

USER_ESCALATION_PHRASES = (
    "expert",
    "human",
    "agent",
    "helpdesk",
    "operator",
    "development agent",
    "da",
    "escalate",
    "handoff",
    "ለባለሙያ",
    "ባለሙያ",
    "ወደ ባለሙያ",
    "ኤክስፐርት",
    "ሰው ያገናኙኝ",
    "ሰው አገናኝ",
    "ሰው",
    "ያስተላልፉ",
    "አስተላልፍ",
    "እርዳታ",
)

AGRICULTURE_DOMAIN_SIGNALS = (
    "ሰብል",
    "ግብርና",
    "እርሻ",
    "ማዳበሪያ",
    "ዘር",
    "መሬት",
    "አፈር",
    "ተባይ",
    "በሽታ",
    "አረም",
    "ምርት",
    "መስኖ",
    "ውሃ",
    "ዝናብ",
    "ገበያ",
    "ዋጋ",
    "fertilizer",
    "crop",
    "farm",
    "farming",
    "soil",
    "seed",
    "pest",
    "disease",
    "irrigation",
    "harvest",
    "market",
    "weather",
)


def user_requested_escalation(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    return any(phrase in q for phrase in USER_ESCALATION_PHRASES)


def is_out_of_domain(text: str, nlu: Any) -> bool:
    """Conservative out-of-domain detector used before RAG retrieval."""
    if user_requested_escalation(text):
        return False
    if os.getenv("RAG_ESCALATE_OUT_OF_DOMAIN", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False

    q = (text or "").strip().lower()
    if not q:
        return False
    if any(signal in q for signal in AGRICULTURE_DOMAIN_SIGNALS):
        return False

    intent = getattr(nlu, "primary_intent", "unknown")
    confidence = float(getattr(nlu, "confidence", 0.0) or 0.0)
    threshold = float(os.getenv("RAG_OUT_OF_DOMAIN_NLU_CONFIDENCE", "0.35") or "0.35")
    return intent == "unknown" and confidence <= threshold
