"""
High-stakes agrochemical topics: when the KB has no grounding chunks, do not
invent doses or spray advice from the LLM or dynamic context alone.
"""

from __future__ import annotations

import os
import re

# Default off so local runs without KB are not blocked; enable in prod (see docker-compose).
def agrochemical_expert_only_enabled() -> bool:
    raw = os.getenv("RAG_AGROCHEM_EXPERT_ONLY", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


_AGROCHEM_PATTERN = re.compile(
    r"(?:"
    r"pesticide|pesticides|herbicide|insecticide|fungicide|nematicide|"
    r"fertilizer|fertiliser|fertilizers|npk|urea|glyphosate|paraquat|"
    r"spray|spraying|application\s+rate|mixing\s+ratio|dose|dosage|ppm|"
    r"ማዳበሪያ|ፀረ[\s-]?ተባይ|ፀረ\s+ተባይ|መርጨት|መርጨ|ርጭት|የመድሐኒት|መድሐኒት|"
    r"መጠን|ምን\s*ያህል|እንዴት\s*መርጨት"
    r")",
    re.IGNORECASE,
)


def is_high_risk_agrochemical_query(text: str) -> bool:
    if not (text or "").strip():
        return False
    return bool(_AGROCHEM_PATTERN.search(text))


# Spoken / text reply: clear deferral to a human agronomist; no dosing hints.
CANNED_AGROCHEM_ESCALATION_AM = (
    "ለመድሐኒት መጠን፣ የመርጨት ጊዜ፣ መገናኛና የደህንነት ጥያቄዎች ከዚህ ስልክ አጠቃላይ መረጃ ብቻ ሙሉ መልስ ልንሰጥ አንችልም። "
    "እባክዎን የገበር ልማት ባለሙያ፣ ኮኦፕ ወይም የአካባቢ ተቋም ይጠይቁ። ጥያቄዎን ለባለሙያ አስተላልፈናል።"
)
