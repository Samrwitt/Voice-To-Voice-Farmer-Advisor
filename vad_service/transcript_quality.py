"""
Detect unusable ASR output (gibberish / noise) so we do not send junk to RAG.
"""

from __future__ import annotations

import re

_AMH = re.compile(r"[\u1200-\u137F]")

# Greeting read at call start (before ASR).
GREETING_AM = "ጤና ይስጥልን! ግብርና ምክር አገልግሎት ነን ፤ ዛሬ ምን እንርዳዎ ?"


def is_asr_gibberish(transcript: str | None, confidence: float | None = None) -> bool:
    t = (transcript or "").strip()
    if len(t) < 2:
        return True
    # Very low diversity (e.g. repeated symbols)
    if len(t) > 10 and len(set(t)) <= 2:
        return True
    ethiopic = len(_AMH.findall(t))
    ratio = ethiopic / max(len(t), 1)
    if len(t) >= 12 and ratio < 0.12:
        return True
    if confidence is not None and confidence < 0.22 and len(t) < 36:
        return True
    return False


GIBBERISH_REPLY_AM = (
    "ጥያቄዎ ግልፅ አልሆነልኝም። እባክዎ ጥያቄውን ረጋ ብለዉ ይድገሙ።"
)
