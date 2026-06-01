"""
VAD-owned confirmation policy and best-transcript selection.

ASR may still emit ``needs_confirmation`` for direct API clients; the voice
pipeline should use ``vad_flow_decision`` instead.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from shared.farmer_text_normalize import normalize_farmer_query

FlowDecision = Literal["proceed", "confirm", "reprompt"]

CLARIFY_REPROMPT_AM = (
    "ይቅርታ፣ ጥያቄዎን በግልጽ አልሰማሁም። እባክዎን የግብርና ጥያቄዎን በአማርኛ እንደገና ይናገሩ።"
)

_KNOWN_SHORT_REPLIES = {
    "አዎ", "አወ", "እሺ", "እሽ", "አይ", "አይደለም",
    "awo", "aw", "eshi", "ishi", "ok", "okay", "yes", "no",
}

_AGRICULTURE_SIGNALS = (
    "በቆሎ", "ስንዴ", "ገብስ", "በርበሬ", "ሽንኩርት", "ጤፍ",
    "የአፈር", "አፈር", "አሲዳ", "ሲዳ", "ተባይ", "ብጥረት", "ማዳበሪያ",
    "መስኖ", "ውሃ", "ዝናብ", "የምርት", "የዝርያ", "የመዝገብ", "የእርሻ",
    "crop", "maize", "wheat", "teff", "fertilizer", "irrigation", "pest",
    "soil", "seed", "harvest", "farm",
)


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def looks_like_greeting(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False
    greeting_markers = (
        "ሰላም", "ደህና", "እንዴት", "እንደምን", "እንደመ", "እንደመን",
        "selam", "salam", "hello", "hi",
    )
    return any(marker in normalized for marker in greeting_markers)


def has_agriculture_signals(text: str) -> bool:
    normalized = normalize_farmer_query(text)
    if not normalized:
        return False
    compact = normalized.replace(" ", "").lower()
    lowered = normalized.lower()
    return any(
        signal in compact or signal in lowered
        for signal in _AGRICULTURE_SIGNALS
    )


def best_transcript_from_asr(asr_result: dict) -> str:
    """Pick the richest ASR field, then apply shared normalization."""
    candidates = (
        asr_result.get("structured_transcript"),
        asr_result.get("final_transcript"),
        asr_result.get("domain_corrected_transcript"),
        asr_result.get("phrase_corrected_transcript"),
        asr_result.get("transcript"),
        asr_result.get("text"),
        asr_result.get("final"),
        asr_result.get("cleaned_transcript"),
        asr_result.get("raw_transcript"),
    )
    for candidate in candidates:
        if candidate and str(candidate).strip():
            return normalize_farmer_query(str(candidate))
    return normalize_farmer_query(asr_result.get("transcript") or "")


def apply_normalized_transcript_to_asr_result(asr_result: dict) -> dict:
    """Return a copy with transcript fields aligned to shared normalization."""
    normalized = best_transcript_from_asr(asr_result)
    if not normalized:
        return dict(asr_result)
    updated = dict(asr_result)
    for key in (
        "transcript",
        "text",
        "final_transcript",
        "structured_transcript",
        "final",
    ):
        updated[key] = normalized
    updated["vad_normalized_transcript"] = normalized
    return updated


def _normalize_confidence(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0:
        return 0.0
    if score <= 1.0:
        return score
    if score <= 100.0:
        return score / 100.0
    return 1.0


def vad_flow_decision(asr_result: dict) -> FlowDecision:
    """
    Decide whether to proceed to RAG, ask yes/no confirmation, or reprompt.

    Confirmation is reserved for low-confidence utterances that still look like
    farmer questions after normalization.
    """
    raw = (asr_result.get("raw_transcript") or asr_result.get("raw") or "").strip()
    text = best_transcript_from_asr(asr_result)
    if not text.strip():
        return "reprompt"

    if "\ufffd" in raw:
        return "reprompt"

    if looks_like_greeting(text):
        return "proceed"

    normalized = re.sub(r"\s+", " ", text.strip())
    if normalized.lower() in _KNOWN_SHORT_REPLIES:
        return "proceed"

    word_count = _word_count(normalized)
    min_words = int(os.getenv("VAD_CONFIRMATION_MIN_WORDS", "3") or "3")
    ag_signals = has_agriculture_signals(normalized)

    if word_count < min_words and not ag_signals:
        return "reprompt"

    confidence = _normalize_confidence(asr_result.get("confidence"))
    min_confidence = float(
        os.getenv(
            "VAD_CONFIRMATION_MIN_CONFIDENCE",
            os.getenv("ASR_CONFIRMATION_MIN_CONFIDENCE", "0.68"),
        )
        or "0.68"
    )

    unusual = [w for w in (asr_result.get("unusual_words") or []) if len(str(w).strip()) >= 2]
    unusual_ratio = (len(unusual) / max(word_count, 1)) if word_count else 0.0
    ratio_threshold = float(os.getenv("ASR_CONFIRMATION_UNUSUAL_RATIO", "0.55") or "0.55")
    min_unusual = int(os.getenv("ASR_CONFIRMATION_MIN_UNUSUAL_WORDS", "2") or "2")

    low_confidence = confidence is not None and confidence < min_confidence
    high_unusual = word_count and len(unusual) >= min_unusual and unusual_ratio >= ratio_threshold

    if not ag_signals:
        if low_confidence or high_unusual or word_count < min_words:
            return "reprompt"
        return "proceed"

    if low_confidence or high_unusual:
        return "confirm"

    return "proceed"


def vad_confirmation_gate_enabled() -> bool:
    return os.getenv("VAD_ASR_CONFIRMATION_GATE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
