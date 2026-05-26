import os
import re


VAD_TTS_SINGLE_CHUNK_MAX_CHARS = int(os.getenv("VAD_TTS_SINGLE_CHUNK_MAX_CHARS", "180"))


def _normalize_confirmation_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    normalized = normalized.replace("ኣ", "አ").replace("ዐ", "አ").replace("ዓ", "አ")
    normalized = re.sub(r"[።፣፤፦፧!?.,:;\"'`]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def classify_confirmation_reply(text: str) -> str:
    normalized = _normalize_confirmation_text(text)
    if not normalized:
        return "unknown"

    yes_terms = (
        "አዎ",
        "አወ",
        "አው",
        "አዋ",
        "አዌ",
        "አውነት",
        "እሺ",
        "እሽ",
        "ልክ",
        "ትክክል",
        "ትክክል ነው",
        "እውነት",
        "awo",
        "aw",
        "awe",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "correct",
        "right",
    )
    no_terms = (
        "አይ",
        "አይደለም",
        "አይደለ",
        "አይ አይደለም",
        "አይደል",
        "አይዎ",
        "የለም",
        "aye",
        "no",
        "nope",
        "wrong",
        "incorrect",
    )

    tokens = set(normalized.split())
    if any(term == normalized or term in tokens or term in normalized for term in no_terms):
        return "no"
    if any(term == normalized or term in tokens or term in normalized for term in yes_terms):
        return "yes"

    return "unknown"


def classify_confirmation_reply_from_asr(asr_result: dict) -> str:
    """Use all ASR transcript variants because short yes/no words are often unstable."""
    candidates = [
        asr_result.get("transcript"),
        asr_result.get("final_transcript"),
        asr_result.get("structured_transcript"),
        asr_result.get("domain_corrected_transcript"),
        asr_result.get("cleaned_transcript"),
        asr_result.get("raw_transcript"),
        asr_result.get("text"),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = _normalize_confirmation_text(str(candidate))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        classification = classify_confirmation_reply(normalized)
        if classification != "unknown":
            return classification
    return "unknown"


def should_synthesize_as_single_chunk(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return False
    if 0 < VAD_TTS_SINGLE_CHUNK_MAX_CHARS >= len(normalized):
        return True
    clarity_markers = (
        "አዎ ወይም አይ",
        "እባክዎን ሰብሉን",
        "ተጨማሪ መረጃ",
        "እንደገና ይናገሩ",
        "አልተረዳሁም",
    )
    return any(marker in normalized for marker in clarity_markers)
