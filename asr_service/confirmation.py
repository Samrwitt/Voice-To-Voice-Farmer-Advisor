import os
import re


_KNOWN_SHORT_REPLIES = {
    "አዎ",
    "አወ",
    "እሺ",
    "እሽ",
    "አይ",
    "አይደለም",
    "awo",
    "aw",
    "eshi",
    "ishi",
    "ok",
    "okay",
    "yes",
    "no",
}


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def normalize_confidence_score(value: float | int | str | None) -> float | None:
    """Normalize external ASR metadata into 0..1 without trusting its scale."""
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


def transcript_quality_confidence(
    text: str,
    *,
    unusual_words: list[str] | None = None,
    fuzzy_average: float | None = None,
    acoustic_confidence: float | int | str | None = None,
) -> float:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return 0.0
    if looks_like_greeting(normalized) or normalized.lower() in _KNOWN_SHORT_REPLIES:
        return 0.92

    word_count = max(_word_count(normalized), 1)
    unusual = [w for w in (unusual_words or []) if len(str(w).strip()) >= 2]
    lexical_coverage = max(0.0, min(1.0, 1.0 - (len(unusual) / word_count)))
    fuzzy_score = max(0.0, min(1.0, fuzzy_average if fuzzy_average is not None else lexical_coverage))
    acoustic_score = normalize_confidence_score(acoustic_confidence)

    if acoustic_score is None:
        quality = (0.6 * fuzzy_score) + (0.4 * lexical_coverage)
    else:
        quality = (0.5 * fuzzy_score) + (0.35 * lexical_coverage) + (0.15 * acoustic_score)

    min_words = int(os.getenv("ASR_CONFIRMATION_MIN_WORDS", "3") or "3")
    if word_count < min_words:
        quality = min(quality, 0.49)

    return round(max(0.0, min(1.0, quality)), 3)


def looks_like_greeting(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False
    greeting_markers = (
        "ሰላም",
        "ደህና",
        "እንዴት",
        "እንደምን",
        # Common ASR variants for "እንደምን/እንዴት" greetings.
        "እንደመ",
        "እንደመን",
        "selam",
        "salam",
        "hello",
        "hi",
    )
    return any(marker in normalized for marker in greeting_markers)


def needs_confirmation(
    raw_text: str,
    corrected_text: str,
    *,
    unusual_words: list[str] | None = None,
    confidence: float | None = None,
) -> bool:
    raw_text = raw_text or ""
    corrected_text = corrected_text or ""

    if not raw_text.strip():
        return True

    if "�" in raw_text:
        return True

    if looks_like_greeting(corrected_text):
        return False

    normalized = re.sub(r"\s+", " ", corrected_text.strip())
    if normalized.lower() in _KNOWN_SHORT_REPLIES:
        return False

    word_count = _word_count(normalized)
    min_words = int(os.getenv("ASR_CONFIRMATION_MIN_WORDS", "3") or "3")
    if 0 < word_count < min_words:
        return True

    min_confidence = float(os.getenv("ASR_CONFIRMATION_MIN_CONFIDENCE", "0.68") or "0.68")
    if confidence is not None and confidence < min_confidence:
        return True

    raw_words = raw_text.split()
    corrected_words = corrected_text.split()

    length_diff = abs(len(corrected_words) - len(raw_words)) / max(len(raw_words), 1)

    if length_diff > 0.35:
        return True

    unusual = [w for w in (unusual_words or []) if len(str(w).strip()) >= 2]
    ratio_threshold = float(os.getenv("ASR_CONFIRMATION_UNUSUAL_RATIO", "0.55") or "0.55")
    min_unusual = int(os.getenv("ASR_CONFIRMATION_MIN_UNUSUAL_WORDS", "2") or "2")
    if word_count and len(unusual) >= min_unusual:
        unusual_ratio = len(unusual) / max(word_count, 1)
        if unusual_ratio >= ratio_threshold:
            return True

    return False


def build_confirmation_prompt(corrected_text: str) -> str:
    assumed = re.sub(r"\s+", " ", (corrected_text or "").strip())
    if not assumed:
        return "የሰማሁትን በትክክል አላረጋገጥኩም። እባክዎ ጥያቄዎን እንደገና ይናገሩ።"
    return f"የሰማሁት ይህ ነው፦ {assumed}። ትክክል ነው? እባክዎ አዎ ወይም አይ ይበሉ።"
