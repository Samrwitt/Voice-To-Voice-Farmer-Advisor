import os
import requests
from rapidfuzz import process, fuzz

from confirmation import build_confirmation_prompt, needs_confirmation
from domain_terms import get_asr_vocabulary
from shared.farmer_text_normalize import (
    basic_asr_cleanup,
    correct_known_agri_phrases,
    normalize_amharic_homophones,
    normalize_amharic_pronunciation,
    normalize_farmer_query,
)
# from config import USE_HOSTED_LLM_FIX
# Hosted Groq/Gemini correction is disabled for now to avoid ASR token usage.
# from hosted_llm_fix import hosted_fix_enabled, semantic_correction_hosted


def correct_domain_terms_with_meta(text: str, threshold: int | None = None) -> tuple[str, dict]:
    if threshold is None:
        threshold = int(os.getenv("ASR_DOMAIN_TERM_FUZZY_THRESHOLD", "84") or "84")
    vocabulary = get_asr_vocabulary()
    words = text.split()
    corrected_words = []
    matches = []
    scores = []

    for word in words:
        if word in vocabulary:
            corrected_words.append(word)
            scores.append(100.0)
            matches.append(
                {
                    "word": word,
                    "matched": word,
                    "score": 100.0,
                    "status": "exact",
                }
            )
            continue

        match = process.extractOne(word, vocabulary, scorer=fuzz.ratio)
        if match is not None:
            best_term, score, _ = match
            scores.append(float(score))
            if score >= threshold:
                corrected_words.append(best_term)
                status = "corrected"
            else:
                corrected_words.append(word)
                status = "uncertain"
            matches.append(
                {
                    "word": word,
                    "matched": best_term,
                    "score": round(float(score), 1),
                    "status": status,
                }
            )
        else:
            corrected_words.append(word)
            scores.append(0.0)
            matches.append(
                {
                    "word": word,
                    "matched": None,
                    "score": 0.0,
                    "status": "unmatched",
                }
            )

    fuzzy_average = (sum(scores) / len(scores) / 100.0) if scores else 0.0
    return " ".join(corrected_words), {
        "threshold": threshold,
        "average_score": round(fuzzy_average, 3),
        "matches": matches,
    }


def correct_domain_terms(text: str, threshold: int | None = None) -> str:
    corrected, _ = correct_domain_terms_with_meta(text, threshold=threshold)
    return corrected


def detect_unusual_words(text: str, min_len: int = 3) -> list[str]:
    vocab = set(get_asr_vocabulary())
    unusual = []

    for word in text.split():
        if len(word) >= min_len and word not in vocab:
            unusual.append(word)

    return unusual


def _apply_semantic_correction(domain_corrected: str) -> tuple[str, str | None, str | None]:
    """
    Returns ``(final_text, semantic_corrected_or_none, fix_backend)``.
    fix_backend: groq | gemini | none
    """
    return domain_corrected, None, None


def postprocess_asr_transcript(raw_text: str, acoustic_confidence: float | None = None) -> dict:
    cleaned = basic_asr_cleanup(raw_text)
    homophone_normalized = normalize_amharic_homophones(cleaned)
    pronunciation_normalized = normalize_amharic_pronunciation(homophone_normalized)
    phrase_corrected = correct_known_agri_phrases(pronunciation_normalized)
    domain_corrected, fuzzy_meta = correct_domain_terms_with_meta(phrase_corrected)

    final_text, semantic_corrected, fix_backend = _apply_semantic_correction(domain_corrected)
    final_text = normalize_farmer_query(final_text)

    unusual_words = detect_unusual_words(final_text)
    from confirmation import transcript_quality_confidence

    quality_confidence = transcript_quality_confidence(
        final_text,
        unusual_words=unusual_words,
        fuzzy_average=fuzzy_meta.get("average_score"),
        acoustic_confidence=acoustic_confidence,
    )
    confirm = needs_confirmation(
        raw_text,
        final_text,
        unusual_words=unusual_words,
        confidence=quality_confidence,
    )

    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "homophone_normalized": homophone_normalized,
        "pronunciation_normalized": pronunciation_normalized,
        "phrase_corrected": phrase_corrected,
        "domain_corrected": domain_corrected,
        "semantic_corrected": semantic_corrected,
        "transcript_fix_backend": fix_backend,
        "final": final_text,
        "transcript": final_text,
        "text": final_text,
        "structured_transcript": final_text,
        "fuzzy": fuzzy_meta,
        "confidence": quality_confidence,
        "unusual_words": unusual_words,
        "needs_confirmation": confirm,
        "confirmation_prompt": build_confirmation_prompt(final_text) if confirm else None,
    }
