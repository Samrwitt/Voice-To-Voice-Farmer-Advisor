import os
import requests
from rapidfuzz import process, fuzz

from confirmation import (
    build_confirmation_prompt,
    looks_like_greeting,
    needs_confirmation,
)
from domain_terms import get_asr_vocabulary
from shared.farmer_text_normalize import (
    basic_asr_cleanup,
    correct_known_agri_phrases,
    normalize_amharic_homophones,
    normalize_amharic_pronunciation,
    normalize_farmer_query,
)

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
            # Never replace a single ASR token with a multi-word KB phrase.
            if " " in (best_term or "") and " " not in word:
                best_term, score = word, 0.0
            elif score < threshold and len(word) >= 4:
                partial = process.extractOne(word, vocabulary, scorer=fuzz.partial_ratio)
                if partial is not None and partial[1] > score:
                    cand, cand_score, _ = partial
                    if " " not in cand or " " in word:
                        best_term, score = cand, cand_score
            scores.append(float(score))
            if score >= threshold and not (" " in (best_term or "") and " " not in word):
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


def _skip_llm_fix(
    domain_corrected: str,
    *,
    quality_confidence: float,
    confirm: bool,
    fix_threshold: float,
    unusual_words: list[str] | None = None,
    fuzzy_meta: dict | None = None,
) -> bool:
    """Avoid LLM fix on utterances that are already lexically clean (saves latency + truncation risk)."""
    if confirm:
        return False
    # Never skip when confidence is below the fix gate — that path exists to catch
    # “looks fine on paper” transcripts Whisper got wrong.
    if quality_confidence < fix_threshold:
        return False
    words = [w for w in (domain_corrected or "").split() if w]
    wc = len(words)
    if wc <= 6 and quality_confidence >= fix_threshold:
        return True
    if looks_like_greeting(domain_corrected) and quality_confidence >= max(0.85, fix_threshold - 0.03):
        return True
    if wc >= 5 and quality_confidence >= fix_threshold:
        unusual = [w for w in (unusual_words or []) if len(str(w).strip()) >= 2]
        if len(unusual) / max(wc, 1) < float(
            os.getenv("ASR_LLM_FIX_SKIP_MAX_UNUSUAL_RATIO", "0.22") or "0.22"
        ):
            return True
        fuzzy_avg = (fuzzy_meta or {}).get("average_score")
        if fuzzy_avg is not None and float(fuzzy_avg) >= float(
            os.getenv("ASR_LLM_FIX_SKIP_MIN_FUZZY_AVG", "0.90") or "0.90"
        ):
            return True
    return False


def _llm_fix_always_on() -> bool:
    return os.getenv("ASR_LLM_FIX_ALWAYS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _should_trigger_llm_fix_heuristic(
    domain_corrected: str,
    unusual_words: list[str],
    fuzzy_meta: dict,
) -> bool:
    """
    Run hosted correction when quality score looks fine but the transcript is
    still lexically suspicious (common when Whisper is wrong but fuzzy scores
    stay inflated by short tokens).
    """
    words = [w for w in (domain_corrected or "").split() if w]
    wc = len(words)
    if wc < 4:
        return False

    unusual = [w for w in (unusual_words or []) if len(str(w).strip()) >= 2]
    ratio = len(unusual) / max(wc, 1)
    min_ratio = float(os.getenv("ASR_LLM_FIX_TRIGGER_UNUSUAL_RATIO", "0.20") or "0.20")
    if ratio >= min_ratio:
        return True

    fuzzy_avg = fuzzy_meta.get("average_score")
    if fuzzy_avg is not None and wc >= 5:
        max_fuzzy = float(os.getenv("ASR_LLM_FIX_TRIGGER_MAX_FUZZY_AVG", "0.88") or "0.88")
        if float(fuzzy_avg) < max_fuzzy:
            return True

    return False


def _apply_semantic_correction(
    domain_corrected: str,
    *,
    raw_text: str | None = None,
) -> tuple[str, str | None, str | None]:
    """
    Returns ``(final_text, semantic_corrected_or_none, fix_backend)``.
    fix_backend: groq | gemini | none
    """
    mode = os.getenv("ASR_HOSTED_LLM_FIX", "0").strip().lower()
    if mode not in ("1", "true", "yes", "on", "auto"):
        return domain_corrected, None, None
    try:
        from hosted_llm_fix import hosted_fix_enabled, semantic_correction_hosted

        if hosted_fix_enabled():
            fixed, backend = semantic_correction_hosted(
                domain_corrected,
                whisper_raw=raw_text,
            )
            if backend and backend != "none" and fixed.strip():
                return fixed, fixed, backend
    except Exception as exc:
        print(f"Hosted ASR fix failed, falling back: {exc}")

    return domain_corrected, None, None


def postprocess_asr_transcript(raw_text: str, acoustic_confidence: float | None = None) -> dict:
    cleaned = basic_asr_cleanup(raw_text)
    homophone_normalized = normalize_amharic_homophones(cleaned)
    pronunciation_normalized = normalize_amharic_pronunciation(homophone_normalized)
    phrase_corrected = correct_known_agri_phrases(pronunciation_normalized)
    domain_corrected, fuzzy_meta = correct_domain_terms_with_meta(phrase_corrected)

    from confirmation import transcript_quality_confidence

    unusual_words = detect_unusual_words(domain_corrected)
    quality_confidence = transcript_quality_confidence(
        domain_corrected,
        unusual_words=unusual_words,
        fuzzy_average=fuzzy_meta.get("average_score"),
        acoustic_confidence=acoustic_confidence,
    )
    confirm = needs_confirmation(
        raw_text,
        domain_corrected,
        unusual_words=unusual_words,
        confidence=quality_confidence,
    )
    final_text = domain_corrected
    semantic_corrected = None
    fix_backend = None
    # Hosted fix runs when confidence is *below* this threshold (strict gate).
    # Default 0.82 so ~0.72–0.75 “looks OK” transcripts still get corrected.
    fix_threshold = float(os.getenv("ASR_LLM_FIX_CONFIDENCE_THRESHOLD", "0.88") or "0.88")
    run_llm_fix = not _skip_llm_fix(
        domain_corrected,
        quality_confidence=quality_confidence,
        confirm=confirm,
        fix_threshold=fix_threshold,
        unusual_words=unusual_words,
        fuzzy_meta=fuzzy_meta,
    ) and (
        _llm_fix_always_on()
        or confirm
        or quality_confidence < fix_threshold
        or _should_trigger_llm_fix_heuristic(domain_corrected, unusual_words, fuzzy_meta)
    )
    if run_llm_fix:
        draft_words = len([w for w in domain_corrected.split() if w])
        final_text, semantic_corrected, fix_backend = _apply_semantic_correction(
            domain_corrected,
            raw_text=raw_text,
        )
        if fix_backend and fix_backend != "none":
            out_words = len([w for w in (final_text or "").split() if w])
            fuzzy_avg = float(fuzzy_meta.get("average_score") or 0.0)
            # If the draft was already lexically strong, reject severely shortened LLM
            # output. If the draft was noisy, allow partial fixes (still better than raw).
            min_ratio = float(os.getenv("ASR_LLM_FIX_MIN_LENGTH_RATIO", "0.55") or "0.55")
            if fuzzy_avg >= 0.78:
                too_short = draft_words >= 8 and out_words < max(3, int(draft_words * min_ratio))
            else:
                too_short = draft_words >= 12 and out_words < max(3, int(draft_words * 0.35))
            if too_short:
                final_text = domain_corrected
                semantic_corrected = None
                fix_backend = None

    # Re-snap domain terms / phrases on LLM output (LLM fixes semantics; fuzzy
    # vocabulary still aligns farmer terms to KB surface forms).
    if fix_backend and fix_backend != "none" and (final_text or "").strip():
        llm_cleaned = basic_asr_cleanup(final_text)
        llm_hom = normalize_amharic_homophones(llm_cleaned)
        llm_pro = normalize_amharic_pronunciation(llm_hom)
        llm_phrase = correct_known_agri_phrases(llm_pro)
        final_text, fuzzy_meta = correct_domain_terms_with_meta(llm_phrase)

    final_text = normalize_farmer_query(final_text)
    unusual_words = detect_unusual_words(final_text)
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
