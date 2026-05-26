import re
import os
import requests
from rapidfuzz import process, fuzz

from confirmation import build_confirmation_prompt, needs_confirmation
from domain_terms import get_asr_vocabulary
from config import USE_HOSTED_LLM_FIX, USE_OLLAMA, OLLAMA_URL, OLLAMA_MODEL
# Hosted Groq/Gemini correction is disabled for now to avoid ASR token usage.
# from hosted_llm_fix import hosted_fix_enabled, semantic_correction_hosted



def clean_repetitions(text: str, max_repeat: int = 2) -> str:
    words = text.split()
    cleaned = []

    last_word = None
    repeat_count = 0

    for word in words:
        if word == last_word:
            repeat_count += 1
        else:
            repeat_count = 1
            last_word = word

        if repeat_count <= max_repeat:
            cleaned.append(word)

    return " ".join(cleaned)


def basic_asr_cleanup(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.replace("�", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = clean_repetitions(text, max_repeat=2)
    return text


AMHARIC_HOMOPHONE_MAP = {
    "ሐ": "ሀ", "ኀ": "ሀ", "ሃ": "ሀ", "ሓ": "ሀ", "ኃ": "ሀ", "ኻ": "ሀ",
    "ሑ": "ሁ", "ኁ": "ሁ", "ኹ": "ሁ",
    "ሒ": "ሂ", "ኂ": "ሂ", "ኺ": "ሂ",
    "ሔ": "ሄ", "ኄ": "ሄ", "ኼ": "ሄ",
    "ሕ": "ህ", "ኅ": "ህ", "ኽ": "ህ",
    "ሖ": "ሆ", "ኆ": "ሆ", "ኾ": "ሆ",

    "ኣ": "አ", "ዐ": "አ", "ዓ": "አ",
    "ዑ": "ኡ",
    "ዒ": "ኢ",
    "ዔ": "ኤ",
    "ዕ": "እ",
    "ዖ": "ኦ",

    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",

    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}


def normalize_amharic_homophones(text: str) -> str:
    text = basic_asr_cleanup(text)
    text = "".join(AMHARIC_HOMOPHONE_MAP.get(ch, ch) for ch in text)
    return re.sub(r"\s+", " ", text).strip()


LABIALIZED_TO_FULL_MAP = {
    "ኋ": "ሁአ",
    "ሏ": "ሉአ",
    "ሟ": "ሙአ",
    "ሯ": "ሩአ",
    "ሷ": "ሱአ",
    "ሿ": "ሹአ",
    "ቋ": "ቁአ",
    "ቧ": "ቡአ",
    "ቯ": "ቩአ",
    "ቷ": "ቱአ",
    "ቿ": "ቹአ",
    "ኗ": "ኑአ",
    "ኟ": "ኙአ",
    "ኳ": "ኩአ",
    "ዋ": "ውአ",
    "ጓ": "ጉአ",
    "ዟ": "ዙአ",
    "ዧ": "ዡአ",
    "ዷ": "ዱአ",
    "ጇ": "ጁአ",
    "ጧ": "ጡአ",
    "ጯ": "ጩአ",
    "ጿ": "ጹአ",
    "ፏ": "ፉአ",
    "ፗ": "ፑአ",
}

SADIS_WA_TO_FULL_MAP = {
    "ህዋ": "ሁአ",
    "ልዋ": "ሉአ",
    "ምዋ": "ሙአ",
    "ርዋ": "ሩአ",
    "ስዋ": "ሱአ",
    "ሥዋ": "ሱአ",
    "ሽዋ": "ሹአ",
    "ቅዋ": "ቁአ",
    "ብዋ": "ቡአ",
    "ቭዋ": "ቩአ",
    "ትዋ": "ቱአ",
    "ችዋ": "ቹአ",
    "ንዋ": "ኑአ",
    "ኝዋ": "ኙአ",
    "ክዋ": "ኩአ",
    "ግዋ": "ጉአ",
    "ዝዋ": "ዙአ",
    "ዥዋ": "ዡአ",
    "ድዋ": "ዱአ",
    "ጅዋ": "ጁአ",
    "ጥዋ": "ጡአ",
    "ጭዋ": "ጩአ",
    "ጽዋ": "ጹአ",
    "ፅዋ": "ጹአ",
    "ፍዋ": "ፉአ",
    "ፕዋ": "ፑአ",
}

PRONUNCIATION_VARIANT_MAP = {
    **SADIS_WA_TO_FULL_MAP,
    **LABIALIZED_TO_FULL_MAP,
}


def normalize_amharic_pronunciation(text: str) -> str:
    text = normalize_amharic_homophones(text)

    for src in sorted(PRONUNCIATION_VARIANT_MAP.keys(), key=len, reverse=True):
        text = text.replace(src, PRONUNCIATION_VARIANT_MAP[src])

    return re.sub(r"\s+", " ", text).strip()


def correct_known_agri_phrases(text: str) -> str:
    """Repair common multi-word ASR splits that single-token fuzzy matching misses."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return ""

    # Example observed from SIP audio:
    # "የአስ ቫር አ ሲዳን ማጅመት ከምኑ ይታወቃል"
    # should be routed as "የአፈር አሲዳማነት ምልክት ከምን ይታወቃል".
    normalized = re.sub(r"\bየ?አስ\s+ቫር\s+አ\s+ሲዳን\b", "የአፈር አሲዳማነት", normalized)
    normalized = re.sub(r"\bየ?አፈር?\s+ራሲ\s+ዳማነት\b", "የአፈር አሲዳማነት", normalized)
    normalized = re.sub(r"\bየ?አሰ\s+ፊዳብ\b", "የአፈር አሲዳማነት", normalized)
    normalized = re.sub(r"\bየ?አስ\s+ቫር\b", "የአፈር", normalized)
    normalized = re.sub(r"\bአ\s+ሲዳን\b|\bአሲዳን\b|\bሲዳን\b", "አሲዳማነት", normalized)
    normalized = re.sub(r"\bማጅመት\b|\bመጅመት\b|\bማጅኘት\b", "ምልክት", normalized)
    normalized = re.sub(r"\bከምኑ\b", "ከምን", normalized)
    normalized = re.sub(r"\bበውን\b", "በምን", normalized)
    normalized = re.sub(r"\bተወቃል\b", "ይታወቃል", normalized)

    compact = normalized.replace(" ", "")
    soilish = any(token in compact for token in ("የአፈር", "አፈር", "አስቫር", "አሰፊዳብ", "ፊዳብ"))
    acidish = any(token in compact for token in ("አሲዳ", "ሲዳ", "ዳማነት", "ራሲዳማነት", "ፊዳብ"))
    questionish = any(token in compact for token in ("ይታወቃል", "ታወቃል", "ምልክት", "በምን", "ከምን"))
    if soilish and acidish:
        if questionish:
            return "የአፈር አሲዳማነት ምልክት በምን ይታወቃል"
        return "የአፈር አሲዳማነት"

    return re.sub(r"\s+", " ", normalized).strip()


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
            # This is intentionally simple for demo.
            # Later you can merge ALFFA lexicon + KB vocabulary.
            unusual.append(word)

    return unusual


def semantic_correction_ollama(text: str) -> str:
    """
    Use Ollama to semantically correct the Amharic transcript.
    This helps with grammar, context, and regional dialects.
    """
    if not text.strip():
        return text

    prompt = (
        "You are an Amharic linguistics expert and agricultural advisor. "
        "The following text is a raw transcription from an Amharic farmer's voice query. "
        "Correct any grammatical errors, spelling mistakes, or nonsense words while preserving the agricultural context. "
        "Return ONLY the corrected Amharic text. No explanations.\n\n"
        f"Raw Text: {text}\n"
        "Corrected Text:"
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        corrected = response.json().get("response", "").strip()
        return corrected if corrected else text
    except Exception as e:
        # Fallback to original text if Ollama fails
        print(f"Ollama correction failed: {e}")
        return text


def _apply_semantic_correction(domain_corrected: str) -> tuple[str, str | None, str | None]:
    """
    Returns ``(final_text, semantic_corrected_or_none, fix_backend)``.
    fix_backend: groq | gemini | ollama | none
    """
    # Hosted Groq/Gemini correction is intentionally commented out for now.
    # use_hosted = USE_HOSTED_LLM_FIX in ("1", "true", "yes", "on") or (
    #     USE_HOSTED_LLM_FIX == "auto" and hosted_fix_enabled()
    # )
    # if use_hosted:
    #     try:
    #         fixed, backend = semantic_correction_hosted(domain_corrected)
    #         if backend and backend != "none" and fixed.strip():
    #             return fixed, fixed, backend
    #     except Exception as exc:
    #         print(f"Hosted ASR fix failed, falling back: {exc}")

    if USE_OLLAMA:
        fixed = semantic_correction_ollama(domain_corrected)
        return fixed, fixed, "ollama"

    return domain_corrected, None, None


def postprocess_asr_transcript(raw_text: str, acoustic_confidence: float | None = None) -> dict:
    cleaned = basic_asr_cleanup(raw_text)
    homophone_normalized = normalize_amharic_homophones(cleaned)
    pronunciation_normalized = normalize_amharic_pronunciation(homophone_normalized)
    phrase_corrected = correct_known_agri_phrases(pronunciation_normalized)
    domain_corrected, fuzzy_meta = correct_domain_terms_with_meta(phrase_corrected)

    final_text, semantic_corrected, fix_backend = _apply_semantic_correction(domain_corrected)

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