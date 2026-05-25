import re
import requests
from rapidfuzz import process, fuzz

from domain_terms import DOMAIN_TERMS
from config import USE_HOSTED_LLM_FIX, USE_OLLAMA, OLLAMA_URL, OLLAMA_MODEL
from hosted_llm_fix import hosted_fix_enabled, semantic_correction_hosted



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


def correct_domain_terms(text: str, threshold: int = 82) -> str:
    words = text.split()
    corrected_words = []

    for word in words:
        match = process.extractOne(word, DOMAIN_TERMS, scorer=fuzz.ratio)
        if match is not None:
            best_term, score, _ = match
            if score >= threshold:
                corrected_words.append(best_term)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def detect_unusual_words(text: str, min_len: int = 3) -> list[str]:
    vocab = set(DOMAIN_TERMS)
    unusual = []

    for word in text.split():
        if len(word) >= min_len and word not in vocab:
            # This is intentionally simple for demo.
            # Later you can merge ALFFA lexicon + KB vocabulary.
            unusual.append(word)

    return unusual


def needs_confirmation(raw_text: str, corrected_text: str) -> bool:
    raw_text = raw_text or ""
    corrected_text = corrected_text or ""

    if not raw_text.strip():
        return True

    if "�" in raw_text:
        return True

    raw_words = raw_text.split()
    corrected_words = corrected_text.split()

    length_diff = abs(len(corrected_words) - len(raw_words)) / max(len(raw_words), 1)

    if length_diff > 0.35:
        return True

    return False


def build_confirmation_prompt(corrected_text: str) -> str:
    return f"የጠየቁት፦ {corrected_text} ነው? እባክዎ አዎ ወይም አይ ይበሉ።"


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
    use_hosted = USE_HOSTED_LLM_FIX in ("1", "true", "yes", "on") or (
        USE_HOSTED_LLM_FIX == "auto" and hosted_fix_enabled()
    )
    if use_hosted:
        try:
            fixed, backend = semantic_correction_hosted(domain_corrected)
            if backend and backend != "none" and fixed.strip():
                return fixed, fixed, backend
        except Exception as exc:
            print(f"Hosted ASR fix failed, falling back: {exc}")

    if USE_OLLAMA:
        fixed = semantic_correction_ollama(domain_corrected)
        return fixed, fixed, "ollama"

    return domain_corrected, None, None


def postprocess_asr_transcript(raw_text: str) -> dict:
    cleaned = basic_asr_cleanup(raw_text)
    homophone_normalized = normalize_amharic_homophones(cleaned)
    pronunciation_normalized = normalize_amharic_pronunciation(homophone_normalized)
    domain_corrected = correct_domain_terms(pronunciation_normalized)

    final_text, semantic_corrected, fix_backend = _apply_semantic_correction(domain_corrected)

    unusual_words = detect_unusual_words(final_text)
    confirm = needs_confirmation(raw_text, final_text)

    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "homophone_normalized": homophone_normalized,
        "pronunciation_normalized": pronunciation_normalized,
        "domain_corrected": domain_corrected,
        "semantic_corrected": semantic_corrected,
        "transcript_fix_backend": fix_backend,
        "final": final_text,
        "transcript": final_text,
        "text": final_text,
        "unusual_words": unusual_words,
        "needs_confirmation": confirm,
        "confirmation_prompt": build_confirmation_prompt(final_text) if confirm else None,
    }