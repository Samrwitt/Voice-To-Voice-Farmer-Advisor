"""
Shared Amharic farmer-query normalization for ASR postprocess, VAD, and RAG.

Keeps phrase repairs and character normalization in one place so confirmation
prompts and RAG retrieval see the same text.
"""

from __future__ import annotations

import re


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
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text).strip()
    return clean_repetitions(text, max_repeat=2)


AMHARIC_HOMOPHONE_MAP = {
    "ሐ": "ሀ", "ኀ": "ሀ", "ሃ": "ሀ", "ሓ": "ሀ", "ኃ": "ሀ", "ኻ": "ሀ",
    "ሑ": "ሁ", "ኁ": "ሁ", "ኹ": "ሁ",
    "ሒ": "ሂ", "ኂ": "ሂ", "ኺ": "ሂ",
    "ሔ": "ሄ", "ኄ": "ሄ", "ኼ": "ሄ",
    "ሕ": "ህ", "ኅ": "ህ", "ኽ": "ህ",
    "ሖ": "ሆ", "ኆ": "ሆ", "ኾ": "ሆ",
    "ኣ": "አ", "ዐ": "አ", "ዓ": "አ",
    "ዑ": "ኡ", "ዒ": "ኢ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}

LABIALIZED_TO_FULL_MAP = {
    "ኋ": "ሁአ", "ሏ": "ሉአ", "ሟ": "ሙአ", "ሯ": "ሩአ", "ሷ": "ሱአ", "ሿ": "ሹአ",
    "ቋ": "ቁአ", "ቧ": "ቡአ", "ቯ": "ቩአ", "ቷ": "ቱአ", "ቿ": "ቹአ",
    "ኗ": "ኑአ", "ኟ": "ኙአ", "ኳ": "ኩአ", "ዋ": "ውአ", "ጓ": "ጉአ",
    "ዟ": "ዙአ", "ዧ": "ዡአ", "ዷ": "ዱአ", "ጇ": "ጁአ", "ጧ": "ጡአ",
    "ጯ": "ጩአ", "ጿ": "ጹአ", "ፏ": "ፉአ", "ፗ": "ፑአ",
}

SADIS_WA_TO_FULL_MAP = {
    "ህዋ": "ሁአ", "ልዋ": "ሉአ", "ምዋ": "ሙአ", "ርዋ": "ሩአ", "ስዋ": "ሱአ",
    "ሥዋ": "ሱአ", "ሽዋ": "ሹአ", "ቅዋ": "ቁአ", "ብዋ": "ቡአ", "ቭዋ": "ቩአ",
    "ትዋ": "ቱአ", "ችዋ": "ቹአ", "ንዋ": "ኑአ", "ኝዋ": "ኙአ", "ክዋ": "ኩአ",
    "ግዋ": "ጉአ", "ዝዋ": "ዙአ", "ዥዋ": "ዡአ", "ድዋ": "ዱአ", "ጅዋ": "ጁአ",
    "ጥዋ": "ጡአ", "ጭዋ": "ጩአ", "ጽዋ": "ጹአ", "ፅዋ": "ጹአ", "ፍዋ": "ፉአ",
    "ፕዋ": "ፑአ",
}

PRONUNCIATION_VARIANT_MAP = {**SADIS_WA_TO_FULL_MAP, **LABIALIZED_TO_FULL_MAP}


def normalize_amharic_homophones(text: str) -> str:
    text = basic_asr_cleanup(text)
    text = "".join(AMHARIC_HOMOPHONE_MAP.get(ch, ch) for ch in text)
    return re.sub(r"\s+", " ", text).strip()


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


def normalize_farmer_query(text: str) -> str:
    """Full shared normalization pipeline (no domain-term fuzzy matching)."""
    cleaned = basic_asr_cleanup(text)
    homophone = normalize_amharic_homophones(cleaned)
    pronounced = normalize_amharic_pronunciation(homophone)
    return correct_known_agri_phrases(pronounced)


# Backward-compatible alias used by RAG NLU.
normalize_asr_farmer_query = normalize_farmer_query
