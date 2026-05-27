"""Gemini API key pools for ASR: paid (text fix + rare audio fallback) vs optional free backup."""

from __future__ import annotations

import os
import re

_SPLIT_RE = re.compile(r"[,;\n|]+")


def _parse_keys(*env_names: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in env_names:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        for part in _SPLIT_RE.split(raw):
            k = part.strip().strip('"').strip("'")
            if k and k not in seen:
                seen.add(k)
                out.append(k)
    return out


def paid_gemini_keys() -> list[str]:
    return _parse_keys(
        "ASR_GEMINI_API_KEY",
        "ASR_GEMINI_PAID_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_BACKUP",
        "GOOGLE_API_KEY",
        "GENAI_API_KEY",
    )


def free_gemini_keys() -> list[str]:
    return _parse_keys(
        "ASR_LLM_FIX_GEMINI_API_KEYS",
        "FREE_GEMINI_API_KEYS",
        "FREE_GEMINI_API_KEY",
        "GEMINI_API_KEY_FREE",
        "GEMINI_API_KEY_FREE_2",
        "GEMINI_API_KEY_FREE_3",
    )


def gemini_keys_for_asr_fix() -> list[str]:
    """
    Text-only correction uses the paid key by default (cheap vs audio ASR).
    Optional free keys as backup when ASR_LLM_FIX_FREE_FALLBACK=1.
    """
    keys = paid_gemini_keys()
    if os.getenv("ASR_LLM_FIX_FREE_FALLBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        for key in free_gemini_keys():
            if key not in keys:
                keys.append(key)
    return keys


def paid_gemini_keys_for_asr_audio() -> list[str]:
    """Rare full-audio re-transcribe — paid key only."""
    return paid_gemini_keys()


# Backwards-compatible alias
def free_gemini_keys_for_asr_fix() -> list[str]:
    return gemini_keys_for_asr_fix()
