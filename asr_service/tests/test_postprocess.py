import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confirmation import (
    build_confirmation_prompt,
    looks_like_greeting,
    needs_confirmation,
    normalize_confidence_score,
    transcript_quality_confidence,
)
from postprocess import postprocess_asr_transcript


def test_needs_confirmation_for_replacement_character():
    assert needs_confirmation("በቆሎ �", "በቆሎ") is True


def test_confirmation_prompt_asks_yes_or_no():
    prompt = build_confirmation_prompt("በቆሎ ላይ ተባይ አለ")

    assert "በቆሎ ላይ ተባይ አለ" in prompt
    assert "አዎ" in prompt
    assert "አይ" in prompt


def test_needs_confirmation_for_short_unclear_transcript():
    assert needs_confirmation("ቢ ነው", "ቢ ነው", unusual_words=["ቢ"]) is True


def test_greeting_like_asr_variant_does_not_force_confirmation():
    text = "ም እንደመነች"

    assert looks_like_greeting(text) is True
    assert needs_confirmation(text, text, unusual_words=["እንደመነች"]) is False


def test_latin_greeting_and_short_reply_do_not_force_confirmation():
    assert looks_like_greeting("selam") is True
    assert needs_confirmation("selam", "selam", confidence=0.2) is False
    assert needs_confirmation("eshi", "eshi", confidence=0.2) is False
    assert transcript_quality_confidence("eshi", acoustic_confidence=0.1) == 0.92


def test_external_confidence_is_normalized_before_use():
    assert normalize_confidence_score(2.8) == pytest.approx(0.028)
    assert normalize_confidence_score(0.82) == 0.82


def test_transcript_quality_confidence_penalizes_uncertain_text():
    confidence = transcript_quality_confidence(
        "ቢ ነው ምኖረወጡ",
        unusual_words=["ቢ", "ምኖረወጡ"],
        fuzzy_average=0.32,
        acoustic_confidence=2.8,
    )

    assert confidence < 0.68
    assert needs_confirmation(
        "ቢ ነው ምኖረወጡ",
        "ቢ ነው ምኖረወጡ",
        unusual_words=["ቢ", "ምኖረወጡ"],
        confidence=confidence,
    ) is True


def test_soil_acidity_asr_phrase_is_structured_before_rag():
    variants = [
        "የአስ ቫር አ ሲዳን ማጅመት ከምኑ ይታወቃል",
        "የአፈ ራሲ ዳማነት በምን ተወቃል",
        "የአፈር ራሲ ዳማነት በምን ተወቃል",
        "የአሰ ፊዳብ ማጅኘት በውን ይታወቃል",
    ]

    for text in variants:
        result = postprocess_asr_transcript(text)

        assert "የአፈር" in result["final"]
        assert "አሲዳማነት" in result["final"]
        assert "ምልክት" in result["final"]
        assert result["needs_confirmation"] is False
