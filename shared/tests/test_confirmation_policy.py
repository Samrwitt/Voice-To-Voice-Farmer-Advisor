import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.confirmation_policy import (
    best_transcript_from_asr,
    has_agriculture_signals,
    vad_flow_decision,
)


def test_best_transcript_prefers_structured_field():
    meta = {
        "raw_transcript": "የአስ ቫር አ ሲዳን",
        "structured_transcript": "የአፈር አሲዳማነት ምልክት በምን ይታወቃል",
        "transcript": "partial",
    }
    assert "አሲዳማነት" in best_transcript_from_asr(meta)


def test_short_non_ag_utterance_reprompts_instead_of_confirm():
    decision = vad_flow_decision(
        {
            "raw_transcript": "ት እንዴሬ መጪ",
            "transcript": "ት እንዴሬ መጪ",
            "confidence": 0.58,
            "unusual_words": ["እንዴሬ", "መጪ"],
        }
    )
    assert decision == "reprompt"


def test_low_confidence_ag_question_confirms():
    decision = vad_flow_decision(
        {
            "raw_transcript": "የአስ ቫር አ ሲዳን",
            "structured_transcript": "የአፈር አሲዳማነት ምልክት በምን ይታወቃል",
            "transcript": "የአፈር አሲዳማነት ምልክት በምን ይታወቃል",
            "confidence": 0.55,
            "unusual_words": [],
        }
    )
    assert decision == "confirm"
    assert has_agriculture_signals("የአፈር አሲዳማነት")


def test_greeting_proceeds_without_confirm():
    assert vad_flow_decision({"transcript": "selam", "confidence": 0.2}) == "proceed"
