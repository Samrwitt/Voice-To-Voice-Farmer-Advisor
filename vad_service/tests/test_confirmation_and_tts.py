import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_flow import (
    classify_confirmation_reply,
    classify_confirmation_reply_from_asr,
    should_synthesize_as_single_chunk,
)


def test_classify_confirmation_reply_amharic_yes_no():
    assert classify_confirmation_reply("አዎ ትክክል ነው") == "yes"
    assert classify_confirmation_reply("አይ አይደለም") == "no"


def test_classify_confirmation_reply_accepts_asr_yes_variants():
    for text in ("awo", "aw", "አው", "አዋ", "ኣዎ", "እሽ", "ok"):
        assert classify_confirmation_reply(text) == "yes"


def test_classify_confirmation_reply_uses_raw_asr_when_final_is_unclear():
    result = {
        "transcript": "አ ወ",
        "final_transcript": "አው",
        "raw_transcript": "awo",
    }

    assert classify_confirmation_reply_from_asr(result) == "yes"


def test_short_clarity_prompt_is_single_tts_chunk():
    prompt = "የጠየቁት፦ በቆሎ ላይ ተባይ አለ ነው? እባክዎ አዎ ወይም አይ ይበሉ።"

    assert should_synthesize_as_single_chunk(prompt) is True
