import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_utils import get_target_sample_rate, get_tts_atempo, get_tts_volume


def test_tts_runtime_config_is_telephony_safe():
    assert get_target_sample_rate() > 0
    assert 0.5 <= get_tts_atempo() <= 2.0
    assert 0.1 <= get_tts_volume() <= 3.0
