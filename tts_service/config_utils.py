import os


def get_target_sample_rate() -> int:
    return int(os.getenv("TTS_SAMPLE_RATE", "16000"))


def get_tts_atempo() -> float:
    return min(2.0, max(0.5, float(os.getenv("TTS_ATEMPO", "1.35"))))


def get_tts_volume() -> float:
    return min(3.0, max(0.1, float(os.getenv("TTS_VOLUME", "1.5"))))
