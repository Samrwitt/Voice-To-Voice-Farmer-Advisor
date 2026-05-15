import sys
from pathlib import Path

# Add asr_service to path so we can import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

import io
import wave
import numpy as np
import pytest
import soundfile as sf

from audio_utils import save_upload_file, prepare_audio_for_asr
from config import TARGET_SR


# -------------------------
# Helper Mock Upload File
# -------------------------

class MockUploadFile:
    def __init__(self, filename, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)


# -------------------------
# Test save_upload_file
# -------------------------

def test_save_upload_file(tmp_path, monkeypatch):
    # Mock TMP_DIR
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    content = b"fake audio content"
    upload = MockUploadFile("test.wav", content)

    saved_path = save_upload_file(upload)

    assert saved_path.exists()
    assert saved_path.suffix == ".wav"

    with open(saved_path, "rb") as f:
        assert f.read() == content


# -------------------------
# Test Already Correct WAV
# -------------------------

def test_prepare_audio_already_correct(tmp_path, monkeypatch):
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    wav_path = tmp_path / "correct.wav"
    # Create a 16k mono wav
    with wave.open(str(wav_path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b'\x00' * 3200)

    # Should return original path
    result = prepare_audio_for_asr(wav_path)
    assert result == wav_path


# -------------------------
# Test Conversion
# -------------------------

def test_prepare_audio_needs_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    # Create a 44.1k wav
    src_path = tmp_path / "highrate.wav"
    with wave.open(str(src_path), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b'\x00' * 8820)

    # We can mock librosa or just let it run if dependencies exist
    result = prepare_audio_for_asr(src_path)
    
    assert result != src_path
    assert result.suffix == ".wav"
    assert "_prepared" in result.name
    assert result.exists()
