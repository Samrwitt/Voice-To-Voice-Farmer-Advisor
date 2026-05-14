import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "asr_service")
)
import io
import wave
from pathlib import Path

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
# Helper to Create WAV File
# -------------------------

def create_wav_file(path, sr=16000, channels=1, duration=1):
    samples = np.zeros(sr * duration)

    sf.write(path, samples, sr)

    # Convert to stereo if needed
    if channels == 2:
        stereo = np.column_stack((samples, samples))
        sf.write(path, stereo, sr)


# -------------------------
# Test Already Correct WAV
# -------------------------

def test_prepare_audio_already_correct(tmp_path, monkeypatch):
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    wav_path = tmp_path / "correct.wav"

    create_wav_file(wav_path, sr=TARGET_SR, channels=1)

    result = prepare_audio_for_asr(wav_path)

    # Should return original file
    assert result == wav_path


# -------------------------
# Test Resampling Needed
# -------------------------

def test_prepare_audio_resample(tmp_path, monkeypatch):
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    wav_path = tmp_path / "wrong.wav"

    # Create stereo 22050Hz audio
    stereo_audio = np.random.randn(22050, 2)
    sf.write(wav_path, stereo_audio, 22050)

    result = prepare_audio_for_asr(wav_path)

    assert result.exists()
    assert result != wav_path

    # Verify output format
    with wave.open(str(result), "rb") as f:
        assert f.getframerate() == TARGET_SR
        assert f.getnchannels() == 1


# -------------------------
# Test Non-WAV Fallback
# -------------------------

def test_prepare_audio_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("audio_utils.TMP_DIR", tmp_path)

    fake_audio = tmp_path / "audio.mp3"

    # create fake mp3-like audio using soundfile
    samples = np.random.randn(16000)
    sf.write(fake_audio, samples, 16000)

    result = prepare_audio_for_asr(fake_audio)

    assert result.exists()