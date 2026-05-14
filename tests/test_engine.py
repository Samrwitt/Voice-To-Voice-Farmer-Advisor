import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "asr_service"))

import pytest

from engine import ASREngine


# ----------------------------
# Test init success
# ----------------------------

@patch("engine.WhisperModel")
def test_engine_init_success(mock_model):
    mock_instance = Mock()
    mock_model.return_value = mock_instance

    with patch("engine.Path.exists", return_value=True):
        engine = ASREngine()

        assert engine.model is not None


# ----------------------------
# Test init failure (no model)
# ----------------------------

def test_engine_init_missing_model():
    with patch("engine.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            ASREngine()


# ----------------------------
# Test transcribe pipeline
# ----------------------------

@patch("engine.postprocess_asr_transcript")
@patch("engine.WhisperModel")
def test_transcribe(mock_model, mock_postprocess):
    # ---------------- mock whisper ----------------
    mock_instance = Mock()
    mock_model.return_value = mock_instance

    fake_segment = Mock()
    fake_segment.text = "hello world"
    fake_segment.start = 0.0
    fake_segment.end = 1.0

    mock_instance.transcribe.return_value = (
        [fake_segment],
        Mock(language="en", language_probability=0.95),
    )

    # ---------------- mock postprocess ----------------
    mock_postprocess.return_value = {
        "raw": "hello world",
        "cleaned": "hello world",
        "homophone_normalized": "hello world",
        "pronunciation_normalized": "hello world",
        "domain_corrected": "hello world",
        "final": "hello world",
        "unusual_words": [],
        "needs_confirmation": False,
        "confirmation_prompt": None,
    }

    with patch("engine.Path.exists", return_value=True):
        engine = ASREngine()

        result = engine.transcribe("dummy.wav")

        assert result["final_transcript"] == "hello world"
        assert "segments" in result
        assert result["engine"] == "whisper_local"


# ----------------------------
# Test segment formatting
# ----------------------------

@patch("engine.postprocess_asr_transcript")
@patch("engine.WhisperModel")
def test_segments_format(mock_model, mock_postprocess):
    mock_instance = Mock()
    mock_model.return_value = mock_instance

    fake_segment = Mock()
    fake_segment.text = "hi"
    fake_segment.start = 0.0
    fake_segment.end = 2.0

    mock_instance.transcribe.return_value = (
        [fake_segment],
        Mock(language="en", language_probability=0.8),
    )

    mock_postprocess.return_value = {
        "raw": "hi",
        "cleaned": "hi",
        "homophone_normalized": "hi",
        "pronunciation_normalized": "hi",
        "domain_corrected": "hi",
        "final": "hi",
        "unusual_words": [],
        "needs_confirmation": False,
        "confirmation_prompt": None,
    }

    with patch("engine.Path.exists", return_value=True):
        engine = ASREngine()
        result = engine.transcribe("dummy.wav")

        assert isinstance(result["segments"], list)
        assert result["segments"][0]["text"] == "hi"