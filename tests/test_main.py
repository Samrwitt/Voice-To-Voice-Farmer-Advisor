import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add asr_service to path so we can import main
sys.path.append(str(Path(__file__).resolve().parents[1] / "asr_service"))

from main import app

# Initialize the TestClient
client = TestClient(app)

# -------------------------
# Mock ASR response data
# -------------------------

fake_response = {
    "language": "en",
    "language_probability": 0.9,
    "raw_transcript": "hello world",
    "cleaned_transcript": "hello world",
    "homophone_normalized_transcript": "hello world",
    "pronunciation_normalized_transcript": "hello world",
    "domain_corrected_transcript": "hello world",
    "final_transcript": "hello world",
    "transcript": "hello world",
    "text": "hello world",
    "confidence": 0.9,
    "engine": "whisper_local",
    "audio_id": "test",
    "unusual_words": [],
    "needs_confirmation": False,
    "confirmation_prompt": None,
    "segments": [],
    "latency_seconds": 0.1,
}


# -------------------------
# Global Mocks Fixture
# -------------------------

@pytest.fixture(autouse=True)
def mock_asr_components():
    """
    Mock all components that interact with the filesystem or heavy models.
    """
    with patch("main.asr_engine") as mock_engine, \
         patch("main.save_upload_file") as mock_save, \
         patch("main.prepare_audio_for_asr") as mock_prepare:
        
        # Configure the transcription mock
        mock_engine.transcribe.return_value = fake_response
        
        # Configure file processing mocks to return dummy paths
        mock_save.return_value = Path("fake_audio.wav")
        mock_prepare.return_value = Path("fake_prepared.wav")
        
        yield {
            "engine": mock_engine,
            "save": mock_save,
            "prepare": mock_prepare
        }


# -------------------------
# Tests
# -------------------------

def test_health():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_transcribe_upload():
    """Test uploading an audio file for transcription."""
    file_content = b"fake audio data"

    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", file_content, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_transcript"] == "hello world"
    assert data["audio_id"] == "test"


def test_transcribe_file():
    """Test transcribing a file that already exists in the shared volume."""
    # Mock Path.exists to simulate that the file exists in the shared volume
    with patch("pathlib.Path.exists", return_value=True):
        response = client.post(
            "/transcribe-file",
            json={"filename": "shared_test.wav", "language": "am"},
        )

        assert response.status_code == 200
        assert response.json()["final_transcript"] == "hello world"
