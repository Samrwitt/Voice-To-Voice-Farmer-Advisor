import pytest
from unittest.mock import patch, MagicMock
import os

def test_synthesize_empty_text(client):
    response = client.post("/synthesize", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text must not be empty."

@patch("main.gTTS")
@patch("main.subprocess.run")
def test_synthesize_success(mock_run, mock_gtts, client):
    # Setup mocks
    mock_tts_instance = MagicMock()
    mock_gtts.return_value = mock_tts_instance
    
    # Simulate ffmpeg creating the output file
    def side_effect(args, **kwargs):
        # The last argument is the wav_path
        wav_path = args[-1]
        with open(wav_path, "wb") as f:
            f.write(b"fake wav content")
        return MagicMock(returncode=0)
    
    mock_run.side_effect = side_effect
    
    response = client.post("/synthesize", json={"text": "ሰላም"})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"fake wav content"
    
    # Verify calls
    mock_gtts.assert_called_once_with(text="ሰላም", lang="am")
    mock_tts_instance.save.assert_called_once()
    mock_run.assert_called_once()

@patch("main.gTTS")
def test_synthesize_failure(mock_gtts, client):
    mock_gtts.side_effect = Exception("Google API Error")
    
    response = client.post("/synthesize", json={"text": "error test"})
    assert response.status_code == 500
    assert response.json()["detail"] == "TTS generation failed."
