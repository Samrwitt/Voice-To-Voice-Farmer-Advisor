import pytest
import json
from unittest.mock import Mock, patch

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "phone-gateway"}

def test_calculate_audio_level():
    from main import calculate_pcm16_audio_level
    
    # Silence (all zeros)
    silence = b"\x00" * 100
    assert calculate_pcm16_audio_level(silence) == 0.0
    
    # Empty bytes
    assert calculate_pcm16_audio_level(b"") == 0.0
    
    # Some data (pseudo-random bytes representing PCM16)
    data = b"\x01\x00\xff\xff" * 10
    level = calculate_pcm16_audio_level(data)
    assert 0.0 <= level <= 1.0

def test_dedupe_asr_transcripts():
    from main import dedupe_asr_transcripts
    
    transcripts = [
        {"utterance_path": "a.wav", "transcript": "hello"},
        {"utterance_path": "a.wav", "transcript": "hello"}, # Duplicate
        {"utterance_path": "b.wav", "transcript": "world"},
    ]
    
    unique = dedupe_asr_transcripts(transcripts)
    assert len(unique) == 2
    assert unique[0]["utterance_path"] == "a.wav"
    assert unique[1]["utterance_path"] == "b.wav"

def test_build_asr_transcripts_from_active_call():
    from main import build_asr_transcripts_from_active_call
    
    active_call = {
        "session_id": "test_s",
        "utterances": [
            {
                "utterance_path": "u1.wav",
                "transcript": "hello",
                "confidence": 0.9,
                "engine": "whisper",
                "audio_id": "a1",
                "created_at": "2024-01-01"
            },
            {
                "utterance_path": "u2.wav",
                "transcript": None # Should be skipped
            }
        ]
    }
    
    transcripts = build_asr_transcripts_from_active_call(active_call)
    assert len(transcripts) == 1
    assert transcripts[0]["transcript"] == "hello"
    assert transcripts[0]["session_id"] == "test_s"

@patch("main.create_or_get_caller")
def test_register_caller(mock_create, client):
    mock_create.return_value = {
        "caller_id": "c1",
        "full_name": "Test User",
        "phone_number": "123456"
    }
    
    response = client.post("/api/callers/register", json={
        "full_name": "Test User",
        "phone_number": "123456"
    })
    
    assert response.status_code == 200
    assert response.json()["caller"]["caller_id"] == "c1"

def test_get_monitor_state_empty(client):
    # Testing /api/monitor/state when no calls exist
    response = client.get("/api/monitor/state")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert "asr_transcripts" in data
    assert "recent_calls" in data
