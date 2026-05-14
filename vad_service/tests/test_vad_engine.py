import pytest
from unittest.mock import MagicMock, patch
from vad_engine import SileroStreamingVAD

def test_vad_engine_init():
    vad = SileroStreamingVAD(session_id="test_s", output_dir="test_utterances")
    assert vad.session_id == "test_s"
    assert vad.is_speaking is False
    assert len(vad.pending_bytes) == 0

def test_vad_engine_process_silence(mock_vad_model):
    # Mock model returning 0.1 probability (silence)
    mock_vad_model.return_value.item.return_value = 0.1
    
    vad = SileroStreamingVAD(session_id="test_s")
    
    # Send 1024 bytes (2 windows of 512 samples)
    pcm_data = b"\x00" * 1024
    events = vad.process_pcm_chunk(pcm_data)
    
    assert len(events) == 0
    assert vad.is_speaking is False

def test_vad_engine_speech_detection(mock_vad_model):
    # Mock model returning 0.9 probability (speech)
    mock_vad_model.return_value.item.return_value = 0.9
    
    # Set min_speech_start_ms to 0 for immediate detection in test
    vad = SileroStreamingVAD(session_id="test_s", min_speech_start_ms=0)
    
    # Send 1 window (512 samples = 32ms)
    pcm_data = b"\x01\x00" * 512
    events = vad.process_pcm_chunk(pcm_data)
    
    assert len(events) == 1
    assert events[0]["event"] == "speech_started"
    assert vad.is_speaking is True

def test_vad_engine_speech_ended(mock_vad_model):
    vad = SileroStreamingVAD(
        session_id="test_s", 
        min_speech_start_ms=0,
        speech_end_silence_ms=32 # 1 window
    )
    
    # 1. Start speech
    mock_vad_model.return_value.item.return_value = 0.9
    vad.process_pcm_chunk(b"\x01\x00" * 512)
    assert vad.is_speaking is True
    
    # 2. End speech (silence)
    mock_vad_model.return_value.item.return_value = 0.1
    with patch("vad_engine.save_pcm16_wav", return_value="test.wav"):
        events = vad.process_pcm_chunk(b"\x00\x00" * 512)
        
    assert len(events) == 1
    assert events[0]["event"] == "speech_ended"
    assert events[0]["utterance_path"] == "test.wav"
    assert vad.is_speaking is False

def test_vad_engine_reset():
    vad = SileroStreamingVAD(session_id="test_s")
    vad.is_speaking = True
    vad.pending_bytes.extend(b"123")
    
    vad.reset()
    
    assert vad.is_speaking is False
    assert len(vad.pending_bytes) == 0
