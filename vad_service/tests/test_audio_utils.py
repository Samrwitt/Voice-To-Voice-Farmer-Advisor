import pytest
import numpy as np
from audio_utils import pcm16_bytes_to_float32, save_pcm16_wav
import tempfile
from pathlib import Path
import wave

def test_pcm16_bytes_to_float32():
    # Silence
    pcm_bytes = b"\x00\x00" * 10
    result = pcm16_bytes_to_float32(pcm_bytes)
    assert isinstance(result, np.ndarray)
    assert len(result) == 10
    assert np.all(result == 0.0)
    
    # Max positive value (32767)
    pcm_bytes = b"\xff\x7f" 
    result = pcm16_bytes_to_float32(pcm_bytes)
    assert np.isclose(result[0], 1.0, atol=1e-4)
    
    # Max negative value (-32768)
    pcm_bytes = b"\x00\x80"
    result = pcm16_bytes_to_float32(pcm_bytes)
    assert np.isclose(result[0], -1.0, atol=1e-4)

def test_pcm16_bytes_to_float32_empty():
    assert len(pcm16_bytes_to_float32(b"")) == 0

def test_save_pcm16_wav():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.wav"
        pcm_bytes = b"\x01\x00" * 16000 # 1 second of audio at 16k
        
        saved_path = save_pcm16_wav(file_path, pcm_bytes, 16000)
        
        assert saved_path is not None
        assert Path(saved_path).exists()
        
        # Verify wav header
        with wave.open(saved_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000
