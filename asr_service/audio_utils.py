import uuid
import wave
from pathlib import Path

import librosa
import soundfile as sf

from config import TMP_DIR, TARGET_SR


def save_upload_file(upload_file) -> Path:
    suffix = Path(upload_file.filename or "audio.wav").suffix or ".wav"
    path = TMP_DIR / f"{uuid.uuid4()}{suffix}"

    with open(path, "wb") as f:
        f.write(upload_file.file.read())

    return path


def prepare_audio_for_asr(input_path: str | Path) -> Path:
    """
    Ensures the audio is 16kHz Mono.
    If already in the correct format, returns the original path to save CPU/Disk I/O.
    """
    input_path = Path(input_path)

    # Check if the file is already 16kHz Mono WAV
    try:
        with wave.open(str(input_path), 'rb') as f:
            if f.getframerate() == TARGET_SR and f.getnchannels() == 1:
                # File is already optimized! Return the original path.
                return input_path
    except Exception:
        # If it's not a standard WAV or wave fails, fall back to librosa
        pass

    # Fallback to librosa for resampling/conversion
    y, sr = librosa.load(str(input_path), sr=TARGET_SR, mono=True)

    output_path = TMP_DIR / f"{input_path.stem}_prepared.wav"
    sf.write(str(output_path), y, TARGET_SR)

    return output_path