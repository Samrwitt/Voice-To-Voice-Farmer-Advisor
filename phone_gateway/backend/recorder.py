import os
from pathlib import Path
import wave


class AudioRecorder:
    def __init__(self, session_id: str, sample_rate: int = 16000):
        self.session_id = session_id
        self.sample_rate = sample_rate

        recordings_base = os.getenv("RECORDINGS_DIR", "recordings")
        self.recordings_dir = Path(recordings_base) / "audio"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.recordings_dir / f"{session_id}.pcm"
        self.file = open(self.file_path, "ab")

    def write_chunk(self, chunk: bytes):
        if chunk:
            self.file.write(chunk)
            self.file.flush()

    def close(self) -> str:
        if not self.file.closed:
            self.file.close()

        # Convert raw PCM16 mono -> WAV so browsers can play it.
        wav_path = self.recordings_dir / f"{self.session_id}.wav"
        try:
            pcm_bytes = self.file_path.read_bytes()
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # PCM16
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_bytes)
            return str(wav_path)
        except Exception:
            # Fallback to raw PCM if conversion fails.
            return str(self.file_path)