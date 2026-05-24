import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts_service")

app = FastAPI()

TARGET_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "16000"))
TTS_PROVIDER = (os.getenv("TTS_PROVIDER") or "espeak").strip().lower()
GTTS_ENABLED = (os.getenv("TTS_ENABLE_GTTS") or "0").strip() in ("1", "true", "yes", "on")


class TTSRequest(BaseModel):
    text: str


def _ffmpeg_to_pcm16_wav(src_path: str, wav_path: str) -> None:
    # Convert to PCM16 WAV for telephony playback compatibility.
    # -ac 1: mono, -ar: sample rate, -acodec pcm_s16le: 16-bit PCM
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-af",
            "atempo=1.2,volume=1.5",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-acodec",
            "pcm_s16le",
            wav_path,
        ],
        check=True,
    )


def _synthesize_espeak(text: str, wav_path: str) -> None:
    """
    Offline-first speech synthesis via espeak-ng.
    NOTE: Voice quality is limited, but it avoids mandatory cloud calls.
    """
    # Attempt Amharic voice. If unavailable, espeak-ng may fall back.
    subprocess.run(
        [
            "espeak-ng",
            "-v",
            "am",
            "-s",
            "155",
            "-w",
            wav_path,
            text,
        ],
        check=True,
    )


def _synthesize_gtts(text: str, wav_path: str) -> None:
    """
    Optional cloud TTS (gTTS). Disabled by default for offline/secure deployments.
    """
    if not GTTS_ENABLED:
        raise RuntimeError("gTTS provider is disabled (set TTS_ENABLE_GTTS=1 to allow).")
    from gtts import gTTS  # imported lazily to keep provider optional

    mp3_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    mp3_path = mp3_tmp.name
    mp3_tmp.close()

    try:
        tts = gTTS(text=text, lang="am")
        tts.save(mp3_path)
        _ffmpeg_to_pcm16_wav(mp3_path, wav_path)
    finally:
        try:
            Path(mp3_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = wav_tmp.name
        wav_tmp.close()

        if TTS_PROVIDER == "espeak":
            logger.info("Synthesizing speech (espeak-ng). chars=%s", len(req.text))
            _synthesize_espeak(req.text, wav_path)
        elif TTS_PROVIDER == "gtts":
            logger.info("Synthesizing speech (gTTS). chars=%s", len(req.text))
            _synthesize_gtts(req.text, wav_path)
        else:
            raise RuntimeError(f"Unsupported TTS_PROVIDER={TTS_PROVIDER!r} (use 'espeak' or 'gtts').")

        # Ensure sample-rate + PCM16 format even when provider generates wav.
        # (espeak-ng wav defaults are not guaranteed to match our streaming format)
        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_final = wav_tmp.name
        wav_tmp.close()

        logger.info("Speech generation successful. wav_sample_rate=%s", TARGET_SAMPLE_RATE)

        _ffmpeg_to_pcm16_wav(wav_path, wav_final)

        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass

        return FileResponse(wav_final, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        logger.exception(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed.")