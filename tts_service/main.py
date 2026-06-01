import logging
import os
import subprocess
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts_service")

app = FastAPI()

TARGET_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "16000"))


class TTSRequest(BaseModel):
    text: str


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        # gTTS uses Google Translate TTS (network call).
        logger.info("Synthesizing speech (gTTS). chars=%s", len(req.text))

        tts = gTTS(text=req.text, lang="am")
        mp3_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        mp3_path = mp3_tmp.name
        mp3_tmp.close()

        tts.save(mp3_path)

        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = wav_tmp.name
        wav_tmp.close()

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
                mp3_path,
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

        logger.info("Speech generation successful. wav_sample_rate=%s", TARGET_SAMPLE_RATE)

        return FileResponse(wav_path, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        logger.exception(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed.")