from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from transformers import VitsModel, VitsTokenizer
import torch
import torchaudio
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tts_service")

app = FastAPI()


class TTSRequest(BaseModel):
    text: str


# ── Model Loading ────────────────────────────────────────────────────────────
_model_loaded = False
model = None
tokenizer = None
device = "cpu"

logger.info("Loading MMS TTS Amharic model (facebook/mms-tts-amh)...")
try:
    model_name = "facebook/mms-tts-amh"
    tokenizer = VitsTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    _model_loaded = True
    logger.info(f"MMS TTS loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Failed to load MMS TTS model: {e}. Synthesize will return silence.")


def _make_silent_wav(path: str, duration_seconds: float = 1.0, sample_rate: int = 8000):
    """Writes a silent WAV file as a fallback."""
    n_samples = int(sample_rate * duration_seconds)
    silent = torch.zeros((1, n_samples))
    torchaudio.save(path, silent, sample_rate, bits_per_sample=16, encoding="PCM_S")


# ── Synthesis Endpoint ───────────────────────────────────────────────────────
@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """
    Converts Amharic text to speech.
    Returns a WAV file at 8 kHz (telephony-compatible 16-bit PCM).
    Falls back to 1 second of silence if model cannot synthesize.
    """
    output_file = f"response_{uuid.uuid4()}.wav"

    def cleanup():
        if os.path.exists(output_file):
            os.remove(output_file)

    if not _model_loaded or model is None or tokenizer is None:
        logger.warning("Model not loaded — returning silent WAV fallback.")
        _make_silent_wav(output_file)
        return FileResponse(output_file, media_type="audio/wav",
                            filename="response.wav", background=BackgroundTask(cleanup))

    try:
        logger.info(f"Synthesizing: '{req.text[:80]}...' " if len(req.text) > 80 else f"Synthesizing: '{req.text}'")

        inputs = tokenizer(req.text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs).waveform

        orig_freq = model.config.sampling_rate

        # Resample to 8 kHz for telephony compatibility
        if orig_freq != 8000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=8000
            ).to(device)
            output = resampler(output)

        output = output.cpu()
        torchaudio.save(output_file, output, 8000, bits_per_sample=16, encoding="PCM_S")
        logger.info(f"Synthesis successful → {output_file}")

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e} — returning silent fallback.")
        _make_silent_wav(output_file)

    return FileResponse(output_file, media_type="audio/wav",
                        filename="response.wav", background=BackgroundTask(cleanup))
