from fastapi import FastAPI, UploadFile, File
import aiofiles
import os
import uuid
import math
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("asr_service")

app = FastAPI()

# Load faster-whisper model with CUDA/CPU fallback
logger.info("Loading faster-whisper small model...")
try:
    asr_model = WhisperModel("small", device="cuda", compute_type="float16")
    logger.info("Loaded successfully on CUDA.")
except Exception as e:
    logger.warning(f"Failed to load on CUDA: {e}. Falling back to CPU...")
    asr_model = WhisperModel("small", device="cpu", compute_type="int8")
    logger.info("Loaded on CPU with int8 quantization.")


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    """
    Accepts a WAV audio file, transcribes it using faster-whisper (Amharic),
    and returns the transcription text with an average confidence score.

    Returns:
        {"text": str, "confidence": float}  on success
        {"text": "", "confidence": 0.0, "error": str}  on failure
    """
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    try:
        # Save the uploaded audio to a temp file
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)

        # Transcribe in Amharic
        segments, info = asr_model.transcribe(
            temp_filename,
            language="am",
            beam_size=5,
            vad_filter=True,          # built-in VAD to skip silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text_parts = []
        log_probs = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            # avg_logprob is negative; convert to 0-1 confidence
            if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                log_probs.append(math.exp(max(segment.avg_logprob, -10)))

        text = " ".join(text_parts).strip()
        confidence = round(sum(log_probs) / len(log_probs), 3) if log_probs else 0.0

        logger.info(f"Transcription: '{text}' | Confidence: {confidence}")
        return {"text": text, "confidence": confidence}

    except Exception as e:
        logger.error(f"ASR Error: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)