from fastapi import FastAPI, UploadFile, File
import aiofiles
import os
import uuid
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("asr_service")

app = FastAPI()

# Load faster-whisper model
logger.info("Loading faster-whisper small model on CUDA...")
try:
    asr_model = WhisperModel("small", device="cuda", compute_type="float16")
    logger.info("Loaded successfully on CUDA.")
except Exception as e:
    logger.warning(f"Failed to load on CUDA: {e}. Falling back to CPU...")
    asr_model = WhisperModel("small", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    text = ""
    try:
        # 1. Save File
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)
        
        segments, info = asr_model.transcribe(temp_filename, language="am", beam_size=5)
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text.strip())
            
        text = " ".join(text_segments).strip()
        logger.info(f"Transcription successful: {text}")
        
    except Exception as e:
        logger.error(f"[{session_id}] ASR Error: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)