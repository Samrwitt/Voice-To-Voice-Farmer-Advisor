from fastapi import FastAPI, UploadFile, File
import torch
from transformers import pipeline
import aiofiles
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stt_service")

app = FastAPI()

# Load whisper model (Using base or small for lower CPU/RAM footprint on initial tests)
asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    try:
        logger.info(f"Receiving audio payload for transcription...")
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)
        
        prediction = asr(temp_filename, generate_kwargs={"language": "am"})
        text = prediction.get("text", "").strip()
        logger.info(f"Transcription successful: {text}")
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return {"error": str(e), "text": ""}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return {"text": text}
