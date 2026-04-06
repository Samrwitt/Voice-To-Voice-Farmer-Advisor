from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from gtts import gTTS
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tts_service")

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    output_file = "response.mp3"
    
    try:
        logger.info(f"Synthesizing speech for payload: {req.text}")
        tts = gTTS(text=req.text, lang='am')
        tts.save(output_file)
        logger.info("Speech generation successful.")
    except Exception as e:
        logger.error(f"TTS Failed: {e}")
        with open(output_file, "wb") as f:
            f.write(b"mock audio bytes - tts failed")
            
    return FileResponse(output_file, media_type="audio/mpeg", filename="response.mp3")
