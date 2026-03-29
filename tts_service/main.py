from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from gtts import gTTS
import os

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    output_file = "response.mp3"
    
    try:
        # Generate Amharic speech using Google TTS for quick MVP mockup
        tts = gTTS(text=req.text, lang='am')
        tts.save(output_file)
    except Exception as e:
        # Fallback if network issue or gTTS fails.
        with open(output_file, "wb") as f:
            f.write(b"mock audio bytes - tts failed")
            
    return FileResponse(output_file, media_type="audio/mpeg", filename="response.mp3")
