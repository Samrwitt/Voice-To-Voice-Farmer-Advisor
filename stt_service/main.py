from fastapi import FastAPI, UploadFile, File
import torch
from transformers import pipeline
import aiofiles
import os
import uuid

app = FastAPI()

# Load whisper model (Using base or small for lower CPU/RAM footprint on initial tests)
# We can use whisper-base or whisper-small, which natively support Amharic (lang='am')
asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    # Save the uploaded file temporarily to disk to process with librosa/transformers
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    try:
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)
        
        # Transcribe specifying Amharic if known, or let Whisper detect
        # To force Amharic, one can use generate_kwargs={"language": "am"}
        prediction = asr(temp_filename, generate_kwargs={"language": "am"})
        text = prediction.get("text", "").strip()
        
    except Exception as e:
        return {"error": str(e), "text": ""}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return {"text": text}
