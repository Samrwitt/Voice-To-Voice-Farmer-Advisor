from fastapi import FastAPI, UploadFile, File
import random

app = FastAPI()

# MOCK: In production, load whisper model here
# from transformers import pipeline
# asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    # Read the file to simulate processing
    content = await audio_file.read()
    
    # Mocking the transcription of an Amharic audio chunk.
    # In reality: text = asr(content)["text"]
    
    mock_transcriptions = [
        "የአየር ሁኔታ እንዴት ነው", # How is the weather
        "በሰብሌ ላይ በሽታ አየሁ", # I saw a disease on my crop
        "የማዳበሪያ ዋጋ ስንት ነው" # How much is fertilizer?
    ]
    
    # We just return a random mock text to ensure the pipeline flows seamlessly.
    # We can use file size hash to pick predictably if needed, but random is okay for mocking.
    text = random.choice(mock_transcriptions)
    
    return {"text": text}
