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

# Load MMS TTS Amharic
logger.info("Loading MMS TTS Amharic model...")
try:
    model_name = "facebook/mms-tts-amh"
    tokenizer = VitsTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Failed to load MMS TTS model: {e}")

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    output_file = f"response_{uuid.uuid4()}.wav"
    
    def cleanup():
        if os.path.exists(output_file):
            os.remove(output_file)

    try:
        logger.info(f"Synthesizing speech for payload: {req.text}")
        inputs = tokenizer(req.text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model(**inputs).waveform
            
        orig_freq = model.config.sampling_rate
        
        # Resample to 8kHz for telephony
        target_freq = 8000
        if orig_freq != target_freq:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_freq).to(device)
            output = resampler(output)
            
        output = output.cpu()
        
        # Save as 16-bit PCM WAV  
        torchaudio.save(output_file, output, target_freq, bits_per_sample=16, encoding="PCM_S")
        logger.info("Speech generation successful.")
        
    except Exception as e:
        logger.error(f"TTS Failed: {e}")
        # Create an empty silent 8kHz wav as fallback
        empty_tensor = torch.zeros((1, 8000))
        torchaudio.save(output_file, empty_tensor, 8000, bits_per_sample=16, encoding="PCM_S")
            
    return FileResponse(output_file, media_type="audio/wav", filename="response.wav", background=BackgroundTask(cleanup))
