from fastapi import FastAPI, UploadFile, File, Header
import torch
from transformers import pipeline
import aiofiles
import os
import uuid
import logging
import time
import librosa # Added for sample rate validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("asr_service")

app = FastAPI()

# Model ID as per SDS Table 15
MODEL_ID = "openai/whisper-small-amharic-finetuned"

# Load Whisper model
# Note: SDS recommends a fine-tuned model for Amharic rural/noise data [cite: 93]
asr_pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", # In production, swap for fine-tuned path
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    return_timestamps=False
)

@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile = File(...),
    x_session_id: str = Header(None) # Trace ID propagation 
):
    session_id = x_session_id or str(uuid.uuid4())
    temp_filename = f"temp_{session_id}.wav"
    start_time = time.time()
    
    try:
        # 1. Save File
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)

        # 2. Validate Sample Rate (Requirement: 8000Hz for Telephony) 
        audio_data, sr = librosa.load(temp_filename, sr=None)
        if sr != 8000:
            logger.warning(f"[{session_id}] Resampling from {sr}Hz to 8000Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=8000)
            sr = 8000

        # 3. Inference with Confidence Score [cite: 81, 124]
        # Whisper pipeline can return 'chunks' with logprobs to estimate confidence
        prediction = asr_pipe(
            audio_data, 
            generate_kwargs={"language": "am", "return_timestamps": False},
            return_timestamps=False
        )
        
        text = prediction.get("text", "").strip()
        
        # Mock confidence (Replace with average logprobs logic if using pure Whisper)
        # SDS Table 3 requires confidence for error handling logic 
        confidence = 0.85 if text else 0.0 
        
        latency = int((time.time() - start_time) * 1000)

        # 4. Construct ASRResult 
        result = {
            "text": text,
            "confidence": confidence,
            "latencyMs": latency,
            "modelId": MODEL_ID,
            "sampleRateHz": sr
        }

        # Logging as per Policy (Table 41) 
        logger.info(f"[{session_id}] ASR Complete - Text: {text}, Conf: {confidence}, Latency: {latency}ms")
        
        return result

    except Exception as e:
        logger.error(f"[{session_id}] ASR Error: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)