import os
import requests
import time
import wave
import uuid
import logging
import audioop
import torch
import torchaudio
import numpy as np
from pyvoip.SIP import SIPClient, CallState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("telephony_service")

STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")
LOGIC_URL = os.environ.get("LOGIC_URL", "http://logic_service:8000/ask")
TTS_URL = os.environ.get("TTS_URL", "http://tts_service:8000/synthesize")

SIP_IP = os.environ.get("SIP_IP", "0.0.0.0")
SIP_PORT = int(os.environ.get("SIP_PORT", 5060))
SIP_USER = os.environ.get("SIP_USER", "advisor")
SIP_PASS = os.environ.get("SIP_PASS", "password")

logger.info(f"Evaluating SIP Client on {SIP_IP}:{SIP_PORT} for user {SIP_USER}")

# Load Silero VAD
logger.info("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
vad_model.eval()

def process_audio_pipeline(wav_file_path: str, phone_number: str, session_id: str):
    """Orchestrates HTTP requests STT -> Logic -> TTS"""
    logger.info("1. Sending 16kHz audio to STT...")
    try:
        with open(wav_file_path, "rb") as f:
            stt_resp = requests.post(STT_URL, files={"audio_file": f})
        text = stt_resp.json().get("text", "")
        logger.info(f"STT Output: {text}")
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return None

    if not text:
        return None

    logger.info("2. Sending to Logic (RAG)...")
    try:
        payload = {"text": text, "phone_number": phone_number, "session_id": session_id}
        logic_resp = requests.post(LOGIC_URL, json=payload)
        logic_data = logic_resp.json()
        response_text = logic_data.get("response", "")
        logger.info(f"Logic Output: {response_text}")
    except Exception as e:
        logger.error(f"Logic Error: {e}")
        return None

    logger.info("3. Sending to TTS...")
    try:
        tts_resp = requests.post(TTS_URL, json={"text": response_text})
        output_filename = f"response_{session_id}.wav"
        with open(output_filename, "wb") as f:
            f.write(tts_resp.content)
        logger.info(f"TTS Output written to {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None

class AdvisorSIPClient:
    def __init__(self):
        self.sip = SIPClient(SIP_IP, SIP_PORT, SIP_USER, SIP_PASS)
        
    def start(self):
        logger.info("Listening for incoming calls...")
        while True:
            call = self.sip.get_call()
            if call:
                phone_number = call.request.headers.get('From', {}).get('number', 'UnknownPhone')
                session_id = str(uuid.uuid4())
                logger.info(f"Incoming call from {phone_number}, designated Session ID: {session_id}")
                call.answer()
                
                logger.info("VAD Tracker initialized. Listening for audio chunks...")
                audio_file = f"temp_{session_id}.wav"
                
                audio_frames = bytearray()
                vad_buffer = bytearray()
                silence_frames = 0
                vad_model.reset_states()
                
                while call.state == CallState.ANSWERED:
                    frame = call.read_audio()
                    if not frame:
                        break
                    
                    try:
                        # PyVoIP yields 160 bytes = 20ms G711 PCMU frames. We convert to 16-bit PCM (320 bytes)
                        pcm_frame = audioop.ulaw2lin(frame, 2)
                        audio_frames.extend(pcm_frame)
                        vad_buffer.extend(pcm_frame)
                        
                        # Silero VAD at 8000Hz works well with chunks of 256 samples = 512 bytes (32ms)
                        while len(vad_buffer) >= 512:
                            chunk_bytes = vad_buffer[:512]
                            vad_buffer = vad_buffer[512:]
                            
                            chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            chunk_tensor = torch.from_numpy(chunk_np)
                            
                            speech_prob = vad_model(chunk_tensor, 8000).item()
                            
                            if speech_prob > 0.5:
                                silence_frames = 0
                            else:
                                silence_frames += 1
                                
                        # 30 * 32ms ~= 960ms silence before endpointing
                        if silence_frames > 30:
                            logger.info("End of utterance detected.")
                            break
                    except Exception as e:
                        pass

                # Upsample accumulated 8000Hz frames to 16000Hz for ASR/STT
                tensor_8k = torch.from_numpy(np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32) / 32768.0)
                resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
                tensor_16k = resampler(tensor_8k)
                
                torchaudio.save(audio_file, tensor_16k.unsqueeze(0), 16000, bits_per_sample=16, encoding="PCM_S")

                logger.info("Utterance captured and upsampled. Routing to pipeline...")
                response_file = process_audio_pipeline(audio_file, phone_number, session_id)
                
                if response_file and os.path.exists(response_file):
                    logger.info(f"Playing HTTP TTS response back to SIP caller: {response_file}")
                    call.play_audio(response_file)
                
                time.sleep(2)
                if call.state == CallState.ANSWERED:
                    call.hangup()
            time.sleep(0.5)

if __name__ == "__main__":
    time.sleep(5)
    logger.info("Telephony Service Process Started.")
    client = AdvisorSIPClient()
    
    logger.info("Starting production SIP loop -> binding pyvoip client server...")
    client.start() 

    while True:
        time.sleep(60)
