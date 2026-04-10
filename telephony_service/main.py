import os
import requests
import time
import wave
import uuid
import logging
import webrtcvad
import audioop
from pyvoip.SIP import SIPClient, CallState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("telephony_service")

STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")
LOGIC_URL = os.environ.get("LOGIC_URL", "http://logic_service:8000/ask")
SAVE_RECORD_URL = os.environ.get("SAVE_RECORD_URL", "http://logic_service:8000/save_call_record")
TTS_URL = os.environ.get("TTS_URL", "http://tts_service:8000/synthesize")

SIP_IP = os.environ.get("SIP_IP", "0.0.0.0")
SIP_PORT = int(os.environ.get("SIP_PORT", 5060))
SIP_USER = os.environ.get("SIP_USER", "advisor")
SIP_PASS = os.environ.get("SIP_PASS", "password")

logger.info(f"Evaluating SIP Client on {SIP_IP}:{SIP_PORT} for user {SIP_USER}")

# VAD Object, set aggressiveness to 3 (highest)
vad = webrtcvad.Vad(3)

def upload_call_record(wav_file_path: str, phone_number: str, session_id: str, duration: int):
    logger.info("Uploading full call record to database via Logic Service...")
    try:
        with open(wav_file_path, "rb") as f:
            data = {"session_id": session_id, "phone_number": phone_number, "duration": duration}
            files = {"audio_file": f}
            resp = requests.post(SAVE_RECORD_URL, data=data, files=files)
            if resp.status_code == 200:
                logger.info("Successfully recorded call in database.")
            else:
                logger.warning(f"Failed to record call. Status: {resp.status_code}")
    except Exception as e:
        logger.error(f"Error uploading call record: {e}")

def process_audio_pipeline(wav_file_path: str, phone_number: str, session_id: str):
    """Orchestrates HTTP requests STT -> Logic -> TTS"""
    logger.info("1. Sending to STT...")
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
        output_filename = f"response_{session_id}.mp3"
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
                
                # SRS FR02: Real-time Audio End-pointing Sequence 
                logger.info("VAD Tracker initialized. Listening for audio chunks...")
                audio_file = f"temp_{session_id}.wav"
                
                audio_frames = bytearray()
                silence_frames = 0
                
                while call.state == CallState.ANSWERED:
                    frame = call.read_audio()
                    if not frame:
                        break
                    
                    try:
                        # PyVoIP typically yields standard 20ms G711 PCMU frames (160 bytes)
                        pcm_frame = audioop.ulaw2lin(frame, 2)
                        audio_frames.extend(pcm_frame)
                        
                        if len(pcm_frame) == 320: # 16-bit * 8000Hz * 0.02s
                            is_speech = vad.is_speech(pcm_frame, 8000)
                            if is_speech:
                                silence_frames = 0
                            else:
                                silence_frames += 1
                                
                        if silence_frames > 40: # ~800ms silence ends utterance
                            logger.info("End of utterance detected.")
                            break
                    except Exception as e:
                        # If frame is unexpected size or codec mismatch
                        pass

                with wave.open(audio_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(audio_frames)

                logger.info("Utterance captured. Routing to pipeline...")
                response_file = process_audio_pipeline(audio_file, phone_number, session_id)
                
                if response_file and os.path.exists(response_file):
                    logger.info(f"Playing HTTP TTS response back to SIP caller: {response_file}")
                    # Play synthesized audio stream back to handset
                    call.play_audio(response_file)
                
                # Brief pause before hanging up to ensure audio empties buffer
                time.sleep(2)
                if call.state == CallState.ANSWERED:
                    call.hangup()
                
                # Upload recording
                duration_sec = int(len(audio_frames) / 16000)
                upload_call_record(audio_file, phone_number, session_id, duration_sec)
                
            time.sleep(0.5)

if __name__ == "__main__":
    time.sleep(5)
    logger.info("Telephony Service Process Started.")
    client = AdvisorSIPClient()
    
    logger.info("Starting production SIP loop -> binding pyvoip client server...")
    client.start() 

    while True:
        time.sleep(60)
