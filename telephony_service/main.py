import os
import requests
import time
import wave
import time
from pyvoip.SIP import SIPClient, CallState

STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")
LOGIC_URL = os.environ.get("LOGIC_URL", "http://logic_service:8000/ask")
TTS_URL = os.environ.get("TTS_URL", "http://tts_service:8000/synthesize")

SIP_IP = os.environ.get("SIP_IP", "0.0.0.0")
SIP_PORT = int(os.environ.get("SIP_PORT", 5060))
SIP_USER = os.environ.get("SIP_USER", "advisor")
SIP_PASS = os.environ.get("SIP_PASS", "password")

print(f"Starting SIP Client on {SIP_IP}:{SIP_PORT} for user {SIP_USER}")

# Generic fallback logic for pyvoip
def process_audio_pipeline(wav_file_path: str):
    """Orchestrates the HTTP requests STT -> Logic -> TTS"""
    print("1. Sending to STT...")
    try:
        with open(wav_file_path, "rb") as f:
            stt_resp = requests.post(STT_URL, files={"audio_file": f})
        text = stt_resp.json().get("text", "")
        print(f"STT Output: {text}")
    except Exception as e:
        print(f"STT Error: {e}")
        return None

    if not text:
        return None

    print("2. Sending to Logic (RAG)...")
    try:
        logic_resp = requests.post(LOGIC_URL, json={"text": text})
        logic_data = logic_resp.json()
        response_text = logic_data.get("response", "")
        print(f"Logic Output: {response_text}")
    except Exception as e:
        print(f"Logic Error: {e}")
        return None

    print("3. Sending to TTS...")
    try:
        tts_resp = requests.post(TTS_URL, json={"text": response_text})
        output_filename = "response_audio.mp3"
        with open(output_filename, "wb") as f:
            f.write(tts_resp.content)
        print("TTS Output received")
        return output_filename
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

class AdvisorSIPClient:
    def __init__(self):
        # pyvoip SIPClient setup 
        self.sip = SIPClient(SIP_IP, SIP_PORT, SIP_USER, SIP_PASS)
        
    def start(self):
        print("Listening for incoming calls...")
        while True:
            call = self.sip.get_call()
            if call:
                print(f"Incoming call from {call.request.headers['From']['number']}")
                call.answer()
                
                # Prototype simplification: We record x seconds of audio, process it, and respond.
                # In production, this would be a constant stream using a VAD trigger.
                print("Recording audio...")
                
                # Mock recording step: Since real pyvoip audio streaming handling requires 
                # a dedicated thread reading RTP packets, we stub the pipeline trigger here.
                # In reality, you'd collect PCM data from `call.read_audio()`
                
                audio_file = "temp_recording.wav" # Assumed recorded audio from caller
                
                # For demonstration of the orchestration loop, we process it:
                # response_file = process_audio_pipeline(audio_file)
                # if response_file:
                #     call.play_audio(response_file)
                
                time.sleep(5)
                call.hangup()
            time.sleep(1)

if __name__ == "__main__":
    # We add a sleep allowing other services to boot up
    time.sleep(10)
    client = AdvisorSIPClient()
    # client.start() # Disabled until valid SIP server is configured to prevent loop crashes
    
    # As a fallback proxy for testing the pipeline without a SIP server:
    # process_audio_pipeline(mock_file)
    print("Telephony Service initialized.")
    while True:
        time.sleep(60)
