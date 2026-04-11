import os
import requests
import time
import wave
import uuid
import logging
import audioop
import threading
import torch
import torchaudio
import numpy as np
from pyvoip.SIP import SIPClient, CallState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("telephony_service")

# ── Service URLs ─────────────────────────────────────────────────────────────
STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")
LOGIC_URL = os.environ.get("LOGIC_URL", "http://logic_service:8000/ask")
REPEAT_URL = os.environ.get("LOGIC_URL", "http://logic_service:8000") + "/repeat"
SAVE_RECORD_URL = os.environ.get("SAVE_RECORD_URL", "http://logic_service:8000/save_call_record")
TTS_URL = os.environ.get("TTS_URL", "http://tts_service:8000/synthesize")

# ── SIP Config ────────────────────────────────────────────────────────────────
SIP_IP = os.environ.get("SIP_IP", "0.0.0.0")
SIP_PORT = int(os.environ.get("SIP_PORT", 5060))
SIP_USER = os.environ.get("SIP_USER", "advisor")
SIP_PASS = os.environ.get("SIP_PASS", "password")

# ── VAD Config ────────────────────────────────────────────────────────────────
VAD_SILENCE_THRESHOLD = 30          # 30 × 32ms ≈ 960ms of silence → end of utterance
VAD_NO_SPEECH_TIMEOUT = 500         # 500 × 32ms ≈ 16s with no speech → reprompt
VAD_CALL_TIMEOUT_ROUNDS = 3         # Reprompt up to 3 times before hanging up

# ── Amharic prompts ───────────────────────────────────────────────────────────
PROMPT_GREETING    = "እንኳን ደህና ነዎት! የገበሬ አማካሪ ስልክ ላይ ደወሉ። ጥያቄዎን ይጠይቁ።"
PROMPT_ASK_MORE    = "ሌላ ጥያቄ አለዎት?"           # Do you have another question?
PROMPT_REPROMPT    = "እባክዎ ጥያቄዎን ይጠይቁ።"      # Please ask your question.
PROMPT_GOODBYE     = "ሰላም! ጥሩ ቀን ይሁንልዎ።"     # Goodbye! Have a good day.
PROMPT_NO_SPEECH   = "ምንም ድምፅ አልሰማሁም። ጥሩ ቀን ይሁንልዎ።"  # No audio detected. Goodbye.
PROMPT_STT_FAIL    = "ይቅርታ፣ ድምፅዎን ሰምቼ ለመረዳት አልቻልኩም። እባክዎ ይድገሙ።"  # Could not understand. Please repeat.


def _load_vad():
    """Load Silero VAD from torch hub (uses local cache after first download)."""
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        verbose=False
    )
    model.eval()
    return model


def synthesize_text(text: str, prefix: str = "tts") -> Optional[str]:
    """Call TTS service and save the resulting WAV. Returns file path or None."""
    try:
        resp = requests.post(TTS_URL, json={"text": text}, timeout=20)
        if resp.status_code == 200:
            filename = f"{prefix}_{uuid.uuid4()}.wav"
            with open(filename, "wb") as f:
                f.write(resp.content)
            return filename
        logger.error(f"TTS returned HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"TTS request failed: {e}")
    return None


def _play_and_cleanup(call, filepath: Optional[str]):
    """Play a WAV over the SIP call and then delete the file."""
    if filepath and os.path.exists(filepath):
        try:
            call.play_audio(filepath)
        except Exception as e:
            logger.warning(f"play_audio error: {e}")
        finally:
            os.remove(filepath)


def capture_utterance(call, vad_model, session_id: str):
    """
    Captures one spoken utterance from the call using Silero VAD.

    Returns:
        (audio_frames: bytearray, speech_detected: bool)

    Logic:
        - Accumulate G.711 μ-law frames, convert to 16-bit PCM.
        - Run VAD on 32ms chunks at 8000 Hz.
        - End of utterance = speech was found AND then 960ms of silence.
        - Timeout = ~16s of no speech (triggers reprompt/hangup upstream).
    """
    audio_frames = bytearray()
    vad_buffer = bytearray()
    silence_frames = 0
    no_speech_frames = 0
    speech_detected = False
    vad_model.reset_states()

    while call.state == CallState.ANSWERED:
        frame = call.read_audio()
        if not frame:
            break

        try:
            pcm_frame = audioop.ulaw2lin(frame, 2)   # G.711 → linear 16-bit
            audio_frames.extend(pcm_frame)
            vad_buffer.extend(pcm_frame)

            # Process 512-byte (256-sample, 32ms at 8 kHz) VAD chunks
            while len(vad_buffer) >= 512:
                chunk_bytes = bytes(vad_buffer[:512])
                vad_buffer = vad_buffer[512:]

                chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                chunk_tensor = torch.from_numpy(chunk_np)
                speech_prob = vad_model(chunk_tensor, 8000).item()

                if speech_prob > 0.5:
                    silence_frames = 0
                    no_speech_frames = 0
                    speech_detected = True
                else:
                    no_speech_frames += 1
                    if speech_detected:
                        silence_frames += 1

            if speech_detected and silence_frames > VAD_SILENCE_THRESHOLD:
                break  # End of utterance

            if not speech_detected and no_speech_frames > VAD_NO_SPEECH_TIMEOUT:
                break  # Timeout — no speech heard

        except Exception as e:
            logger.debug(f"VAD frame error: {e}")

    return audio_frames, speech_detected


def frames_to_wav_16k(audio_frames: bytearray, out_path: str):
    """Upsample 8 kHz PCM bytearray to 16 kHz WAV file for Whisper."""
    tensor_8k = torch.from_numpy(
        np.frombuffer(bytes(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
    )
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    tensor_16k = resampler(tensor_8k)
    torchaudio.save(out_path, tensor_16k.unsqueeze(0), 16000, bits_per_sample=16, encoding="PCM_S")


def process_audio_pipeline(wav_path: str, phone_number: str, session_id: str) -> Optional[str]:
    """
    Runs the full STT → Logic/RAG → TTS pipeline.
    Returns path to TTS response WAV, or None on failure.
    """
    # 1. STT
    try:
        with open(wav_path, "rb") as f:
            stt_resp = requests.post(STT_URL, files={"audio_file": f}, timeout=30)
        data = stt_resp.json()
        text = data.get("text", "").strip()
        confidence = data.get("confidence", 0.0)
        logger.info(f"[{session_id}] STT: '{text}' (conf={confidence:.2f})")
    except Exception as e:
        logger.error(f"[{session_id}] STT request failed: {e}")
        return None

    if not text:
        logger.warning(f"[{session_id}] STT returned empty text.")
        return None

    # 2. Logic / RAG
    try:
        payload = {"text": text, "phone_number": phone_number, "session_id": session_id}
        logic_resp = requests.post(LOGIC_URL, json=payload, timeout=30)
        response_text = logic_resp.json().get("response", "").strip()
        logger.info(f"[{session_id}] Logic: '{response_text[:80]}'")
    except Exception as e:
        logger.error(f"[{session_id}] Logic request failed: {e}")
        return None

    if not response_text:
        return None

    # 3. TTS
    return synthesize_text(response_text, prefix=f"resp_{session_id}")


def upload_call_record(wav_path: str, phone_number: str, session_id: str, duration_sec: int):
    """Uploads the full call recording to the logic service for DB storage."""
    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                SAVE_RECORD_URL,
                data={"session_id": session_id, "phone_number": phone_number, "duration": duration_sec},
                files={"audio_file": f},
                timeout=30
            )
        if resp.status_code == 200:
            logger.info(f"[{session_id}] Call record uploaded.")
        else:
            logger.warning(f"[{session_id}] Upload failed: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"[{session_id}] Upload error: {e}")


# ── Per-Call Handler (runs in its own thread) ────────────────────────────────
from typing import Optional

def handle_call(call, phone_number: str, session_id: str):
    """
    Handles a single SIP call:
      1. Answers and plays Amharic greeting.
      2. Captures utterance via Silero VAD.
      3. Runs STT → Logic → TTS pipeline.
      4. Plays response, then asks if farmer has another question.
      5. Loops up to VAD_CALL_TIMEOUT_ROUNDS times with no speech before hanging up.
      6. Saves full call recording to DB.
    """
    logger.info(f"[{session_id}] Call handler started for {phone_number}")
    all_audio = bytearray()  # accumulates the full call for recording
    vad_model = _load_vad()

    try:
        call.answer()
        logger.info(f"[{session_id}] Call answered.")

        # Play greeting
        _play_and_cleanup(call, synthesize_text(PROMPT_GREETING, f"greet_{session_id}"))

        reprompt_count = 0

        while call.state == CallState.ANSWERED:
            # Capture one utterance
            audio_frames, speech_detected = capture_utterance(call, vad_model, session_id)
            all_audio.extend(audio_frames)

            if not speech_detected:
                reprompt_count += 1
                logger.info(f"[{session_id}] No speech (attempt {reprompt_count}/{VAD_CALL_TIMEOUT_ROUNDS})")
                if reprompt_count >= VAD_CALL_TIMEOUT_ROUNDS:
                    _play_and_cleanup(call, synthesize_text(PROMPT_NO_SPEECH, f"ns_{session_id}"))
                    break
                _play_and_cleanup(call, synthesize_text(PROMPT_REPROMPT, f"rp_{session_id}"))
                continue

            reprompt_count = 0  # Reset on successful speech

            # Convert captured frames to 16 kHz WAV
            utterance_file = f"utt_{session_id}_{uuid.uuid4()}.wav"
            try:
                frames_to_wav_16k(audio_frames, utterance_file)
            except Exception as e:
                logger.error(f"[{session_id}] Resampling failed: {e}")
                continue

            # Run pipeline
            response_file = process_audio_pipeline(utterance_file, phone_number, session_id)

            # Cleanup utterance file
            if os.path.exists(utterance_file):
                os.remove(utterance_file)

            if response_file:
                _play_and_cleanup(call, response_file)
            else:
                # STT/Logic could not produce a response
                _play_and_cleanup(call, synthesize_text(PROMPT_STT_FAIL, f"fail_{session_id}"))

            # Brief pause then ask if they have another question
            time.sleep(0.5)
            if call.state == CallState.ANSWERED:
                _play_and_cleanup(call, synthesize_text(PROMPT_ASK_MORE, f"more_{session_id}"))

        # Hang up gracefully
        if call.state == CallState.ANSWERED:
            _play_and_cleanup(call, synthesize_text(PROMPT_GOODBYE, f"bye_{session_id}"))
            call.hangup()

    except Exception as e:
        logger.error(f"[{session_id}] Unhandled error in call handler: {e}")
        try:
            call.hangup()
        except Exception:
            pass

    finally:
        # Save full call recording to database
        if all_audio:
            full_path = f"full_{session_id}.wav"
            try:
                # Save at 8 kHz (telephony rate) for storage
                tensor = torch.from_numpy(
                    np.frombuffer(bytes(all_audio), dtype=np.int16).astype(np.float32) / 32768.0
                )
                torchaudio.save(full_path, tensor.unsqueeze(0), 8000, bits_per_sample=16, encoding="PCM_S")
                # Duration: 16-bit = 2 bytes/sample, 8000 samples/sec → 16000 bytes/sec
                duration_sec = int(len(all_audio) / 16000)
                upload_call_record(full_path, phone_number, session_id, duration_sec)
            except Exception as e:
                logger.error(f"[{session_id}] Failed to save call recording: {e}")
            finally:
                if os.path.exists(full_path):
                    os.remove(full_path)

        logger.info(f"[{session_id}] Call handler finished.")


# ── SIP Server ───────────────────────────────────────────────────────────────
class AdvisorSIPClient:
    def __init__(self):
        logger.info(f"Initializing SIP client on {SIP_IP}:{SIP_PORT} as user '{SIP_USER}'")
        self.sip = SIPClient(SIP_IP, SIP_PORT, SIP_USER, SIP_PASS)

    def start(self):
        logger.info("Listening for incoming calls (multi-threaded)...")
        while True:
            try:
                call = self.sip.get_call()
                if call:
                    phone_number = call.request.headers.get('From', {}).get('number', 'UnknownPhone')
                    session_id = str(uuid.uuid4())
                    logger.info(f"Incoming call from {phone_number} → session {session_id}")

                    # Spawn a dedicated thread per call (non-blocking accept loop)
                    t = threading.Thread(
                        target=handle_call,
                        args=(call, phone_number, session_id),
                        daemon=True
                    )
                    t.start()
            except Exception as e:
                logger.error(f"SIP loop error: {e}")

            time.sleep(0.1)   # Yield to other threads


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    time.sleep(5)   # Allow other containers to start first
    logger.info("Telephony Service starting...")
    client = AdvisorSIPClient()
    client.start()  # Blocks forever in accept loop
