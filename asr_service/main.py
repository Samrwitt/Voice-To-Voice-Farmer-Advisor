import os
import threading
import wave

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path

from schemas import ASRResponse, FileTranscribeRequest, PostprocessTextRequest
from engine import create_asr_engine, GeminiASREngine, WhisperASREngine
from audio_utils import save_upload_file, prepare_audio_for_asr
from config import SHARED_UTTERANCES_DIR, ASR_ENGINE
from postprocess import postprocess_asr_transcript
# Hosted Groq/Gemini ASR correction is disabled for now to avoid token usage.
# from hosted_llm_fix import (
#     hosted_fix_enabled,
#     groq_keys,
#     gemini_keys,
#     free_gemini_keys,
#     shared_gemini_keys,
#     use_shared_gemini_keys_for_asr,
#     _backend_mode,
# )


app = FastAPI(
    title="Amharic ASR Service",
    description="Local Whisper Amharic ASR with post-processing and optional Gemini typo fix.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

asr_engine = None
asr_engine_loading = False
asr_engine_error: str | None = None
_load_lock = threading.Lock()
_gemini_fallback_engine = None
_gemini_fallback_lock = threading.Lock()


def _wav_duration_sec(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


def _gemini_fallback_enabled() -> bool:
    return os.getenv("ASR_GEMINI_FALLBACK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _maybe_gemini_fallback(prepared_path: Path, whisper_result: dict) -> dict:
    """Re-transcribe with Gemini audio when local Whisper quality is poor."""
    if not _gemini_fallback_enabled():
        return whisper_result
    if not isinstance(asr_engine, WhisperASREngine):
        return whisper_result
    try:
        min_words = int(os.getenv("ASR_GEMINI_FALLBACK_MIN_WORDS", "5") or "5")
        max_conf = float(os.getenv("ASR_GEMINI_FALLBACK_MAX_CONFIDENCE", "0.84") or "0.84")
        min_audio_sec = float(os.getenv("ASR_GEMINI_FALLBACK_MIN_AUDIO_SEC", "6") or "6")
    except ValueError:
        min_words, max_conf, min_audio_sec = 5, 0.84, 6.0

    final = (whisper_result.get("final_transcript") or whisper_result.get("transcript") or "")
    word_count = len([w for w in final.split() if w])
    confidence = float(whisper_result.get("confidence") or 0.0)
    audio_sec = _wav_duration_sec(prepared_path)

    # Paid audio fallback is expensive; skip if text fix already ran or transcript looks clean.
    if whisper_result.get("transcript_fix_backend"):
        return whisper_result
    unusual = [w for w in (whisper_result.get("unusual_words") or []) if len(str(w).strip()) >= 2]
    if word_count >= 5:
        unusual_ratio = len(unusual) / word_count
        max_unusual = float(os.getenv("ASR_GEMINI_FALLBACK_MAX_UNUSUAL_RATIO", "0.32") or "0.32")
        if unusual_ratio < max_unusual:
            return whisper_result

    if word_count < min_words or confidence >= max_conf:
        return whisper_result
    if min_audio_sec > 0 and audio_sec < min_audio_sec:
        return whisper_result

    global _gemini_fallback_engine
    with _gemini_fallback_lock:
        if _gemini_fallback_engine is None:
            try:
                _gemini_fallback_engine = GeminiASREngine()
            except Exception as exc:
                print(f"Gemini ASR fallback unavailable: {exc}", flush=True)
                return whisper_result

    try:
        gemini_result = _gemini_fallback_engine.transcribe(prepared_path)
        gemini_result["engine"] = f"{whisper_result.get('engine', 'whisper_local')}+gemini_fallback"
        return gemini_result
    except Exception as exc:
        print(f"Gemini ASR fallback failed: {exc}", flush=True)
        return whisper_result


def _load_asr_engine_background() -> None:
    global asr_engine, asr_engine_loading, asr_engine_error
    with _load_lock:
        if asr_engine is not None or asr_engine_loading:
            return
        asr_engine_loading = True
        asr_engine_error = None

    print(f"ASR background load started (engine={ASR_ENGINE})", flush=True)
    try:
        engine = create_asr_engine()
        with _load_lock:
            asr_engine = engine
            print("ASR engine ready.", flush=True)
    except Exception as e:
        with _load_lock:
            asr_engine_error = str(e)
        print(f"Failed to initialize ASR engine: {e}", flush=True)
    finally:
        with _load_lock:
            asr_engine_loading = False


@app.on_event("startup")
def startup_event():
    # Load the local Whisper model in the background so health endpoints respond.
    threading.Thread(target=_load_asr_engine_background, daemon=True).start()


def _asr_ready_or_http() -> None:
    if asr_engine is not None:
        return
    if asr_engine_loading:
        raise HTTPException(
            status_code=503,
            detail=(
                f"ASR model ({ASR_ENGINE}) is still loading. "
                "Check GET /health and retry in a few minutes on first boot."
            ),
        )
    detail = asr_engine_error or "ASR engine is not initialized."
    raise HTTPException(status_code=503, detail=detail)


@app.get("/health")
def health():
    if asr_engine is not None:
        status = "ok"
    elif asr_engine_loading:
        status = "loading"
    else:
        status = "degraded"
    payload = {
        "status": status,
        "service": "asr_service",
        "engine_loaded": asr_engine is not None,
        "engine_loading": asr_engine_loading,
        "engine_error": asr_engine_error,
        "asr_engine_config": ASR_ENGINE,
        "asr_engine_runtime": getattr(asr_engine, "engine_name", None),
    }
    if asr_engine is not None:
        payload["whisper_device"] = getattr(asr_engine, "device", None)
        payload["whisper_compute_type"] = getattr(asr_engine, "compute_type", None)
    try:
        from engine import _cuda_device_count

        payload["cuda_devices_visible"] = _cuda_device_count()
    except Exception:
        payload["cuda_devices_visible"] = None
    return payload


@app.get("/fix-status")
def fix_status():
    """Whether hosted Groq/Gemini post-ASR correction is active (keys + env)."""
    return {
        "hosted_fix_enabled": False,
        "backend_mode": "disabled",
        "groq_key_count": 0,
        "gemini_key_count": 0,
        "free_gemini_key_count": 0,
        "shared_gemini_key_count": 0,
        "using_shared_gemini_keys": False,
        "note": "Hosted ASR token usage is commented out for now.",
    }


@app.post("/postprocess-text")
def postprocess_text(request: PostprocessTextRequest):
    """
    Run the same post-ASR pipeline on arbitrary text (typo / homophone / LLM fix).
    Does not run the acoustic model.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    p = postprocess_asr_transcript(text)
    return {
        "input": text,
        "raw_transcript": p["raw"],
        "domain_corrected_transcript": p["domain_corrected"],
        "semantic_corrected_transcript": p.get("semantic_corrected"),
        "transcript_fix_backend": p.get("transcript_fix_backend"),
        "final_transcript": p["final"],
        "structured_transcript": p["structured_transcript"],
        "transcript": p["final"],
        "confidence": p["confidence"],
        "fuzzy": p.get("fuzzy") or {},
        "needs_confirmation": p["needs_confirmation"],
        "confirmation_prompt": p["confirmation_prompt"],
    }


@app.post("/transcribe", response_model=ASRResponse)
async def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Main transcription endpoint for audio uploads.
    Accepts both 'file' and 'audio_file' for backward compatibility.
    """
    _asr_ready_or_http()

    target_file = file or audio_file

    if not target_file:
        raise HTTPException(
            status_code=400,
            detail="No audio file provided. Please use 'file' or 'audio_file' field."
        )

    try:
        uploaded_path = save_upload_file(target_file)
        prepared_path = prepare_audio_for_asr(uploaded_path)
        result = asr_engine.transcribe(prepared_path)
        return _maybe_gemini_fallback(prepared_path, result)


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-file", response_model=ASRResponse)
async def transcribe_file(request: FileTranscribeRequest):
    """
    Endpoint for VAD service to request transcription of a file in the shared volume.
    """
    _asr_ready_or_http()

    safe_filename = Path(request.filename).name
    audio_path = Path(SHARED_UTTERANCES_DIR) / safe_filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found in shared volume: {safe_filename}. Expected path: {audio_path}",
        )

    try:
        prepared_path = prepare_audio_for_asr(audio_path)
        result = asr_engine.transcribe(prepared_path)
        return _maybe_gemini_fallback(prepared_path, result)


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))