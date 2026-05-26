import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path

from schemas import ASRResponse, FileTranscribeRequest, PostprocessTextRequest
from engine import create_asr_engine
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
    return {
        "status": status,
        "service": "asr_service",
        "engine_loaded": asr_engine is not None,
        "engine_loading": asr_engine_loading,
        "engine_error": asr_engine_error,
        "asr_engine_config": ASR_ENGINE,
        "asr_engine_runtime": getattr(asr_engine, "engine_name", None),
    }


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
        return result


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
        return result


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))