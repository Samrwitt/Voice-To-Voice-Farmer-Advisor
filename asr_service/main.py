# import os
# import uuid
# import shutil
# import logging
# from pathlib import Path
# from typing import Optional
# 
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from pydantic import BaseModel
# 
# 
# # ============================================================
# # Logging
# # ============================================================
# 
# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "INFO"),
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
# 
# logger = logging.getLogger("asr-service")
# 
# 
# # ============================================================
# # Configuration
# # ============================================================
# 
# ASR_ENGINE = os.getenv("ASR_ENGINE", "mock").lower()
# # Supported:
# #   mock
# #   whisper_local
# #   speechbrain
# 
# MODEL_PATH = os.getenv("ASR_MODEL_PATH", "/models")
# ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
# 
# UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_audio"))
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# 
# MODEL_DIR = Path(MODEL_PATH)
# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# 
# SHARED_UTTERANCES_DIR = Path(
#     os.getenv("SHARED_UTTERANCES_DIR", "/shared/utterances")
# )
# SHARED_UTTERANCES_DIR.mkdir(parents=True, exist_ok=True)
# 
# 
# # ============================================================
# # Schemas
# # ============================================================
# 
# class ASRResponse(BaseModel):
#     transcript: str
#     language: str
#     confidence: float
#     engine: str
#     audio_id: str
# 
# 
# class FileTranscribeRequest(BaseModel):
#     filename: str
#     language: str = "am"
# 
# 
# class HealthResponse(BaseModel):
#     status: str
#     service: str
#     engine: str
#     model_path: str
#     shared_utterances_dir: str
#     provider_loaded: bool
#     error: Optional[str] = None
# 
# 
# # ============================================================
# # ASR Provider Interface
# # ============================================================
# 
# class BaseASRProvider:
#     engine_name = "base"
# 
#     def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
#         raise NotImplementedError
# 
# 
# # ============================================================
# # Mock ASR Provider
# # ============================================================
# 
# class MockASRProvider(BaseASRProvider):
#     engine_name = "mock"
# 
#     def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
#         audio_id = Path(audio_path).stem
# 
#         return ASRResponse(
#             transcript="የበቆሎ ተባይ እንዴት መከላከል እችላለሁ?",
#             language=language,
#             confidence=0.92,
#             engine=self.engine_name,
#             audio_id=audio_id,
#         )
# 
# 
# # ============================================================
# # Whisper Local Provider Placeholder
# # ============================================================
# 
# class WhisperLocalASRProvider(BaseASRProvider):
#     engine_name = "whisper_local"
# 
#     def __init__(self, model_path: str):
#         if not model_path:
#             raise ValueError(
#                 "ASR_MODEL_PATH is required when ASR_ENGINE=whisper_local"
#             )
# 
#         self.model_path = model_path
#         self.model = None
#         self.load_model()
# 
#     def load_model(self):
#         logger.info("Loading Whisper model from: %s", self.model_path)
# 
#         # Later, when your faster-whisper model is ready:
#         #
#         # from faster_whisper import WhisperModel
#         #
#         # self.model = WhisperModel(
#         #     self.model_path,
#         #     device=ASR_DEVICE,
#         #     compute_type="float16" if ASR_DEVICE == "cuda" else "int8"
#         # )
# 
#         logger.warning("Whisper provider is not connected yet.")
# 
#     def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
#         raise HTTPException(
#             status_code=501,
#             detail=(
#                 "Real Whisper ASR is not connected yet. "
#                 "Use ASR_ENGINE=mock or ASR_ENGINE=speechbrain for now."
#             ),
#         )
# 
# 
# # ============================================================
# # SpeechBrain Amharic Provider
# # ============================================================
# 
# class SpeechBrainASRProvider(BaseASRProvider):
#     engine_name = "speechbrain"
# 
#     def __init__(
#         self,
#         source: str = "speechbrain/asr-wav2vec2-dvoice-amharic",
#     ):
#         try:
#             from speechbrain.inference.ASR import EncoderASR
#         except ImportError as e:
#             raise ImportError(
#                 "speechbrain is not installed. "
#                 "Add speechbrain, torch, torchaudio, and soundfile to requirements.txt."
#             ) from e
# 
#         self.source = source
# 
#         savedir = MODEL_DIR / source.split("/")[-1]
#         savedir.mkdir(parents=True, exist_ok=True)
# 
#         logger.info("Loading SpeechBrain model from: %s", source)
#         logger.info("Saving/loading model files at: %s", savedir)
#         logger.info("Using device: %s", ASR_DEVICE)
# 
#         self.model = EncoderASR.from_hparams(
#             source=source,
#             savedir=str(savedir),
#             run_opts={"device": ASR_DEVICE},
#         )
# 
#         logger.info("SpeechBrain model loaded successfully.")
# 
#     def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
#         path = Path(audio_path)
# 
#         if not path.exists():
#             raise FileNotFoundError(f"Audio file not found: {audio_path}")
# 
#         if not path.is_file():
#             raise ValueError(f"Audio path is not a file: {audio_path}")
# 
#         audio_id = path.stem
# 
#         logger.info("Transcribing audio file: %s", audio_path)
# 
#         transcript = self.model.transcribe_file(str(path))
# 
#         if isinstance(transcript, list):
#             transcript = " ".join(str(x) for x in transcript)
# 
#         transcript = str(transcript).strip()
# 
#         return ASRResponse(
#             transcript=transcript,
#             language=language,
#             confidence=1.0,
#             engine=self.engine_name,
#             audio_id=audio_id,
#         )
# 
# 
# # ============================================================
# # Provider Factory
# # ============================================================
# 
# def create_asr_provider() -> BaseASRProvider:
#     logger.info("Initializing ASR provider: %s", ASR_ENGINE)
# 
#     if ASR_ENGINE == "mock":
#         return MockASRProvider()
# 
#     if ASR_ENGINE == "whisper_local":
#         return WhisperLocalASRProvider(str(MODEL_DIR))
# 
#     if ASR_ENGINE == "speechbrain":
#         return SpeechBrainASRProvider()
# 
#     raise ValueError(
#         f"Unsupported ASR_ENGINE='{ASR_ENGINE}'. "
#         "Use one of: mock, whisper_local, speechbrain."
#     )
# 
# 
# # ============================================================
# # FastAPI App
# # ============================================================
# 
# app = FastAPI(title="Amharic ASR Service")
# 
# asr_provider: Optional[BaseASRProvider] = None
# startup_error: Optional[str] = None
# 
# 
# @app.on_event("startup")
# def startup_event():
#     global asr_provider, startup_error
# 
#     try:
#         asr_provider = create_asr_provider()
#         startup_error = None
#         logger.info("ASR service started successfully.")
# 
#     except Exception as e:
#         asr_provider = None
#         startup_error = str(e)
#         logger.exception("ASR provider failed to start.")
# 
# 
# def get_provider() -> BaseASRProvider:
#     if asr_provider is None:
#         raise HTTPException(
#             status_code=503,
#             detail={
#                 "message": "ASR provider is not available.",
#                 "engine": ASR_ENGINE,
#                 "error": startup_error,
#             },
#         )
# 
#     return asr_provider
# 
# 
# # ============================================================
# # Health Check
# # ============================================================
# 
# @app.get("/health", response_model=HealthResponse)
# def health_check():
#     return HealthResponse(
#         status="ok" if asr_provider is not None else "degraded",
#         service="asr-service",
#         engine=ASR_ENGINE,
#         model_path=str(MODEL_DIR),
#         shared_utterances_dir=str(SHARED_UTTERANCES_DIR),
#         provider_loaded=asr_provider is not None,
#         error=startup_error,
#     )
# 
# 
# # ============================================================
# # Shared-volume endpoint used by VAD
# # ============================================================
# 
# @app.post("/transcribe-file", response_model=ASRResponse)
# async def transcribe_file(request: FileTranscribeRequest):
#     """
#     VAD sends only a filename.
# 
#     Example request:
#     {
#         "filename": "utterance_001.wav",
#         "language": "am"
#     }
# 
#     ASR reads the file from:
#         /shared/utterances/utterance_001.wav
#     """
# 
#     provider = get_provider()
# 
#     safe_filename = Path(request.filename).name
#     audio_path = SHARED_UTTERANCES_DIR / safe_filename
# 
#     if not audio_path.exists():
#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 "Audio file not found in shared utterance volume: "
#                 f"{safe_filename}. Expected path: {audio_path}"
#             ),
#         )
# 
#     if not audio_path.is_file():
#         raise HTTPException(
#             status_code=400,
#             detail=f"Path is not a file: {safe_filename}",
#         )
# 
#     try:
#         result = provider.transcribe(
#             audio_path=str(audio_path),
#             language=request.language,
#         )
#         return result
# 
#     except HTTPException:
#         raise
# 
#     except Exception as e:
#         logger.exception("ASR transcription failed.")
#         raise HTTPException(status_code=500, detail=str(e))
# 
# 
# # ============================================================
# # Manual Upload Endpoint
# # For testing only. VAD should use /transcribe-file.
# # ============================================================
# 
# @app.post("/transcribe", response_model=ASRResponse)
# async def transcribe_audio(
#     audio: UploadFile = File(...),
#     language: str = Form("am"),
# ):
#     provider = get_provider()
# 
#     if not audio.filename:
#         raise HTTPException(
#             status_code=400,
#             detail="No audio file uploaded.",
#         )
# 
#     audio_id = str(uuid.uuid4())
#     suffix = Path(audio.filename).suffix or ".wav"
#     audio_path = UPLOAD_DIR / f"{audio_id}{suffix}"
# 
#     try:
#         with audio_path.open("wb") as buffer:
#             shutil.copyfileobj(audio.file, buffer)
# 
#         result = provider.transcribe(
#             audio_path=str(audio_path),
#             language=language,
#         )
# 
#         return result
# 
#     except HTTPException:
#         raise
# 
#     except Exception as e:
#         logger.exception("ASR upload transcription failed.")
#         raise HTTPException(status_code=500, detail=str(e))
# 
#     finally:
#         await audio.close()

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
        # Files from VAD are already 16kHz Mono, so we skip the preparation step
        # and transcribe them directly from the shared volume.
        result = asr_engine.transcribe(audio_path)
        return result


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))