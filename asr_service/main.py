import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("asr-service")


# ============================================================
# Configuration
# ============================================================

ASR_ENGINE = os.getenv("ASR_ENGINE", "mock").lower()
# Supported:
#   mock
#   whisper_local
#   speechbrain

MODEL_PATH = os.getenv("ASR_MODEL_PATH", "/models")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_audio"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(MODEL_PATH)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SHARED_UTTERANCES_DIR = Path(
    os.getenv("SHARED_UTTERANCES_DIR", "/shared/utterances")
)
SHARED_UTTERANCES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Schemas
# ============================================================

class ASRResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    engine: str
    audio_id: str


class FileTranscribeRequest(BaseModel):
    filename: str
    language: str = "am"


class HealthResponse(BaseModel):
    status: str
    service: str
    engine: str
    model_path: str
    shared_utterances_dir: str
    provider_loaded: bool
    error: Optional[str] = None


# ============================================================
# ASR Provider Interface
# ============================================================

class BaseASRProvider:
    engine_name = "base"

    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        raise NotImplementedError


# ============================================================
# Mock ASR Provider
# ============================================================

class MockASRProvider(BaseASRProvider):
    engine_name = "mock"

    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        audio_id = Path(audio_path).stem

        return ASRResponse(
            transcript="የበቆሎ ተባይ እንዴት መከላከል እችላለሁ?",
            language=language,
            confidence=0.92,
            engine=self.engine_name,
            audio_id=audio_id,
        )


# ============================================================
# Whisper Local Provider Placeholder
# ============================================================

class WhisperLocalASRProvider(BaseASRProvider):
    engine_name = "whisper_local"

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError(
                "ASR_MODEL_PATH is required when ASR_ENGINE=whisper_local"
            )

        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        logger.info("Loading Whisper model from: %s", self.model_path)

        # Later, when your faster-whisper model is ready:
        #
        # from faster_whisper import WhisperModel
        #
        # self.model = WhisperModel(
        #     self.model_path,
        #     device=ASR_DEVICE,
        #     compute_type="float16" if ASR_DEVICE == "cuda" else "int8"
        # )

        logger.warning("Whisper provider is not connected yet.")

    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        raise HTTPException(
            status_code=501,
            detail=(
                "Real Whisper ASR is not connected yet. "
                "Use ASR_ENGINE=mock or ASR_ENGINE=speechbrain for now."
            ),
        )


# ============================================================
# SpeechBrain Amharic Provider
# ============================================================

class SpeechBrainASRProvider(BaseASRProvider):
    engine_name = "speechbrain"

    def __init__(
        self,
        source: str = "speechbrain/asr-wav2vec2-dvoice-amharic",
    ):
        try:
            from speechbrain.inference.ASR import EncoderASR
        except ImportError as e:
            raise ImportError(
                "speechbrain is not installed. "
                "Add speechbrain, torch, torchaudio, and soundfile to requirements.txt."
            ) from e

        self.source = source

        savedir = MODEL_DIR / source.split("/")[-1]
        savedir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading SpeechBrain model from: %s", source)
        logger.info("Saving/loading model files at: %s", savedir)
        logger.info("Using device: %s", ASR_DEVICE)

        self.model = EncoderASR.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": ASR_DEVICE},
        )

        logger.info("SpeechBrain model loaded successfully.")

    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")

        audio_id = path.stem

        logger.info("Transcribing audio file: %s", audio_path)

        transcript = self.model.transcribe_file(str(path))

        if isinstance(transcript, list):
            transcript = " ".join(str(x) for x in transcript)

        transcript = str(transcript).strip()

        return ASRResponse(
            transcript=transcript,
            language=language,
            confidence=1.0,
            engine=self.engine_name,
            audio_id=audio_id,
        )


# ============================================================
# Provider Factory
# ============================================================

def create_asr_provider() -> BaseASRProvider:
    logger.info("Initializing ASR provider: %s", ASR_ENGINE)

    if ASR_ENGINE == "mock":
        return MockASRProvider()

    if ASR_ENGINE == "whisper_local":
        return WhisperLocalASRProvider(str(MODEL_DIR))

    if ASR_ENGINE == "speechbrain":
        return SpeechBrainASRProvider()

    raise ValueError(
        f"Unsupported ASR_ENGINE='{ASR_ENGINE}'. "
        "Use one of: mock, whisper_local, speechbrain."
    )


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Amharic ASR Service")

asr_provider: Optional[BaseASRProvider] = None
startup_error: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global asr_provider, startup_error

    try:
        asr_provider = create_asr_provider()
        startup_error = None
        logger.info("ASR service started successfully.")

    except Exception as e:
        asr_provider = None
        startup_error = str(e)
        logger.exception("ASR provider failed to start.")


def get_provider() -> BaseASRProvider:
    if asr_provider is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "ASR provider is not available.",
                "engine": ASR_ENGINE,
                "error": startup_error,
            },
        )

    return asr_provider


# ============================================================
# Health Check
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if asr_provider is not None else "degraded",
        service="asr-service",
        engine=ASR_ENGINE,
        model_path=str(MODEL_DIR),
        shared_utterances_dir=str(SHARED_UTTERANCES_DIR),
        provider_loaded=asr_provider is not None,
        error=startup_error,
    )


# ============================================================
# Shared-volume endpoint used by VAD
# ============================================================

@app.post("/transcribe-file", response_model=ASRResponse)
async def transcribe_file(request: FileTranscribeRequest):
    """
    VAD sends only a filename.

    Example request:
    {
        "filename": "utterance_001.wav",
        "language": "am"
    }

    ASR reads the file from:
        /shared/utterances/utterance_001.wav
    """

    provider = get_provider()

    safe_filename = Path(request.filename).name
    audio_path = SHARED_UTTERANCES_DIR / safe_filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Audio file not found in shared utterance volume: "
                f"{safe_filename}. Expected path: {audio_path}"
            ),
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file: {safe_filename}",
        )

    try:
        result = provider.transcribe(
            audio_path=str(audio_path),
            language=request.language,
        )
        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("ASR transcription failed.")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Manual Upload Endpoint
# For testing only. VAD should use /transcribe-file.
# ============================================================

@app.post("/transcribe", response_model=ASRResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("am"),
):
    provider = get_provider()

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file uploaded.",
        )

    audio_id = str(uuid.uuid4())
    suffix = Path(audio.filename).suffix or ".wav"
    audio_path = UPLOAD_DIR / f"{audio_id}{suffix}"

    try:
        with audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        result = provider.transcribe(
            audio_path=str(audio_path),
            language=language,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("ASR upload transcription failed.")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        await audio.close()