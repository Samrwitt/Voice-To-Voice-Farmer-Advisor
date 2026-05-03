import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

ASR_ENGINE = os.getenv("ASR_ENGINE", "mock")
# options:
#   mock
#   whisper_local

MODEL_PATH = os.getenv("ASR_MODEL_PATH", "")

# Used only for manual upload testing through /transcribe
UPLOAD_DIR = Path("uploaded_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

# Used by /transcribe-file.
# In Docker Compose, this points to the same named volume VAD writes into.
SHARED_UTTERANCES_DIR = Path(
    os.getenv("SHARED_UTTERANCES_DIR", "/shared/utterances")
)


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


# ============================================================
# ASR Provider Interface
# ============================================================

class BaseASRProvider:
    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        raise NotImplementedError


# ============================================================
# Mock ASR Provider
# ============================================================

class MockASRProvider(BaseASRProvider):
    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        audio_id = Path(audio_path).stem

        mock_transcript = "የበቆሎ ተባይ እንዴት መከላከል እችላለሁ?"

        return ASRResponse(
            transcript=mock_transcript,
            language=language,
            confidence=0.92,
            engine="mock",
            audio_id=audio_id
        )


# ============================================================
# Real Whisper Provider Placeholder
# ============================================================

class WhisperLocalASRProvider(BaseASRProvider):
    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError(
                "ASR_MODEL_PATH is required when ASR_ENGINE=whisper_local"
            )

        self.model_path = model_path
        self.model = None
        self.processor = None

        self.load_model()

    def load_model(self):
        """
        Later, when your model is ready, load it here.

        Example:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_path,
                device="cuda",
                compute_type="float16"
            )
        """

        print(f"Loading ASR model from: {self.model_path}", flush=True)

    def transcribe(self, audio_path: str, language: str = "am") -> ASRResponse:
        audio_id = Path(audio_path).stem

        # Later, replace this with real ASR inference.
        #
        # Example:
        # segments, info = self.model.transcribe(
        #     audio_path,
        #     language=language,
        #     beam_size=5
        # )
        # transcript = " ".join([segment.text for segment in segments]).strip()
        #
        # return ASRResponse(
        #     transcript=transcript,
        #     language=language,
        #     confidence=1.0,
        #     engine="whisper_local",
        #     audio_id=audio_id
        # )

        raise HTTPException(
            status_code=501,
            detail=(
                "Real Whisper ASR is not connected yet. "
                "Use ASR_ENGINE=mock for now."
            )
        )


# ============================================================
# Provider factory
# ============================================================

def get_asr_provider() -> BaseASRProvider:
    if ASR_ENGINE == "mock":
        return MockASRProvider()

    if ASR_ENGINE == "whisper_local":
        return WhisperLocalASRProvider(MODEL_PATH)

    raise ValueError(f"Unsupported ASR_ENGINE: {ASR_ENGINE}")


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Amharic ASR Service")

asr_provider = get_asr_provider()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "asr-service",
        "engine": ASR_ENGINE,
        "shared_utterances_dir": str(SHARED_UTTERANCES_DIR)
    }


# ============================================================
# Shared-volume endpoint used by VAD
# ============================================================

@app.post("/transcribe-file", response_model=ASRResponse)
async def transcribe_file(request: FileTranscribeRequest):
    """
    VAD sends only the filename.

    Example request:
        {
          "filename": "some_utterance.wav",
          "language": "am"
        }

    ASR reads:
        /shared/utterances/some_utterance.wav

    This avoids copying the audio twice.
    """

    safe_filename = Path(request.filename).name
    audio_path = SHARED_UTTERANCES_DIR / safe_filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Audio file not found in shared utterance volume: "
                f"{safe_filename}"
            )
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file: {safe_filename}"
        )

    try:
        result = asr_provider.transcribe(
            audio_path=str(audio_path),
            language=request.language
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Manual upload endpoint
# Keep this for testing only.
# VAD does not use this endpoint in the shared-volume design.
# ============================================================

@app.post("/transcribe", response_model=ASRResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("am")
):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded")

    audio_id = str(uuid.uuid4())
    suffix = Path(audio.filename).suffix or ".webm"
    audio_path = UPLOAD_DIR / f"{audio_id}{suffix}"

    try:
        with audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        result = asr_provider.transcribe(
            audio_path=str(audio_path),
            language=language
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))