import os
from pathlib import Path

# whisper_local | speechbrain
ASR_ENGINE = os.getenv("ASR_ENGINE", "whisper_local").strip().lower()

MODEL_DIR = os.getenv(
    "ASR_MODEL_PATH",
    "./models/asr/whisper-small-amharic-bdu-8khz-aug-ct2-fp16",
)

SPEECHBRAIN_SOURCE = os.getenv(
    "ASR_SPEECHBRAIN_SOURCE",
    "speechbrain/asr-wav2vec2-dvoice-amharic",
).strip()

SPEECHBRAIN_SAVEDIR = Path(
    os.getenv(
        "ASR_SPEECHBRAIN_SAVEDIR",
        "/opt/asr-models/speechbrain-wav2vec2-dvoice-amharic",
    )
)

SHARED_UTTERANCES_DIR = os.getenv(
    "SHARED_UTTERANCES_DIR",
    "/shared/utterances",
)

DEVICE = os.getenv("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")


LANGUAGE = os.getenv("ASR_LANGUAGE", "am")
TASK = os.getenv("ASR_TASK", "transcribe")

BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "160"))

REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.2"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("ASR_NO_REPEAT_NGRAM_SIZE", "3"))

USE_VAD = os.getenv("ASR_USE_VAD", "true").lower() == "true"
CONDITION_ON_PREVIOUS_TEXT = False

TARGET_SR = 16000

TMP_DIR = Path(os.getenv("ASR_TMP_DIR", "/tmp/asr_service"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Post-ASR semantic / typo correction
# Hosted Groq/Gemini (preferred when keys exist): ASR_HOSTED_LLM_FIX=auto
USE_HOSTED_LLM_FIX = os.getenv("ASR_HOSTED_LLM_FIX", "auto").strip().lower()

# Ollama fallback when hosted fix is off or fails
USE_OLLAMA = os.getenv("ASR_USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")