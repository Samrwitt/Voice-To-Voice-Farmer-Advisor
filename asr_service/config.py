import os
from pathlib import Path

# Local Whisper CT2 ASR.
ASR_ENGINE = os.getenv("ASR_ENGINE", "whisper_local").strip().lower()

MODEL_DIR = os.getenv(
    "ASR_MODEL_PATH",
    "./models/asr/whisper-small-amharic-bdu-8khz-aug-ct2-fp16",
)

SHARED_UTTERANCES_DIR = os.getenv(
    "SHARED_UTTERANCES_DIR",
    "/shared/utterances",
)

DEVICE = os.getenv("ASR_DEVICE", "auto")
# Compute types:
# - CPU: typically int8 is faster and good enough for this CT2 model.
# - GPU: float16 is typical for speed/quality balance.
CPU_COMPUTE_TYPE = os.getenv("ASR_CPU_COMPUTE_TYPE", "int8")
GPU_COMPUTE_TYPE = os.getenv("ASR_GPU_COMPUTE_TYPE", "float16")

# Backwards-compatible single value (used only when ASR_DEVICE is explicitly set).
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")


LANGUAGE = os.getenv("ASR_LANGUAGE", "am")
TASK = os.getenv("ASR_TASK", "transcribe")

GEMINI_ASR_MODEL = os.getenv("ASR_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ASR_TIMEOUT_SEC = float(os.getenv("ASR_GEMINI_TIMEOUT_SEC", "60"))

ASR_INITIAL_PROMPT = os.getenv("ASR_INITIAL_PROMPT", "").strip()
ASR_USE_DOMAIN_INITIAL_PROMPT = os.getenv("ASR_USE_DOMAIN_INITIAL_PROMPT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ASR_INITIAL_PROMPT_MAX_TERMS = int(os.getenv("ASR_INITIAL_PROMPT_MAX_TERMS", "24"))

BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))
# Amharic uses many subword tokens per word; 160 often stops after one clause (~7 words).
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "384"))
ASR_MAX_NEW_TOKENS_DYNAMIC = os.getenv("ASR_MAX_NEW_TOKENS_DYNAMIC", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ASR_MAX_NEW_TOKENS_CAP = int(os.getenv("ASR_MAX_NEW_TOKENS_CAP", "448"))

REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.2"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("ASR_NO_REPEAT_NGRAM_SIZE", "3"))

USE_VAD = os.getenv("ASR_USE_VAD", "true").lower() == "true"
CONDITION_ON_PREVIOUS_TEXT = False

TARGET_SR = 16000

TMP_DIR = Path(os.getenv("ASR_TMP_DIR", "/tmp/asr_service"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Post-ASR semantic / typo correction
# Hosted Groq/Gemini token usage is intentionally disabled for now.
# To re-enable later, set ASR_HOSTED_LLM_FIX=auto or ASR_HOSTED_LLM_FIX=1
# and uncomment the hosted block in postprocess._apply_semantic_correction().
USE_HOSTED_LLM_FIX = os.getenv("ASR_HOSTED_LLM_FIX", "0").strip().lower()
