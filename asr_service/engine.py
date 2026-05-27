import logging
import base64
import mimetypes
import os
import time
import wave
from pathlib import Path
from typing import Any

import requests

from config import (
    ASR_ENGINE,
    MODEL_DIR,
    DEVICE,
    COMPUTE_TYPE,
    CPU_COMPUTE_TYPE,
    GPU_COMPUTE_TYPE,
    LANGUAGE,
    TASK,
    BEAM_SIZE,
    MAX_NEW_TOKENS,
    ASR_MAX_NEW_TOKENS_CAP,
    ASR_MAX_NEW_TOKENS_DYNAMIC,
    REPETITION_PENALTY,
    NO_REPEAT_NGRAM_SIZE,
    USE_VAD,
    CONDITION_ON_PREVIOUS_TEXT,
    GEMINI_ASR_MODEL,
    GEMINI_ASR_TIMEOUT_SEC,
    ASR_INITIAL_PROMPT,
    ASR_USE_DOMAIN_INITIAL_PROMPT,
    ASR_INITIAL_PROMPT_MAX_TERMS,
)
from postprocess import postprocess_asr_transcript

logger = logging.getLogger("asr-engine")
logging.basicConfig(level=logging.INFO)

def _cuda_device_count() -> int:
    try:
        import ctranslate2 as ct2  # faster-whisper uses CT2 under the hood

        return int(getattr(ct2, "get_cuda_device_count", lambda: 0)() or 0)
    except Exception:
        return 0


def _resolve_whisper_device_and_compute_type() -> tuple[str, str]:
    """
    Supports ASR_DEVICE=auto:
    - if CUDA is available -> cuda + float16
    - else -> cpu + int8
    """
    requested = (DEVICE or "").strip().lower()
    gpu_requested = requested in ("auto", "cuda", "cuda_if_available", "gpu", "gpu_if_available")
    if gpu_requested:
        if _cuda_device_count() > 0:
            return "cuda", GPU_COMPUTE_TYPE
        return "cpu", CPU_COMPUTE_TYPE

    # Explicit device overrides everything.
    if requested:
        if requested not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported ASR_DEVICE={DEVICE!r}. Use auto/cpu/cuda.")
        # When explicitly setting device, keep the legacy ASR_COMPUTE_TYPE behavior.
        return requested, COMPUTE_TYPE
    return "cpu", CPU_COMPUTE_TYPE


def _audio_duration_sec(audio_path: str | Path) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


WHISPER_CONTEXT_TOKENS = 448


def _prompt_token_reserve() -> int:
    """Tokens consumed by initial_prompt; must stay under Whisper context (448)."""
    if not _whisper_initial_prompt():
        return 16
    return int(os.getenv("ASR_INITIAL_PROMPT_TOKEN_RESERVE", "232") or "232")


def _max_new_tokens_for_audio(audio_path: str | Path) -> int:
    """Scale decode budget with clip length — Amharic needs more tokens than English."""
    ceiling = WHISPER_CONTEXT_TOKENS - _prompt_token_reserve()
    ceiling = min(ceiling, ASR_MAX_NEW_TOKENS_CAP)
    base = min(MAX_NEW_TOKENS, ceiling)
    if not ASR_MAX_NEW_TOKENS_DYNAMIC:
        return max(96, base)
    duration = _audio_duration_sec(audio_path)
    if duration is None or duration <= 0:
        return max(96, base)
    # ~30–40 tokens/sec works for our small Amharic CT2 model on farmer speech.
    scaled = int(duration * 36) + 64
    return max(96, min(ceiling, max(base, scaled)))


def _whisper_initial_prompt() -> str | None:
    if ASR_INITIAL_PROMPT:
        return ASR_INITIAL_PROMPT
    if not ASR_USE_DOMAIN_INITIAL_PROMPT:
        return None
    try:
        from domain_terms import get_asr_vocabulary

        terms = get_asr_vocabulary()[: max(1, ASR_INITIAL_PROMPT_MAX_TERMS)]
    except Exception:
        terms = []
    if not terms:
        return None
    return "እነዚህ የግብርና ቃላት ሊሰሙ ይችላሉ፦ " + "፣ ".join(terms)


def _format_transcription_result(
    *,
    audio_path: str | Path,
    raw_transcript: str,
    engine: str,
    language: str = LANGUAGE,
    language_probability: float = 0.9,
    segments: list[dict[str, Any]] | None = None,
    latency: float | None = None,
) -> dict:
    processed = postprocess_asr_transcript(
        raw_transcript,
        acoustic_confidence=float(language_probability),
    )
    if latency is None:
        latency = 0.0
    if segments is None:
        segments = (
            [{"start": 0.0, "end": float(latency), "text": raw_transcript}]
            if raw_transcript
            else []
        )
    return {
        "language": language,
        "language_probability": float(language_probability),
        "raw_transcript": processed["raw"],
        "cleaned_transcript": processed["cleaned"],
        "homophone_normalized_transcript": processed["homophone_normalized"],
        "pronunciation_normalized_transcript": processed["pronunciation_normalized"],
        "domain_corrected_transcript": processed["domain_corrected"],
        "semantic_corrected_transcript": processed.get("semantic_corrected"),
        "transcript_fix_backend": processed.get("transcript_fix_backend"),
        "final_transcript": processed["final"],
        "structured_transcript": processed["structured_transcript"],
        "transcript": processed["final"],
        "text": processed["final"],
        "confidence": processed["confidence"],
        "acoustic_confidence": float(language_probability),
        "fuzzy": processed.get("fuzzy") or {},
        "engine": engine,
        "audio_id": Path(audio_path).stem,
        "unusual_words": processed["unusual_words"],
        "needs_confirmation": processed["needs_confirmation"],
        "confirmation_prompt": processed["confirmation_prompt"],
        "segments": segments,
        "latency_seconds": latency,
    }


class WhisperASREngine:
    engine_name = "whisper_local"

    def __init__(self):
        from faster_whisper import WhisperModel

        model_path = Path(MODEL_DIR)
        if not model_path.exists():
            raise FileNotFoundError(f"ASR model folder not found: {MODEL_DIR}")

        device, compute_type = _resolve_whisper_device_and_compute_type()
        logger.info(
            "Loading Whisper ASR from: %s (device=%s compute_type=%s)",
            MODEL_DIR,
            device,
            compute_type,
        )
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper ASR loaded on %s (%s)", device, compute_type)

    def transcribe(self, audio_path: str | Path) -> dict:
        start_time = time.time()
        logger.info("Whisper transcribing: %s", audio_path)

        max_tokens = _max_new_tokens_for_audio(audio_path)
        logger.info("Whisper max_new_tokens=%s for %s", max_tokens, audio_path)

        # Safety retry: if prompt+max_new_tokens exceed Whisper context, shrink
        # the budget and retry once or twice (prevents 500s during eval).
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                segments_iter, info = self.model.transcribe(
                    str(audio_path),
                    language=LANGUAGE,
                    task=TASK,
                    beam_size=BEAM_SIZE,
                    # Allow fallback temperatures so Whisper retries on greedy failures
                    # instead of producing silence or repetition loops.
                    temperature=[0.0, 0.2, 0.4],
                    initial_prompt=_whisper_initial_prompt(),
                    vad_filter=USE_VAD,
                    # Silero inside faster-whisper: strips long silence *between* speech chunks.
                    # For full saved utterances, pauses between clauses should stay in the WAV.
                    vad_parameters=dict(
                        threshold=0.35,
                        min_speech_duration_ms=200,
                        min_silence_duration_ms=500,
                        speech_pad_ms=300,
                    ),
                    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
                    max_new_tokens=max_tokens,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "combined length of the prompt" in msg and "max_length" in msg:
                    max_tokens = max(96, int(max_tokens * 0.85) - 16)
                    logger.warning(
                        "Whisper context overflow; retry attempt=%s new_max_new_tokens=%s",
                        attempt,
                        max_tokens,
                    )
                    continue
                raise
        if last_exc is not None:
            raise last_exc

        segments = []
        texts = []
        for segment in segments_iter:
            segment_text = segment.text.strip()
            texts.append(segment_text)
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment_text,
                }
            )

        raw_transcript = " ".join(texts).strip()
        latency = time.time() - start_time
        logger.info("Whisper done in %.2fs: %s", latency, raw_transcript[:120])

        return _format_transcription_result(
            audio_path=audio_path,
            raw_transcript=raw_transcript,
            engine=self.engine_name,
            language=info.language,
            language_probability=float(info.language_probability),
            segments=segments,
            latency=latency,
        )


def _first_env_key(*names: str) -> str:
    keys = _all_gemini_api_keys(*names)
    return keys[0] if keys else ""


def _all_gemini_api_keys(*names: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        for part in raw.replace(";", ",").split(","):
            k = part.strip().strip('"').strip("'")
            if k and k not in seen:
                seen.add(k)
                out.append(k)
    return out


class GeminiASREngine:
    engine_name = "gemini_audio"

    def __init__(self):
        from gemini_keys import paid_gemini_keys_for_asr_audio

        self.model = GEMINI_ASR_MODEL
        self.api_keys = paid_gemini_keys_for_asr_audio()
        if not self.api_keys:
            raise RuntimeError(
                "Gemini audio ASR needs a paid key: ASR_GEMINI_API_KEY, GEMINI_API_KEY, "
                "GEMINI_API_KEY_BACKUP, or GOOGLE_API_KEY (free keys are for text fix only)."
            )
        self.api_key = self.api_keys[0]
        logger.info(
            "Gemini ASR configured with model=%s (%s key(s))",
            self.model,
            len(self.api_keys),
        )

    def _mime_type(self, audio_path: str | Path) -> str:
        guessed, _ = mimetypes.guess_type(str(audio_path))
        if guessed and guessed.startswith("audio/"):
            return guessed
        suffix = Path(audio_path).suffix.lower()
        if suffix == ".wav":
            return "audio/wav"
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".webm":
            return "audio/webm"
        return "audio/wav"

    def transcribe(self, audio_path: str | Path) -> dict:
        start_time = time.time()
        logger.info("Gemini transcribing: %s", audio_path)

        audio_path = Path(audio_path)
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        mime_type = self._mime_type(audio_path)
        prompt = (
            "Transcribe this Amharic farmer phone-call utterance exactly. "
            "Return only the transcript text in Amharic. "
            "Do not translate, summarize, explain, add punctuation commentary, or add labels. "
            "Do not truncate the transcript. If a word is unclear, keep the rest of the transcript."
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": audio_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "topP": 0.1,
                "maxOutputTokens": int(os.getenv("ASR_GEMINI_MAX_OUTPUT_TOKENS", "512")),
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = None
        last_error = ""
        for key_idx, api_key in enumerate(self.api_keys):
            resp = requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=GEMINI_ASR_TIMEOUT_SEC,
            )
            if resp.status_code < 400:
                data = resp.json()
                self.api_key = api_key
                break
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code in (429, 503) and key_idx + 1 < len(self.api_keys):
                logger.warning(
                    "Gemini ASR key %s/%s rate-limited (%s), trying next key",
                    key_idx + 1,
                    len(self.api_keys),
                    resp.status_code,
                )
                continue
            raise RuntimeError(f"Gemini ASR {last_error}")
        if data is None:
            raise RuntimeError(f"Gemini ASR exhausted keys: {last_error}")
        parts = (
            ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")
            or []
        )
        text_parts = [str(p.get("text") or "").strip() for p in parts if p.get("text")]
        raw_transcript = " ".join(p for p in text_parts if p).strip()
        raw_transcript = raw_transcript.strip("` \n\t")
        if raw_transcript.lower().startswith("transcript:"):
            raw_transcript = raw_transcript.split(":", 1)[1].strip()

        latency = time.time() - start_time
        logger.info("Gemini ASR done in %.2fs: %s", latency, raw_transcript[:120])
        return _format_transcription_result(
            audio_path=audio_path,
            raw_transcript=raw_transcript,
            engine=self.engine_name,
            language=LANGUAGE,
            language_probability=1.0 if raw_transcript else 0.0,
            segments=[{"start": 0.0, "end": float(latency), "text": raw_transcript}]
            if raw_transcript
            else [],
            latency=latency,
        )


def create_asr_engine():
    mode = ASR_ENGINE
    if mode in ("whisper", "whisper_local"):
        return WhisperASREngine()
    if mode in ("gemini", "gemini_audio", "gemini_asr"):
        return GeminiASREngine()
    raise ValueError(
        f"Unsupported ASR_ENGINE='{mode}'. Use whisper_local or gemini."
    )


# Backward-compatible alias used by main.py
class ASREngine:
    def __new__(cls):
        return create_asr_engine()
