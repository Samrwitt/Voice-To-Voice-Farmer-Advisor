import logging
import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

from config import (
    ASR_ENGINE,
    MODEL_DIR,
    DEVICE,
    COMPUTE_TYPE,
    LANGUAGE,
    TASK,
    BEAM_SIZE,
    MAX_NEW_TOKENS,
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

        logger.info("Loading Whisper ASR from: %s (device=%s)", MODEL_DIR, DEVICE)
        self.model = WhisperModel(
            str(model_path),
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        logger.info("Whisper ASR loaded successfully.")

    def transcribe(self, audio_path: str | Path) -> dict:
        start_time = time.time()
        logger.info("Whisper transcribing: %s", audio_path)

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            temperature=0.0,
            initial_prompt=_whisper_initial_prompt(),
            vad_filter=USE_VAD,
            vad_parameters=dict(threshold=0.6, min_speech_duration_ms=400),
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

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
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        first = next((part.strip() for part in raw.split(",") if part.strip()), "")
        if first:
            return first
    return ""


class GeminiASREngine:
    engine_name = "gemini_audio"

    def __init__(self):
        self.model = GEMINI_ASR_MODEL
        self.api_key = _first_env_key(
            "ASR_GEMINI_API_KEY",
            "FREE_GEMINI_API_KEYS",
            "FREE_GEMINI_API_KEY",
            "GEMINI_API_KEYS",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GENAI_API_KEY",
        )
        if not self.api_key:
            raise RuntimeError(
                "Gemini ASR needs ASR_GEMINI_API_KEY, GEMINI_API_KEY, GEMINI_API_KEYS, "
                "FREE_GEMINI_API_KEYS, GOOGLE_API_KEY, or GENAI_API_KEY."
            )
        logger.info("Gemini ASR configured with model=%s", self.model)

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
        resp = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=GEMINI_ASR_TIMEOUT_SEC,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini ASR HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
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
