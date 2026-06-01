import logging
import time
from pathlib import Path
from typing import Any

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
)
from postprocess import postprocess_asr_transcript

logger = logging.getLogger("asr-engine")
logging.basicConfig(level=logging.INFO)


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


def create_asr_engine():
    mode = ASR_ENGINE
    if mode in ("whisper", "whisper_local"):
        return WhisperASREngine()
    raise ValueError(
        f"Unsupported ASR_ENGINE='{mode}'. Use whisper_local."
    )


# Backward-compatible alias used by main.py
class ASREngine:
    def __new__(cls):
        return create_asr_engine()
