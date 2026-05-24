import logging
import time
from pathlib import Path

from faster_whisper import WhisperModel

from config import (
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


class ASREngine:
    def __init__(self):
        model_path = Path(MODEL_DIR)

        if not model_path.exists():
            logger.error(f"ASR model folder not found: {MODEL_DIR}")
            raise FileNotFoundError(f"ASR model folder not found: {MODEL_DIR}")

        logger.info(f"Loading ASR model from: {MODEL_DIR}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Compute type: {COMPUTE_TYPE}")

        self.model = WhisperModel(
            str(model_path),
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

        logger.info("ASR model loaded successfully.")


    def transcribe(self, audio_path: str | Path) -> dict:
        start_time = time.time()
        logger.info(f"Transcribing audio file: {audio_path}")

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
        conf_parts: list[float] = []

        for segment in segments_iter:
            segment_text = segment.text.strip()
            texts.append(segment_text)

            # Estimate transcription confidence from per-segment metrics when available.
            # faster-whisper segments commonly expose avg_logprob / no_speech_prob.
            try:
                avg_logprob = getattr(segment, "avg_logprob", None)
                no_speech_prob = getattr(segment, "no_speech_prob", None)
                if isinstance(avg_logprob, (int, float)):
                    # avg_logprob is typically <= 0; exp() maps to (0, 1].
                    import math

                    p = math.exp(float(avg_logprob))
                    if isinstance(no_speech_prob, (int, float)):
                        p = p * (1.0 - float(no_speech_prob))
                    conf_parts.append(max(0.0, min(1.0, float(p))))
            except Exception:
                pass

            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment_text,
                }
            )

        raw_transcript = " ".join(texts).strip()
        processed = postprocess_asr_transcript(raw_transcript)

        latency = time.time() - start_time
        logger.info(f"Transcription completed in {latency:.2f}s. Result: {processed['final']}")

        # Final confidence: prefer segment-derived confidence; fall back to language_probability.
        try:
            seg_conf = (sum(conf_parts) / len(conf_parts)) if conf_parts else None
        except Exception:
            seg_conf = None
        lang_prob = float(info.language_probability)
        if seg_conf is None:
            final_conf = lang_prob
        else:
            # Combine language ID confidence with acoustic/token confidence.
            final_conf = max(0.0, min(1.0, 0.35 * lang_prob + 0.65 * float(seg_conf)))

        return {

            "language": info.language,
            "language_probability": float(info.language_probability),

            "raw_transcript": processed["raw"],
            "cleaned_transcript": processed["cleaned"],
            "homophone_normalized_transcript": processed["homophone_normalized"],
            "pronunciation_normalized_transcript": processed["pronunciation_normalized"],
            "domain_corrected_transcript": processed["domain_corrected"],
            "final_transcript": processed["final"],
            "transcript": processed["final"],
            "text": processed["final"],
            "confidence": float(final_conf),
            "engine": "whisper_local",
            "audio_id": Path(audio_path).stem,



            "unusual_words": processed["unusual_words"],
            "needs_confirmation": processed["needs_confirmation"],
            "confirmation_prompt": processed["confirmation_prompt"],

            "segments": segments,
            "latency_seconds": latency,
        }