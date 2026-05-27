#!/usr/bin/env python3
"""Diagnose Whisper truncation: internal VAD vs full-audio decode."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv(
    "ASR_MODEL_PATH",
    str(ROOT / "models/asr/whisper-small-amharic-bdu-8khz-aug-ct2-fp16"),
)
DEVICE = os.getenv("ASR_DEVICE", "cpu")
COMPUTE = os.getenv("ASR_COMPUTE_TYPE", "int8")


def run(model: WhisperModel, path: Path, label: str, **kwargs) -> None:
    segs, _info = model.transcribe(str(path), language="am", beam_size=5, max_new_tokens=160, **kwargs)
    parts: list[str] = []
    print(f"\n--- {path.name} | {label} ---")
    for seg in segs:
        t = seg.text.strip()
        parts.append(t)
        print(f"  {seg.start:5.2f}-{seg.end:5.2f}s | {t}")
    joined = " ".join(parts).strip()
    print(f"  => {len(joined.split())} words: {joined}")


def main() -> int:
    clips = [
        ROOT / "eval/tts_outputs/safety_message.wav",
        ROOT / "eval/tts_outputs/long_voice_answer.wav",
    ]
    for p in clips:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    print(f"Loading model from {MODEL_PATH} ({DEVICE}/{COMPUTE})")
    model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE)

    for p in clips:
        run(model, p, "no_vad", vad_filter=False)
        run(
            model,
            p,
            "vad_default",
            vad_filter=True,
            vad_parameters=dict(threshold=0.35, min_speech_duration_ms=200),
        )
        run(
            model,
            p,
            "vad_split_pauses",
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.35,
                min_speech_duration_ms=200,
                min_silence_duration_ms=400,
                speech_pad_ms=300,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
