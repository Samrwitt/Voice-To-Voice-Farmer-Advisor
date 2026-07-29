"""Resolve farmer VAD utterance WAV files for expert callbacks."""

from __future__ import annotations

import os
from pathlib import Path


def farmer_utterances_dir() -> Path:
    return Path(os.getenv("FARMER_UTTERANCES_DIR", "/app/utterances"))


def resolve_farmer_utterance_audio(
    session_id: str | None,
    *,
    basename: str | None = None,
) -> str | None:
    """
    Return a playable WAV path for the farmer's question clip.

    Prefer an explicit basename (stored on the escalation). Otherwise use the
    latest ``{session_id}_utterance_*.wav`` in the shared utterances volume.
    """
    session_id = (session_id or "").strip()
    basename = (basename or "").strip()
    root = farmer_utterances_dir()

    if basename:
        candidate = root / Path(basename).name
        if candidate.is_file():
            return str(candidate)

    if not session_id:
        return None

    matches = sorted(root.glob(f"{session_id}_utterance_*.wav"))
    if not matches:
        return None
    return str(matches[-1])
