import os
from pathlib import Path

import httpx


ASR_HTTP_URL = os.getenv(
    "ASR_HTTP_URL",
    "http://asr-service:8001/transcribe-file"
)


async def transcribe_utterance_file(
    utterance_path: str,
    language: str = "am"
) -> dict:
    """
    VAD saves the utterance in its own path, for example:
        /app/utterances/utt_001.wav

    ASR sees the same Docker volume at:
        /shared/utterances/utt_001.wav

    Therefore, VAD only sends the filename to ASR.
    """

    filename = Path(utterance_path).name

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            ASR_HTTP_URL,
            json={
                "filename": filename,
                "language": language
            }
        )

        response.raise_for_status()
        return response.json()