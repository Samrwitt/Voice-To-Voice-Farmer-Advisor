from pydantic import BaseModel
from typing import Optional, List


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class ASRResponse(BaseModel):
    language: str
    language_probability: float

    raw_transcript: str
    cleaned_transcript: str
    homophone_normalized_transcript: str
    pronunciation_normalized_transcript: str
    domain_corrected_transcript: str
    final_transcript: str
    transcript: str
    text: str
    confidence: float

    engine: str
    audio_id: str


    unusual_words: List[str]
    needs_confirmation: bool
    confirmation_prompt: Optional[str]

    segments: List[SegmentOut]
    latency_seconds: float


class FileTranscribeRequest(BaseModel):
    filename: str
    language: str = "am"