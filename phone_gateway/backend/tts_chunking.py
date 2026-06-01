import os
import re


SIP_TTS_CHUNK_MAX_CHARS = int(os.getenv("SIP_TTS_CHUNK_MAX_CHARS", "140"))


def _split_long_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_tts_text(text: str, max_chars: int | None = None) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if max_chars is None:
        max_chars = SIP_TTS_CHUNK_MAX_CHARS
    max_chars = max(60, int(max_chars or 140))
    pieces = [
        piece.strip()
        for piece in re.findall(r".+?(?:[።!?፤፣,;]|$)", normalized)
        if piece.strip()
    ]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        subpieces = [piece] if len(piece) <= max_chars else _split_long_words(piece, max_chars)
        for subpiece in subpieces:
            candidate = f"{current} {subpiece}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = subpiece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks
