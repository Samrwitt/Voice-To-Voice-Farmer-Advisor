"""Context shaping extracted from ``RAG/query.py`` (no Chroma dependency)."""

from __future__ import annotations

import os
import re
from typing import Any


def retrieval_query_for(question: str, conversation: list[dict] | None) -> str:
    """Build the string used for embedding search; folds in recent turns for follow-ups."""
    q = (question or "").strip()
    if not conversation:
        return q
    if os.environ.get("RAG_RETRIEVAL_USE_CONVERSATION", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return q
    parts: list[str] = []
    for m in conversation[-6:]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = (m.get("content") or "").strip()
        if not c:
            continue
        parts.append(f"{role}: {c[:900]}")
    if not parts:
        return q
    hist = "\n".join(parts)
    cap = int(os.environ.get("RAG_RETRIEVAL_CONTEXT_CHARS", "3200"))
    blob = f"ውይይት ታሪክ:\n{hist}\n\nአሁን የተጠየቀው:\n{q}"
    while len(blob) > cap and hist:
        hist = hist[max(200, len(hist) // 5) :]
        blob = f"ውይይት ታሪክ:\n{hist}\n\nአሁን የተጠየቀው:\n{q}"
    return blob.strip()


def build_context(chunks: list[dict[str, Any]], max_chars: int, *, compact: bool = False) -> str:
    parts: list[str] = []
    n = 0
    for i, c in enumerate(chunks, 1):
        meta = c.get("meta") or {}
        kind = meta.get("kind", "")
        page = meta.get("page", "")
        head = f"[{i}]"
        if kind == "pdf" and page and not compact:
            head += f" ገጽ {page}"
        block = f"{head}\n{c.get('text', '')}\n"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n".join(parts).strip()


def _strip_answer_meta_openers(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    first = re.sub(
        r"^(?:መረጃው\s*እንደሚለው|መረጃው\s+እንደሚለው|እንደ\s*መረጃው|በመረጃው\s+መሰረት)\s*[፦:.\s]*",
        "",
        first,
        flags=re.IGNORECASE,
    ).lstrip()
    lines[0] = first
    return "\n".join(lines).strip()


def sanitize_chat_answer(text: str | None) -> str:
    if not text or not str(text).strip():
        return (text or "").strip()
    t0 = text.strip()
    if re.match(r"^\[(?:groq|gemini|ollama|openai)\]", t0, re.I) or t0.startswith("[Ollama]"):
        return t0
    t0 = _strip_answer_meta_openers(t0)
    out: list[str] = []
    for line in t0.splitlines():
        s = line.strip()
        if re.match(r"^ምንጭ\s*[:፦]", s):
            continue
        if "ከላይ ካለው መረጃ ቁጥር" in s and ("አውጣ" in s or "አታድግም" in s):
            continue
        out.append(line)
    joined = "\n".join(out).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined
