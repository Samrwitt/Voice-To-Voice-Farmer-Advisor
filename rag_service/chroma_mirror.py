"""Mirror KB chunks into the legacy Chroma ``agronomy_kb`` collection (same embedder as pgvector)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def chroma_mirror_enabled() -> bool:
    return os.getenv("RAG_CHROMA_MIRROR", "true").strip().lower() in ("1", "true", "yes", "on")


def upsert_kb_chunks(
    external_document_id: str,
    title: str,
    chunks: list[str],
    *,
    kind: str = "kb",
) -> None:
    if not chroma_mirror_enabled() or not chunks or not (external_document_id or "").strip():
        return
    try:
        from database import collection
    except Exception:
        collection = None
    if not collection:
        return
    eid = external_document_id.strip()
    ids = [f"{eid}:{i}" for i in range(len(chunks))]
    safe_title = (title or "kb")[:500]
    metadatas = [{"source": safe_title, "kind": kind, "doc_id": eid} for _ in chunks]
    try:
        collection.upsert(ids=ids, documents=list(chunks), metadatas=metadatas)
    except Exception as exc:
        logger.warning("Chroma mirror upsert failed for %s: %s", eid, exc)


def delete_chroma_by_kind(kind: str) -> None:
    if not chroma_mirror_enabled():
        return
    try:
        from database import collection
    except Exception:
        collection = None
    if not collection:
        return
    try:
        collection.delete(where={"kind": kind})
    except Exception as exc:
        logger.debug("Chroma delete kind=%s: %s", kind, exc)


def clear_mirrored_chroma_kb() -> None:
    """Remove mirrored rows (metadata ``kind`` in kb/qa). Safe to call on FORCE_REINDEX."""
    for k in ("kb", "qa"):
        delete_chroma_by_kind(k)
