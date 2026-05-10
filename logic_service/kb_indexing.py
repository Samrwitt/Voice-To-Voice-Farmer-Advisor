"""
KB document ingestion + indexing pipeline for the Chroma collection.

Supported file types: .txt, .md, .pdf
"""
import os
from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from database import collection
from models import KBDocument, KBDocumentChunk


CHUNK_SIZE_CHARS = int(os.getenv("KB_CHUNK_SIZE_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("KB_CHUNK_OVERLAP_CHARS", "200"))


def extract_text(path: str, mime_type: str | None) -> str:
    """Extract plain text from a stored KB document."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md") or (mime_type and mime_type.startswith("text/")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf" or mime_type == "application/pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    raise RuntimeError(f"Unsupported file type for indexing: {ext or mime_type}")


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def remove_document_from_chroma(db: Session, document: KBDocument) -> int:
    """Delete all Chroma chunks belonging to a document. Returns count removed."""
    chunks = db.query(KBDocumentChunk).filter(
        KBDocumentChunk.document_id == document.id
    ).all()
    if not chunks:
        return 0
    chroma_ids = [c.chroma_id for c in chunks]
    try:
        collection.delete(ids=chroma_ids)
    except Exception as exc:
        print(f"[KB] Chroma delete failed for doc {document.id}: {exc}")
    for c in chunks:
        db.delete(c)
    db.commit()
    return len(chroma_ids)


def index_document(db: Session, document: KBDocument) -> KBDocument:
    """
    Index a stored KB document into Chroma.

    Updates `document.indexing_status`, `chroma_doc_count`, `last_indexed_at`,
    and persists per-chunk rows in `kb_document_chunks`.
    """
    document.indexing_status = "indexing"
    document.indexing_error = None
    db.commit()

    # Wipe existing chunks (re-index path)
    remove_document_from_chroma(db, document)

    try:
        text = extract_text(document.storage_path, document.mime_type)
        chunks = chunk_text(text)
        if not chunks:
            raise RuntimeError("No text could be extracted from the document.")

        ids = [f"kbdoc_{document.id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document.id,
                "filename": document.filename,
                "topic": document.topic or "",
                "crop": document.crop or "",
                "region": document.region or "",
                "category": document.category or "",
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(documents=chunks, metadatas=metadatas, ids=ids)

        for i, chroma_id in enumerate(ids):
            db.add(
                KBDocumentChunk(
                    document_id=document.id,
                    chroma_id=chroma_id,
                    chunk_index=i,
                    status="indexed",
                )
            )

        document.indexing_status = "indexed"
        document.chroma_doc_count = len(ids)
        document.last_indexed_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        db.rollback()
        document.indexing_status = "failed"
        document.indexing_error = str(exc)
        db.commit()
        raise

    return document
