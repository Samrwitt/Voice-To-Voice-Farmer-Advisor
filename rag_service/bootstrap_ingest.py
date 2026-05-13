from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _iter_files(folder: Path) -> list[Path]:
    exts = (".pdf", ".docx", ".txt", ".md", ".jsonl")
    out: list[Path] = []
    for ext in exts:
        out.extend(folder.glob(f"*{ext}"))
    out = [p for p in out if p.is_file() and not p.name.startswith(".") and p.name.lower() != "readme.md"]
    return sorted(out, key=lambda p: p.name.lower())


def auto_ingest_if_empty() -> dict:
    """
    One-time bootstrap: ingest local KB files into Postgres+pgvector if the KB is empty.
    Controlled by:
      AUTO_INGEST_ON_STARTUP=true
      AUTO_INGEST_KB_DIR=/app/kb_documents/amharic
    """
    enabled = os.getenv("AUTO_INGEST_ON_STARTUP", "false").lower() in ("1", "true", "yes")
    if not enabled:
        return {"enabled": False, "ingested": 0, "skipped": "disabled"}

    folder = Path(os.getenv("AUTO_INGEST_KB_DIR", "")).expanduser()
    if not folder or not folder.is_dir():
        return {"enabled": True, "ingested": 0, "skipped": f"folder_not_found:{folder}"}

    import rag_pg
    import psycopg
    import uuid
    import re

    if not rag_pg.kb_pg_enabled():
        return {"enabled": True, "ingested": 0, "skipped": "pg_not_configured"}

    rag_pg.init_pg_schema()

    # Check if already populated
    force_reindex = os.getenv("FORCE_REINDEX_KB", "false").lower() in ("1", "true", "yes")
    
    with psycopg.connect(rag_pg.POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag_kb_chunks;")
            existing = int((cur.fetchone() or [0])[0])
            
    if existing > 0 and not force_reindex:
        return {"enabled": True, "ingested": 0, "skipped": "already_populated", "existing_chunks": existing}

    if force_reindex:
        logger.info("FORCE_REINDEX_KB is true. Clearing old data...")
        with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE rag_kb_chunks CASCADE;")
                cur.execute("TRUNCATE TABLE rag_kb_documents CASCADE;")
        try:
            from chroma_mirror import clear_mirrored_chroma_kb

            clear_mirrored_chroma_kb()
        except Exception as exc:
            logger.warning("Chroma mirror clear on reindex failed: %s", exc)

    def extract_text_from_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            from pdfminer.high_level import extract_text as pdf_extract_text

            return pdf_extract_text(str(path)) or ""
        if suffix == ".docx":
            import docx

            document = docx.Document(str(path))
            parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
            return "\n".join(parts)
        return ""

    files = _iter_files(folder)
    if not files:
        return {"enabled": True, "ingested": 0, "skipped": "no_files"}

    max_files = int(os.getenv("AUTO_INGEST_MAX_FILES", "50"))
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    ingested = 0
    for p in files:
        if ingested >= max_files:
            break
        
        if p.suffix.lower() == ".jsonl":
            import json
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    text = item.get("text_am") or item.get("text")
                    if not text or len(text.strip()) < 20: continue
                    
                    doc_uuid = uuid.uuid4()
                    external_id = f"jsonl:{p.name}:{item.get('id', uuid.uuid4())}"
                    title = item.get("title") or f"KB Item {item.get('id')}"

                    chunks = rag_pg.chunk_amharic_text(text)
                    if not chunks:
                        continue

                    with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=False) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO rag_kb_documents
                                    (id, external_document_id, title, source_org, source_url, language, status, original_filename, extra)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                                """,
                                (doc_uuid, external_id, title, "kb_jsonl", None, "am", "approved", p.name, json.dumps(item)),
                            )

                            embeddings = rag_pg.embed_texts(chunks, batch_size=batch_size)
                            for j, emb in enumerate(embeddings):
                                lit = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                                cur.execute(
                                    "INSERT INTO rag_kb_chunks (document_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s::vector);",
                                    (doc_uuid, j, chunks[j], lit),
                                )
                            conn.commit()
                    try:
                        from chroma_mirror import upsert_kb_chunks

                        upsert_kb_chunks(external_id, title, chunks, kind="kb")
                    except Exception as exc:
                        logger.debug("chroma mirror jsonl: %s", exc)
            ingested += 1
            continue

        raw = extract_text_from_file(p)
        text = re.sub(r"\s+", " ", (raw or "").strip())
        if len(text) < 80:
            continue
        chunks = rag_pg.chunk_amharic_text(text)
        if not chunks:
            continue

        external_id = f"kb_folder:{p.name}"
        title = p.stem.replace("_", " ").replace("-", " ")

        with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM rag_kb_documents WHERE external_document_id = %s LIMIT 1;",
                    (external_id,),
                )
                row = cur.fetchone()
                if row:
                    doc_uuid = row[0]
                    cur.execute("DELETE FROM rag_kb_chunks WHERE document_id = %s;", (doc_uuid,))
                    cur.execute(
                        """
                        UPDATE rag_kb_documents
                        SET title=%s, source_org=%s, source_url=%s, language=%s, status=%s,
                            original_filename=%s, updated_at=NOW()
                        WHERE id=%s;
                        """,
                        (title, "kb_documents_folder", None, "am", "approved", p.name, doc_uuid),
                    )
                else:
                    doc_uuid = uuid.uuid4()
                    cur.execute(
                        """
                        INSERT INTO rag_kb_documents
                            (id, external_document_id, title, source_org, source_url, language, status, original_filename, extra)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb);
                        """,
                        (doc_uuid, external_id, title, "kb_documents_folder", None, "am", "approved", p.name),
                    )

                # Embed + insert in small batches to avoid OOM.
                idx = 0
                while idx < len(chunks):
                    batch = chunks[idx : idx + batch_size]
                    embeddings = rag_pg.embed_texts(batch, batch_size=batch_size)
                    for j, emb in enumerate(embeddings):
                        lit = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                        cur.execute(
                            """
                            INSERT INTO rag_kb_chunks (document_id, chunk_index, content, embedding)
                            VALUES (%s, %s, %s, %s::vector);
                            """,
                            (doc_uuid, idx + j, batch[j], lit),
                        )
                    conn.commit()
                    idx += batch_size
        try:
            from chroma_mirror import upsert_kb_chunks

            upsert_kb_chunks(external_id, title, chunks, kind="kb")
        except Exception as exc:
            logger.debug("chroma mirror folder: %s", exc)
        ingested += 1

    return {
        "enabled": True,
        "ingested": ingested,
        "folder": str(folder),
        "files_seen": len(files),
        "embed_batch_size": batch_size,
        "max_files": max_files,
    }

