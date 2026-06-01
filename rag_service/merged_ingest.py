"""
Ingest ``RAG/merged.json`` style Q&A (``q`` / ``d``) into Postgres+pgvector and Chroma.

Runs in a background thread on service start (see ``main.py`` lifespan).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _merged_json_path() -> Path | None:
    raw = os.getenv("RAG_MERGED_JSON", "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "RAG" / "merged.json")
    candidates.append(Path("/app/RAG/merged.json"))
    for p in candidates:
        if p.is_file():
            return p
    return None


def sync_merged_qa() -> dict[str, Any]:
    """
    Delete prior ``merged_json:qa:*`` documents, then reload from disk.
    Controlled by ``RAG_MERGED_SYNC`` (default: true when file exists and PG is enabled).
    """
    path = _merged_json_path()
    if not path:
        return {"ok": False, "skipped": "no_merged_json_file"}

    flag = os.getenv("RAG_MERGED_SYNC", "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return {"ok": False, "skipped": "RAG_MERGED_SYNC_disabled"}

    import rag_pg

    if not rag_pg.kb_pg_enabled():
        return {"ok": False, "skipped": "pg_not_configured"}

    model_path = Path(rag_pg.EMBEDDING_MODEL_NAME)
    if model_path.is_absolute():
        has_weights = any((model_path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
        if not model_path.exists() or not has_weights:
            return {
                "ok": False,
                "skipped": "embedding_model_missing_or_incomplete",
                "embedding_model": str(model_path),
            }

    import psycopg

    rag_pg.init_pg_schema()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("merged.json read failed: %s", exc)
        return {"ok": False, "error": str(exc)}

    if not isinstance(data, list) or not data:
        return {"ok": False, "skipped": "empty_or_invalid_json"}

    force = os.getenv("FORCE_MERGED_REINDEX", "false").lower() in ("1", "true", "yes")
    existing = 0
    with psycopg.connect(rag_pg.POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rag_kb_documents WHERE external_document_id LIKE 'merged_json:qa:%';"
            )
            existing = int((cur.fetchone() or [0])[0])
    if existing >= len(data) * 0.97 and not force:
        return {"ok": True, "skipped": "already_synced", "existing": existing, "file_items": len(data)}

    from chroma_mirror import delete_chroma_by_kind, upsert_kb_chunks

    if force or existing > 0:
        try:
            delete_chroma_by_kind("qa")
        except Exception as exc:
            logger.debug("chroma delete qa: %s", exc)

    with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rag_kb_documents WHERE external_document_id LIKE 'merged_json:qa:%';")

    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16") or "16")
    ingested = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or "").strip()
        d = (item.get("d") or "").strip()
        if len(q) < 3 or len(d) < 5:
            continue
        body = f"ጥያቄ፦ {q}\n\nምላሽ፦ {d}"
        chunks = rag_pg.chunk_amharic_text(body)
        if not chunks:
            continue
        external_id = f"merged_json:qa:{idx}"
        title = q[:200] + ("…" if len(q) > 200 else "")
        doc_uuid = uuid.uuid4()
        try:
            with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=False) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rag_kb_documents
                            (id, external_document_id, title, source_org, source_url, language, status, original_filename, extra)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                        """,
                        (
                            doc_uuid,
                            external_id,
                            title,
                            "merged_qa",
                            None,
                            "am",
                            "approved",
                            path.name,
                            json.dumps({"qa_index": idx, "kind": "qa"}),
                        ),
                    )
                    pos = 0
                    while pos < len(chunks):
                        batch = chunks[pos : pos + batch_size]
                        embeddings = rag_pg.embed_texts(batch, batch_size=batch_size)
                        for j, emb in enumerate(embeddings):
                            lit = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                            cur.execute(
                                """
                                INSERT INTO rag_kb_chunks (document_id, chunk_index, content, embedding)
                                VALUES (%s, %s, %s, %s::vector);
                                """,
                                (doc_uuid, pos + j, batch[j], lit),
                            )
                        conn.commit()
                        pos += batch_size
            upsert_kb_chunks(external_id, title, chunks, kind="qa")
            ingested += 1
        except Exception as exc:
            logger.warning("merged QA ingest failed at index %s: %s", idx, exc)

    return {"ok": True, "ingested": ingested, "path": str(path), "total_json_items": len(data)}
