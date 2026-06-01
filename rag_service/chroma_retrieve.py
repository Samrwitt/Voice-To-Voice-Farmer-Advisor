"""Optional hybrid retrieval: merge Chroma hits with Postgres pgvector hits."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def chroma_retrieve_enabled() -> bool:
    return os.getenv("RAG_CHROMA_RETRIEVE", "true").strip().lower() in ("1", "true", "yes", "on")


def retrieve_chroma_mirror_hits(query_text: str, top_k: int = 16) -> list[dict[str, Any]]:
    if not chroma_retrieve_enabled():
        return []
    try:
        from database import collection
    except Exception:
        collection = None
    if not collection:
        return []
    q = (query_text or "").strip()
    if not q:
        return []
    try:
        res = collection.query(query_texts=[q], n_results=max(1, int(top_k)))
    except Exception as exc:
        logger.debug("Chroma query failed: %s", exc)
        return []
    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    max_d = float(os.getenv("RAG_CHROMA_MAX_DISTANCE", "2.5").strip() or "2.5")
    out: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        body = (doc or "").strip()
        if not body:
            continue
        dist = float(dists[i]) if i < len(dists) else 999.0
        if dist > max_d:
            continue
        md = metas[i] if i < len(metas) else {}
        title = (md or {}).get("source") or "kb"
        ext = (md or {}).get("doc_id")
        if not ext and i < len(ids):
            cid = ids[i]
            ext = cid.rsplit(":", 1)[0] if ":" in cid else cid
        kind = (md or {}).get("kind") or "kb"
        cid = ids[i] if i < len(ids) else f"chroma:{i}"
        out.append(
            {
                "chunk_id": f"chroma:{cid}",
                "document_id": str(ext or "chroma"),
                "content": body,
                "distance": dist,
                "title": title,
                "source_org": "chroma_mirror",
                "source_url": None,
                "language": "am",
                "chunk_kind": kind,
            }
        )
    return out


def dedupe_hits_by_content_prefix(hits: list[dict], prefix_len: int = 240) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        key = (h.get("content") or "")[:prefix_len].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def merge_pg_chroma_hits(pg_hits: list[dict], chroma_hits: list[dict]) -> list[dict]:
    if not chroma_hits:
        return pg_hits
    if not pg_hits:
        return chroma_hits
    return dedupe_hits_by_content_prefix(pg_hits + chroma_hits)
