"""
Postgres + pgvector retrieval for the agronomy KB (SRS FR08/FR10).
Uses the same embedding space as the legacy Chroma path:
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual / Amharic-friendly).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("rag_pg")

POSTGRES_URL = os.environ.get("POSTGRES_URL", "").strip()
RAG_PG_MAX_L2_DISTANCE = float(os.environ.get("RAG_PG_MAX_L2_DISTANCE", "1.35"))
EMBEDDING_MODEL_NAME = os.environ.get(
    "KB_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIM = 384

_psycopg = None
_schema_ready = False
_embedder = None


def _load_psycopg():
    global _psycopg
    if _psycopg is None:
        try:
            import psycopg
            _psycopg = psycopg
        except ImportError:
            _psycopg = False
    return _psycopg if _psycopg is not False else None


def pg_configured() -> bool:
    return bool(POSTGRES_URL)


def pg_driver_ok() -> bool:
    return _load_psycopg() is not None


def kb_pg_enabled() -> bool:
    return pg_configured() and pg_driver_ok()


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    model = _get_embedder()
    bs = batch_size or int(os.environ.get("EMBED_BATCH_SIZE", "16"))
    vectors = model.encode(
        texts,
        normalize_embeddings=False,
        show_progress_bar=False,
        batch_size=bs,
    )
    return [v.tolist() for v in vectors]


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _conn():
    psycopg = _load_psycopg()
    if not psycopg:
        raise RuntimeError("psycopg is not installed")
    return psycopg.connect(POSTGRES_URL)


def init_pg_schema() -> None:
    """Create extension and tables if missing (idempotent)."""
    global _schema_ready
    if not kb_pg_enabled():
        logger.info("Postgres KB disabled (POSTGRES_URL unset or psycopg missing).")
        return
    if _schema_ready:
        return
    psycopg = _load_psycopg()
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # NOTE: we intentionally do NOT use the existing logic_service
            # `kb_documents` table name to avoid schema collisions.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_kb_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    external_document_id TEXT,
                    title TEXT NOT NULL,
                    source_org TEXT,
                    source_url TEXT,
                    language TEXT NOT NULL DEFAULT 'am',
                    status TEXT NOT NULL DEFAULT 'approved',
                    original_filename TEXT,
                    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_kb_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES rag_kb_documents(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(%s) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                );
                """
                % (EMBEDDING_DIM,),
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_kb_chunks_document_id_idx
                ON rag_kb_chunks(document_id);
                """
            )
    _schema_ready = True
    logger.info("Postgres KB schema ready (pgvector).")


def count_approved_chunks() -> int:
    if not kb_pg_enabled():
        return 0
    init_pg_schema()
    psycopg = _load_psycopg()
    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM rag_kb_chunks c
                JOIN rag_kb_documents d ON d.id = c.document_id
                WHERE d.status = 'approved';
                """
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0


def count_documents() -> int:
    if not kb_pg_enabled():
        return 0
    init_pg_schema()
    psycopg = _load_psycopg()
    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag_kb_documents WHERE status = 'approved';")
            row = cur.fetchone()
            return int(row[0]) if row else 0


def retrieve_for_query(
    query_text: str,
    top_k: int = 4,
    max_l2_distance: float | None = None,
    region: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Returns (hits, best_distance). Each hit:
      chunk_id, document_id, content, distance, title, source_org, source_url, language
    """
    if not kb_pg_enabled():
        return [], 999.0
    init_pg_schema()
    qvec = embed_texts([query_text])[0]
    lit = _vector_literal(qvec)

    # Build the WHERE clause for region filtering
    # We always include 'ethiopia_all' (general info) plus the specific region if provided.
    where_clause = "WHERE d.status = 'approved'"
    query_params = [lit]

    if region:
        # Many dashboard/bootstrap documents are untagged. Keep them eligible so
        # profile region does not accidentally hide national/general guidance.
        where_clause += (
            " AND (d.extra->>'region' = %s "
            "OR d.extra->>'region' = 'ethiopia_all' "
            "OR d.extra->>'region' IS NULL)"
        )
        query_params.append(region)
    
    query_params.extend([lit, top_k])

    psycopg = _load_psycopg()
    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            sql = f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    (c.embedding <-> %s::vector) AS distance,
                    d.title,
                    d.source_org,
                    d.source_url,
                    d.language
                FROM rag_kb_chunks c
                INNER JOIN rag_kb_documents d ON d.id = c.document_id
                {where_clause}
                ORDER BY c.embedding <-> %s::vector
                LIMIT %s;
            """
            cur.execute(sql, tuple(query_params))
            rows = cur.fetchall()

    hits: list[dict[str, Any]] = []
    best = 999.0
    for row in rows:
        dist = float(row[3])
        if dist < best:
            best = dist
        hits.append(
            {
                "chunk_id": str(row[0]),
                "document_id": str(row[1]),
                "content": row[2],
                "distance": dist,
                "title": row[4],
                "source_org": row[5],
                "source_url": row[6],
                "language": row[7],
            }
        )

    return hits, (best if hits else 999.0)


_KEYWORD_STOPWORDS = {
    "ነው",
    "ናቸው",
    "ምን",
    "ላይ",
    "ጋር",
    "እና",
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "is",
    "are",
    "in",
    "to",
    "of",
}


def _keyword_tokens(query_text: str, limit: int = 14) -> list[str]:
    raw = re.findall(r"[\u1200-\u137F]+|[A-Za-z0-9]+", (query_text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for token in raw:
        token = token.strip()
        if len(token) < 2 or token in _KEYWORD_STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def keyword_search_for_query(
    query_text: str,
    top_k: int = 12,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Postgres keyword search used with pgvector for hybrid retrieval."""
    if not kb_pg_enabled():
        return []
    tokens = _keyword_tokens(query_text)
    if not tokens:
        return []
    init_pg_schema()

    where_clause = "WHERE d.status = 'approved'"
    region_params: list[Any] = []
    if region:
        where_clause += (
            " AND (d.extra->>'region' = %s "
            "OR d.extra->>'region' = 'ethiopia_all' "
            "OR d.extra->>'region' IS NULL)"
        )
        region_params.append(region)

    score_terms: list[str] = []
    match_terms: list[str] = []
    score_params: list[Any] = []
    match_params: list[Any] = []
    for token in tokens:
        pat = f"%{token}%"
        score_terms.append("(CASE WHEN c.content ILIKE %s THEN 1 ELSE 0 END)")
        score_params.append(pat)
        score_terms.append("(CASE WHEN d.title ILIKE %s THEN 2 ELSE 0 END)")
        score_params.append(pat)
        match_terms.append("(c.content ILIKE %s OR d.title ILIKE %s)")
        match_params.extend([pat, pat])

    score_expr = " + ".join(score_terms)
    where_clause += " AND (" + " OR ".join(match_terms) + ")"
    params = score_params + region_params + match_params + [top_k]

    psycopg = _load_psycopg()
    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            sql = f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    ({score_expr}) AS keyword_score,
                    d.title,
                    d.source_org,
                    d.source_url,
                    d.language
                FROM rag_kb_chunks c
                INNER JOIN rag_kb_documents d ON d.id = c.document_id
                {where_clause}
                ORDER BY keyword_score DESC, c.created_at DESC
                LIMIT %s;
            """
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    max_score = max(1, len(tokens) * 2)
    hits: list[dict[str, Any]] = []
    for row in rows:
        score = float(row[3] or 0)
        if score <= 0:
            continue
        # Keep keyword-only evidence compatible with existing distance gates.
        distance = max(0.15, 1.05 - min(score / max_score, 1.0))
        hits.append(
            {
                "chunk_id": str(row[0]),
                "document_id": str(row[1]),
                "content": row[2],
                "distance": distance,
                "title": row[4],
                "source_org": row[5],
                "source_url": row[6],
                "language": row[7],
                "retrieval_mode": "keyword",
                "keyword_score": score,
            }
        )
    return hits


def chunk_amharic_text(text: str, chunk_size: int = 1600, overlap: int = 200) -> list[str]:
    """Character-window chunking (works well for Amharic prose without English sentence rules)."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks
