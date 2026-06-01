"""
Shared retrieve + merge + rank path (Postgres pgvector + optional Chroma).
Used by voice ``/rag/answer`` to avoid duplicating ranking logic.
"""

from __future__ import annotations

import os
from typing import Any

from farmer_rag_stack.context_utils import retrieval_query_for
from farmer_rag_stack.nlu_farmer import augment_retrieval_query_with_nlu, parse_farmer_nlu
from farmer_rag_stack.retrieval_ranking import rank_pg_hits
from chroma_retrieve import merge_pg_chroma_hits, retrieve_chroma_mirror_hits


def ranked_hits_for_voice_query(
    *,
    query_text: str,
    nlu,
    user_region: str | None,
    hist_pairs: list[tuple[str, str]],
    max_l2_distance: float,
) -> tuple[list[dict[str, Any]], str, Any, float, dict[str, Any]]:
    """
    Returns ``(hits, retrieval_query, farmer_nlu, best_distance, diagnostics)``.
    ``hits`` is already trimmed to ``RAG_PG_FINAL_TOP_K`` (when non-empty).
    """
    import rag_pg

    conv_msgs = [{"role": r, "content": (m or "").strip()} for r, m in hist_pairs if (m or "").strip()]
    farmer_nlu = parse_farmer_nlu(query_text)
    rq = augment_retrieval_query_with_nlu(
        retrieval_query_for(nlu.retrieval_query or query_text, conv_msgs),
        farmer_nlu,
    )
    pool = max(12, int(os.environ.get("RAG_PG_RETRIEVE_POOL", "40")))
    raw_hits, best = rag_pg.retrieve_for_query(
        rq,
        top_k=pool,
        max_l2_distance=max_l2_distance,
        region=user_region,
    )
    pg_hits = [h for h in raw_hits if float(h.get("distance", 999)) <= max_l2_distance]
    chroma_k = max(8, int(os.getenv("RAG_CHROMA_TOP_K", "18").strip() or "18"))
    chroma_hits = retrieve_chroma_mirror_hits(rq, top_k=chroma_k)
    merged = merge_pg_chroma_hits(pg_hits, chroma_hits)
    diagnostics: dict[str, Any] = {
        "query": rq,
        "user_region": user_region,
        "max_l2_distance": max_l2_distance,
        "pg_raw_count": len(raw_hits),
        "pg_filtered_count": len(pg_hits),
        "chroma_count": len(chroma_hits),
        "merged_count": len(merged),
        "best_distance": best,
        "top_titles": [h.get("title") for h in raw_hits[:5] if h.get("title")],
    }

    def _retrieve_more(q: str) -> list:
        hh, _ = rag_pg.retrieve_for_query(
            q,
            top_k=pool,
            max_l2_distance=max_l2_distance,
            region=user_region,
        )
        return [h for h in hh if float(h.get("distance", 999)) <= max_l2_distance]

    keep = max(4, int(os.environ.get("RAG_PG_FINAL_TOP_K", "6")))
    hits = merged
    if hits:
        hits = rank_pg_hits(
            query_text,
            rq,
            hits,
            farmer_nlu,
            retrieve_more=_retrieve_more,
            max_hits=max(keep, 8),
        )
        hits = hits[:keep]
    diagnostics["final_count"] = len(hits)
    diagnostics["final_titles"] = [h.get("title") for h in hits[:5] if h.get("title")]
    return hits, rq, farmer_nlu, best, diagnostics
