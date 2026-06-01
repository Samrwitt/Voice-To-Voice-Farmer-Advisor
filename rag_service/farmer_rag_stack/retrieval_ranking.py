"""
Postgres KB hit ranking: overlap rerank, crop/topic rules (from ``RAG/query.py``),
optional BM25 rescoring, optional second vector retrieve when crop signal is missing.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from farmer_rag_stack.nlu_farmer import FarmerNLU

_TOKEN_RE = re.compile(r"[\w\u1200-\u137F]{2,}")


def question_overlap_tokens(question: str) -> list[str]:
    toks = _TOKEN_RE.findall(question or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:28]


def rerank_hits_by_question_overlap(query_for_overlap: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits
    if os.environ.get("RAG_RERANK_Q_OVERLAP", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    toks = question_overlap_tokens(query_for_overlap)
    if len(toks) < 2:
        return hits

    def key_pair(idx_h: tuple[int, dict]) -> tuple:
        idx, h = idx_h
        meta = h.get("meta") or {}
        blob = (h.get("text") or "") + " " + str(meta.get("question") or "")
        overlap = sum(1 for t in toks if t in blob)
        rrf = h.get("rrf_rank")
        try:
            rr = int(rrf) if rrf is not None else 9999
        except (TypeError, ValueError):
            rr = 9999
        dist = h.get("distance")
        try:
            d = float(dist) if dist is not None else 1e9
        except (TypeError, ValueError):
            d = 1e9
        return (-overlap, rr, d, idx)

    indexed = list(enumerate(hits))
    indexed.sort(key=key_pair)
    return [h for _, h in indexed]


_CROP_TOPIC_RULES: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "id": "coffee",
        "triggers": ("ለቡና", "ቡና", "ቡናን"),
        "needles": (
            "ቡና",
            "ለቡና",
            "ጎማ",
            "coffee",
            "አረቢካ",
            "አረብካ",
            "ሮቡስታ",
            "አራቢካ",
        ),
        "rivals": ("ስንዴ", "ለስንዴ", "wheat", "ሰሊጥ", "ሰሊት", "ገብስ", "ቴፍ", "teff"),
    },
    {
        "id": "wheat",
        "triggers": ("ስንዴ",),
        "needles": ("ስንዴ", "wheat"),
        "rivals": ("ቡና", "ለቡና", "coffee", "ሰሊጥ", "ገብስ"),
    },
    {
        "id": "sesame",
        "triggers": ("ሰሊጥ", "ሰሊት"),
        "needles": ("ሰሊጥ", "ሰሊት", "sesame"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና", "ገብስ"),
    },
    {
        "id": "barley",
        "triggers": ("ገብስ",),
        "needles": ("ገብስ", "barley"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
    {
        "id": "faba",
        "triggers": ("ቡቃያ",),
        "needles": ("ቡቃያ",),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
    {
        "id": "potato",
        "triggers": ("ድንች",),
        "needles": ("ድንች", "potato"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
)


def _match_crop_topic_rule(question: str) -> dict[str, tuple[str, ...]] | None:
    q = question or ""
    for row in _CROP_TOPIC_RULES:
        if any(t in q for t in row["triggers"]):
            return row
    return None


def _crop_rule_by_id(crop_id: str | None) -> dict[str, tuple[str, ...]] | None:
    if not crop_id:
        return None
    for row in _CROP_TOPIC_RULES:
        if row.get("id") == crop_id:
            return row
    return None


def effective_crop_topic_rule(question: str, nlu: FarmerNLU) -> dict[str, tuple[str, ...]] | None:
    r = _crop_rule_by_id(nlu.crop_id)
    if r:
        return r
    return _match_crop_topic_rule(question)


def _crop_row_matches_question(
    row: dict[str, tuple[str, ...]],
    question: str,
    nlu: FarmerNLU,
) -> bool:
    if nlu.crop_id and row.get("id") == nlu.crop_id:
        return True
    return any(t in question for t in row["triggers"])


def _retrieval_blob(h: dict) -> str:
    m = h.get("meta") or {}
    return ((h.get("text") or "") + " " + str(m.get("question") or "")).strip()


def _topic_good_count(question: str, hits: list[dict], nlu: FarmerNLU) -> int:
    row = effective_crop_topic_rule(question, nlu)
    if not row:
        return len(hits)
    needles = row["needles"]
    return sum(1 for h in hits if any(n in _retrieval_blob(h) for n in needles))


def filter_cross_crop_hits(question: str, hits: list[dict], nlu: FarmerNLU) -> list[dict]:
    if not hits or os.environ.get("RAG_TOPIC_FILTER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    row = effective_crop_topic_rule(question, nlu)
    if not row:
        return hits
    needles, rivals = row["needles"], row["rivals"]
    good: list[dict] = []
    neutral: list[dict] = []
    bad: list[dict] = []
    for h in hits:
        b = _retrieval_blob(h)
        has_needle = any(n in b for n in needles)
        has_rival = any(r in b for r in rivals)
        if has_needle:
            good.append(h)
        elif has_rival:
            bad.append(h)
        else:
            neutral.append(h)
    if good:
        return good + neutral + bad
    return neutral + bad


def boost_hits_by_topic_keywords(question: str, hits: list[dict], nlu: FarmerNLU) -> list[dict]:
    if not hits or os.environ.get("RAG_TOPIC_BOOST", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    q = question or ""
    for row in _CROP_TOPIC_RULES:
        if not _crop_row_matches_question(row, q, nlu):
            continue
        needles = row["needles"]
        on_topic: list[dict] = []
        rest: list[dict] = []
        for h in hits:
            blob = _retrieval_blob(h)
            if any(n in blob for n in needles):
                on_topic.append(h)
            else:
                rest.append(h)
        if on_topic:
            return on_topic + rest
    return hits


def _tokenize_bm25(text: str) -> list[str]:
    """Amharic + alnum tokens (same spirit as ``RAG/hybrid_retrieval.tokenize``)."""
    text = (text or "").strip().lower()
    if not text:
        return []
    parts = re.findall(r"[\u1200-\u137F]+|[a-z0-9]+", text, re.I)
    return parts if parts else text.split()


def _bm25_rescore_rows(query: str, rows: list[dict]) -> list[dict]:
    if not rows or os.environ.get("RAG_PG_BM25", "1").strip().lower() in ("0", "false", "no", "off"):
        return rows
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return rows
    corpus = [_tokenize_bm25((r.get("text") or "")) for r in rows]
    q_tok = _tokenize_bm25(query)
    if not q_tok or not any(corpus):
        return rows
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tok)
    scored = list(zip(scores, range(len(rows)), rows))
    scored.sort(key=lambda x: (-x[0], float(x[2].get("distance") or 1e9)))
    return [t[2] for t in scored]


def pg_hits_to_rank_rows(pg_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape used by rerank / crop rules / ``augment_kb_context`` (expects text, meta, distance)."""
    out: list[dict[str, Any]] = []
    for h in pg_hits:
        cid = str(h.get("chunk_id") or "")
        so = (h.get("source_org") or "").strip().lower()
        ck = (h.get("chunk_kind") or "").strip().lower()
        kind: str = "qa" if ck == "qa" or "merged" in so else "kb"
        title = (h.get("title") or "").strip() or "kb"
        q_meta = title[:400] if kind == "qa" else None
        meta = {
            "source": title,
            "kind": kind,
            "page": None,
            "question": q_meta,
            "chunk_id": cid,
        }
        out.append(
            {
                "text": (h.get("content") or "").strip(),
                "meta": meta,
                "distance": float(h.get("distance") or 0.0),
                "_pg": h,
            }
        )
    return out


def rank_rows_to_pg_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r["_pg"] for r in rows if r.get("_pg")]


def rank_pg_hits(
    question: str,
    rq_overlap: str,
    pg_hits: list[dict[str, Any]],
    farmer_nlu: FarmerNLU,
    *,
    retrieve_more: Callable[[str], list[dict[str, Any]]] | None = None,
    max_hits: int = 24,
) -> list[dict[str, Any]]:
    """
    Rerank + crop filter + optional BM25 + optional second retrieve (``RAG_PG_TOPIC_REFINE``).
    ``retrieve_more`` should return raw pg hit dicts for a boosted query string.
    """
    if not pg_hits:
        return []
    rows = pg_hits_to_rank_rows(pg_hits)
    rows = _bm25_rescore_rows(rq_overlap, rows)
    rows = rerank_hits_by_question_overlap(rq_overlap, rows)
    rows = boost_hits_by_topic_keywords(question, rows, farmer_nlu)
    rows = filter_cross_crop_hits(question, rows, farmer_nlu)

    row = effective_crop_topic_rule(question, farmer_nlu)
    if (
        row
        and retrieve_more
        and _topic_good_count(question, rows, farmer_nlu) == 0
        and os.environ.get("RAG_PG_TOPIC_REFINE", "1").strip().lower()
        not in ("0", "false", "no", "off")
    ):
        tail_parts = list(row["needles"][: min(5, len(row["needles"]))])
        if farmer_nlu.aspect == "altitude" or "ከፍታ" in question:
            tail_parts.append("ከፍታ")
        if farmer_nlu.aspect == "price":
            tail_parts.extend(["ዋጋ", "ገበያ"])
        if farmer_nlu.aspect == "rainfall":
            tail_parts.append("ዝናብ")
        if farmer_nlu.aspect == "soil":
            tail_parts.append("አፈር")
        if farmer_nlu.aspect == "fertilizer":
            tail_parts.append("ማዳበሪያ")
        tail = " ".join(dict.fromkeys(tail_parts))
        rq_boost = f"{rq_overlap.strip()}\n{tail}".strip()
        try:
            extra_raw = retrieve_more(rq_boost)
        except Exception:
            extra_raw = []
        extra_rows = pg_hits_to_rank_rows(extra_raw)
        by_cid: dict[str, dict] = {}
        for r in rows + extra_rows:
            cid = (r.get("meta") or {}).get("chunk_id") or ""
            if cid and cid not in by_cid:
                by_cid[cid] = r
        rows = list(by_cid.values())
        rows = rerank_hits_by_question_overlap(rq_boost, rows)
        rows = boost_hits_by_topic_keywords(question, rows, farmer_nlu)
        rows = filter_cross_crop_hits(question, rows, farmer_nlu)

    out = rank_rows_to_pg_hits(rows)
    return out[: max(1, max_hits)]
