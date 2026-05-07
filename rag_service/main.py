from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import re
import logging
import requests
import base64
import time
from database import (
    collection, add_to_escalation, log_conversation,
    get_conversation_history, get_market_price, register_farmer,
    get_farmer_profile, get_alerts_for_region, set_session_state,
    get_session_state, insert_call_record,
)
from nlu import analyze_intent, needs_slot_filling
from dynamic_layer_runtime import build_dynamic_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("logic_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import rag_pg
        rag_pg.init_pg_schema()
    except Exception as exc:
        logger.warning("Postgres KB init skipped: %s", exc)

    # Optional: auto-ingest local kb_documents/ folder on first boot.
    # IMPORTANT: run in a background thread so the API becomes responsive fast.
    try:
        import threading
        from bootstrap_ingest import auto_ingest_if_empty

        def _run():
            try:
                report = auto_ingest_if_empty()
                if report.get("enabled"):
                    logger.info("KB auto-ingest report: %s", report)
            except Exception as exc:
                logger.warning("KB auto-ingest failed: %s", exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
    except Exception as exc:
        logger.warning("KB auto-ingest setup skipped: %s", exc)
    yield


app = FastAPI(lifespan=lifespan)

# NOTE: This service is dedicated to RAG (static+dynamic) and KB ingestion.
# Dashboard/admin APIs are provided by logic_service; do not mount the legacy
# rag_service admin router (it pulls sqlite/bcrypt dependencies).

# ── Config (externalized) ────────────────────────────────────────────────────
RAG_DISTANCE_THRESHOLD = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "1.2"))
RAG_PG_MAX_L2_DISTANCE = float(os.environ.get("RAG_PG_MAX_L2_DISTANCE", "1.35"))
TTS_URL = os.environ.get("TTS_URL", "http://tts-service:8000/synthesize")
STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")

# ── LLM (optional) ───────────────────────────────────────────────────────────
# Speed is a priority; this service is designed to return grounded responses
# without requiring an LLM dependency. Keep llm=None.
llm = None


# ── Pydantic Models ──────────────────────────────────────────────────────────
class Query(BaseModel):
    text: str
    phone_number: str = "Unknown"
    session_id: str = "default_session"


class FarmerProfile(BaseModel):
    phone_number: str
    name: str
    location: str
    preferred_language: str = "am"


class E2ERequest(BaseModel):
    text_input: str
    phone_number: str = "Unknown"
    session_id: str = "test_session"


class RagAnswerRequest(BaseModel):
    text: str
    phone_number: str = "Unknown"
    session_id: str = "default_session"



# ── Text Normalization ───────────────────────────────────────────────────────
UNIT_MAP = {
    r'\bkg\b': 'ኪሎ ግራም',
    r'\bg\b': 'ግራም',
    r'\bha\b': 'ሄክታር',
    r'\bhectare\b': 'ሄክታር',
    r'\bL\b': 'ሊትር',
    r'\bliter\b': 'ሊትር',
    r'\bml\b': 'ሚሊ ሊትር',
    r'\bquintal\b': 'ኩንታል',
    r'\bqt\b': 'ኩንታል',
    r'\bbirr\b': 'ብር',
    r'\bETB\b': 'ብር',
}


def normalize_text(text: str) -> str:
    """Expand agricultural units/abbreviations for natural TTS pronunciation."""
    for pattern, replacement in UNIT_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Language Detection ───────────────────────────────────────────────────────
def is_amharic(text: str) -> bool:
    """Returns True if the text is primarily in Amharic (Ethiopic Unicode block)."""
    if not text:
        return False
    amharic_chars = sum(1 for c in text if '\u1200' <= c <= '\u137f')
    return amharic_chars / max(len(text.replace(' ', '')), 1) > 0.3


# ── Grounded answer without LLM (combine top chunks, Amharic framing) ───────
def compose_grounded_answer_no_llm(query_text: str, hits: list[dict], max_chars: int = 3200) -> str:
    if not hits:
        return ""
    if len(hits) == 1:
        return (hits[0].get("content") or "")[:max_chars]
    intro = "ከሰነዶች የተገኘው መረጃ እንደሚከተለው ነው።\n\n"
    parts: list[str] = []
    budget = max(200, max_chars - len(intro) - 40)
    per = budget // min(len(hits), 3)
    for i, h in enumerate(hits[:3], 1):
        body = (h.get("content") or "").strip()
        if not body:
            continue
        cap = min(len(body), per)
        parts.append(f"({i}) {body[:cap]}")
    return (intro + "\n\n".join(parts))[:max_chars]


# ── Core RAG Pipeline ────────────────────────────────────────────────────────
def generate_rag_response(query_text: str, phone_number: str, session_id: str):
    """
    Returns (response_text, intent, references, nlu_dict).
    - intent: outcome label for routing (often same as NLU primary_intent for KB turns).
    - nlu_dict: { primary_intent, confidence, entities } from analyze_intent.
    """
    logger.info(f"Processing query for session={session_id} phone={phone_number}: '{query_text}'")
    log_conversation(phone_number, session_id, "user", query_text)

    # ── Expert Response Check ─────────────────────────────────────────────────
    # If a previous escalation was answered but not yet delivered, deliver it now.
    import rag_pg
    if rag_pg.kb_pg_enabled():
        try:
            import psycopg
            with psycopg.connect(rag_pg.POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, query, expert_response FROM escalations WHERE phone_number = %s AND status = 'answered' ORDER BY created_at ASC LIMIT 1",
                        (phone_number,)
                    )
                    row = cur.fetchone()
                    if row:
                        esc_id, orig_query, expert_resp = row
                        cur.execute("UPDATE escalations SET status = 'closed', closed_at = NOW() WHERE id = %s", (esc_id,))
                        conn.commit()
                        
                        resp = f"ቀደም ብለው ስለ '{orig_query}' ላቀረቡት ጥያቄ የባለሙያ ምላሽ አለኝ፡ {expert_resp}\n\nአሁን ደግሞ ስለ አዲሱ ጥያቄዎ ልርዳዎት።"
                        # We return early or just prepend? Prepending is better so we still answer the current query.
                        # But for simplicity in a voice UI, let's deliver the answer first.
                        # Actually, let's just deliver it and then let the user ask again, 
                        # or just prepend. Prepending is better for latency.
                        current_resp, intent, refs, nlu = _generate_core_rag_response(query_text, phone_number, session_id)
                        return f"{resp}\n\n{current_resp}", intent, refs, nlu
        except Exception as exc:
            logger.error(f"Failed to check for expert responses: {exc}")

    return _generate_core_rag_response(query_text, phone_number, session_id)


def _generate_core_rag_response(query_text: str, phone_number: str, session_id: str):
    """
    Returns (response_text, intent, references, nlu_dict).
    - intent: outcome label for routing (often same as NLU primary_intent for KB turns).
    - nlu_dict: { primary_intent, confidence, entities } from analyze_intent.
    """
    # ── Language Check ────────────────────────────────────────────────────────
    if query_text.strip() and not is_amharic(query_text):
        resp = "እባክዎ ጥያቄዎን በአማርኛ ይናገሩ።"  # Please ask your question in Amharic.
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "non_amharic", [], {}

    nlu = analyze_intent(query_text)
    logger.info("NLU intent=%s conf=%.2f entities=%s", nlu.primary_intent, nlu.confidence, nlu.entities)

    # ── Farmer Profile & Context ──────────────────────────────────────────────
    profile = get_farmer_profile(phone_number)
    farmer_location = profile['location'] if profile else "Unknown"
    
    # Identify the relevant region for RAG filtering
    # Priority: 1. NLU extracted region, 2. Profile region
    user_region = nlu.entities.get("region_en")
    if not user_region and profile:
        # Map profile location to a region keyword if possible
        loc = str(profile.get('location', '')).lower()
        if any(k in loc for k in ["highland", "ደጋ"]): user_region = "highland"
        elif any(k in loc for k in ["lowland", "ቆላ"]): user_region = "lowland"
        elif any(k in loc for k in ["midland", "ወይና"]): user_region = "midland"

    user_context = f"Farmer Location: {farmer_location}. Region: {user_region or 'General'}. " if profile else ""

    # ── Active Alerts ─────────────────────────────────────────────────────────
    alerts = get_alerts_for_region(farmer_location)
    alerts_text = f"ማሳሰቢያ: {alerts[0][0]}\n\n" if alerts else ""

    # ── Safety Confirmation State ─────────────────────────────────────────────
    state = get_session_state(session_id)
    if state and state["current_state"] == "awaiting_confirmation":
        if "አዎ" in query_text or "yes" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = alerts_text + state["pending_action"]
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "confirmed_action", [], nlu.to_dict()
        elif "አይ" in query_text or "no" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = "እሺ፣ እርምጃው ተሰርዟል። ሌላ ምን ልርዳዎት?"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "cancelled_action", [], nlu.to_dict()
        else:
            resp = "እባክዎን 'አዎ' ወይም 'አይ' ብለው ያረጋግጡ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "awaiting_confirmation", [], nlu.to_dict()

    # ── Slot Awaiting State ───────────────────────────────────────────────────
    if state and state["current_state"] == "awaiting_slot":
        # user provided the missing crop/info; resume with enriched query
        original_query = state.get("pending_action", "")
        enriched_query = f"{original_query} {query_text}"
        set_session_state(session_id, "active", None)
        return generate_rag_response(enriched_query, phone_number, session_id)

    # ── Slot Filling Check ────────────────────────────────────────────────────
    clarification = needs_slot_filling(query_text, state, nlu)
    if clarification:
        set_session_state(session_id, "awaiting_slot", query_text)
        log_conversation(phone_number, session_id, "assistant", clarification)
        return clarification, "awaiting_slot", [], nlu.to_dict()

    # ── Complex Query Escalation ──────────────────────────────────────────────
    sensitive_intents = {"pest_disease", "soil_fertility", "crop_production"}
    if nlu.primary_intent in sensitive_intents and nlu.confidence < 0.6:
        logger.warning(f"Complex query detected (intent={nlu.primary_intent} conf={nlu.confidence}). Escalating.")
        add_to_escalation(
            query_text,
            f"Complex {nlu.primary_intent} query with low NLU confidence ({nlu.confidence:.2f}).",
            phone_number=phone_number,
            session_id=session_id,
            reason_code="COMPLEX_QUERY",
            confidence=nlu.confidence,
            entities=nlu.entities,
        )
        resp = "ይህ ጥያቄ ዝርዝር መረጃ ስለሚያስፈልገው ለግብርና ባለሙያ አስተላልፌዋለሁ። በቅርቡ መልስ ያገኛሉ።"
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "escalated_complex", [], nlu.to_dict()

    # ── Market Price Intent ───────────────────────────────────────────────────
    if nlu.primary_intent == "market_price":
        crop_name = nlu.entities.get("crop_en")
        logger.info("Market price intent; crop=%s", crop_name)
        if crop_name:
            price_data = get_market_price(crop_name, farmer_location) or get_market_price(crop_name)
            if price_data:
                price, unit, updated_at = price_data
                resp = f"የ{crop_name} ዋጋ {price} ብር በ {unit} ነው። (የዋጋ ቀን: {updated_at})"
                log_conversation(phone_number, session_id, "assistant", resp)
                return resp, "market_price", [], nlu.to_dict()
            else:
                resp = f"ለ{crop_name} ዋጋ መረጃ አሁን የለም። ቆይተው ይደውሉ።"
                log_conversation(phone_number, session_id, "assistant", resp)
                return resp, "market_price_unavailable", [], nlu.to_dict()
        else:
            # Crop not specified
            resp = "ስለ ምን ሰብል ዋጋ ይፈልጋሉ? (ጤፍ፣ ስንዴ፣ ቦሎቄ፣ ወዘተ.)"
            set_session_state(session_id, "awaiting_slot", query_text)
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "awaiting_slot", [], nlu.to_dict()

    # ── RAG: Postgres+pgvector (preferred) or legacy Chroma ───────────────────
    import rag_pg

    def _keyword_overlap_score(query: str, text: str) -> int:
        """
        Tiny hybrid-rerank: prefer chunks that contain key query words.
        This fixes common embedding confusion for generic words like "መመሪያ".
        """
        if not query or not text:
            return 0
        q = re.sub(r"\s+", " ", query.strip())
        t = (text or "")
        # Prefer longer / more specific tokens, keep Ethiopic + ASCII words
        tokens = re.findall(r"[\u1200-\u137F]+|[A-Za-z]+", q)
        stop = {
            "የ",
            "እና",
            "ነው",
            "ለ",
            "በ",
            "ላይ",
            "ነበር",
            "ምን",
            "ማን",
            "እንዴት",
            "እባክዎ",
            "ይህ",
            "ይህን",
            "መሆኑ",
            "መሆን",
        }
        scored = 0
        for tok in tokens:
            if len(tok) < 3:
                continue
            if tok in stop:
                continue
            # Exact substring match is fine for Amharic morphology MVP
            if tok in t:
                scored += 2
        return scored

    references: list = []
    context: str | None = None
    hits: list[dict] = []
    closest_distance = 999.0
    use_pg = rag_pg.kb_pg_enabled() and rag_pg.count_approved_chunks() > 0
    retrieval_query = nlu.retrieval_query or query_text

    if use_pg:
        # Pull more candidates then rerank with keyword overlap.
        hits, closest_distance = rag_pg.retrieve_for_query(
            retrieval_query, 
            top_k=12, 
            max_l2_distance=RAG_PG_MAX_L2_DISTANCE,
            region=user_region
        )
        if not hits:
            logger.warning(
                "Postgres RAG: no chunk within L2 distance %.3f (best was %.3f). Escalating.",
                RAG_PG_MAX_L2_DISTANCE,
                closest_distance,
            )
            add_to_escalation(
                query_text,
                f"PG RAG: no chunk within L2 {RAG_PG_MAX_L2_DISTANCE} (best={closest_distance:.3f}).",
                phone_number=phone_number,
                session_id=session_id,
                reason_code="LOW_CONFIDENCE",
                confidence=closest_distance,
                entities=nlu.entities,
            )
            resp = "ይቅርታ፣ ይህንን ጥያቄ ሙሉ በሙሉ ልመልስ አልቻልኩም። ለባለሙያ አስተላልፌዋለሁ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "escalated", [], nlu.to_dict()

        # Hybrid rerank (keyword overlap) then keep the best 4
        hits = sorted(
            hits,
            key=lambda h: (
                -_keyword_overlap_score(query_text, (h.get("title") or "") + "\n" + (h.get("content") or "")),
                float(h.get("distance") or 999.0),
            ),
        )
        hits = hits[:4]

        references = [
            {
                "chunk_id": h["chunk_id"],
                "document_id": h["document_id"],
                "title": h["title"],
                "source_org": h["source_org"],
                "source_url": h["source_url"],
                "distance": h["distance"],
                "snippet": (h["content"][:400] + "...") if len(h["content"]) > 400 else h["content"],
            }
            for h in hits[:3]
        ]
        context = "\n\n".join(h["content"] for h in hits[:3])
    else:
        if not collection:
            add_to_escalation(query_text, "Chroma disabled and Postgres KB empty/unavailable.")
            resp = "ይቅርታ፣ የመረጃ መዝገቡ አሁን አልተዘጋጀም። እባክዎ ቆይተው ይሞክሩ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "kb_unavailable", [], nlu.to_dict()

        results = collection.query(query_texts=[retrieval_query], n_results=2)

        if not results["documents"] or not results["documents"][0]:
            distances = [999]
        else:
            distances = results["distances"][0]

        closest_distance = distances[0] if distances else 999

        if closest_distance > RAG_DISTANCE_THRESHOLD:
            logger.warning(
                "Chroma distance %.2f > threshold %.2f. Escalating.",
                closest_distance,
                RAG_DISTANCE_THRESHOLD,
            )
            add_to_escalation(
                query_text, 
                f"Chroma distance: {closest_distance:.2f}. No confident KB match.",
                phone_number=phone_number,
                session_id=session_id,
                reason_code="LOW_CONFIDENCE",
                confidence=closest_distance,
                entities=nlu.entities,
            )
            resp = "ይቅርታ፣ ይህንን ጥያቄ ሙሉ በሙሉ ልመልስ አልቻልኩም። ለባለሙያ አስተላልፌዋለሁ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "escalated", [], nlu.to_dict()

        context = results["documents"][0][0]
        hits = [{"content": context}]

    intent = nlu.primary_intent

    history = get_conversation_history(session_id, limit=3)
    history_str = "\n".join([f"{h[0]}: {h[1]}" for h in history])

    # ── LLM or Direct KB Response ─────────────────────────────────────────────
    if llm:
        logger.info("Invoking LLM for grounded response...")
        prompt = (
            f"መመሪያ: ከታች ያለው መረጃ በአማርኛ ነው። ከዚያ ብቻ መልስ ስጥ፤ ከውጭ እውቀት አትጨምር።\n"
            f"Context:\n{user_context}{context}\n\n"
            f"ታሪክ:\n{history_str}\n\n"
            f"ጥያቄ:\n{query_text}\n\n"
            f"መልስ (በአማርኛ ብቻ፣ አጫጫን ያለህ ከሆነ አጭር አድርግ)፦"
        )
        response_text = llm(prompt)
    else:
        if use_pg and hits:
            response_text = compose_grounded_answer_no_llm(query_text, hits)
        else:
            response_text = context or ""

    # ── High-Risk Safety Interceptor ──────────────────────────────────────────
    # Only trigger on the user's question (and NLU) to avoid false-positives
    # from unrelated retrieved context.
    high_risk_keywords = [
        "pest",
        "chemical",
        "disease",
        "fertilizer",
        "spray",
        "ፀረ-ተባይ",
        "ማዳበሪያ",
        "ርጭት",
        "ፀረ",
    ]
    q_lower = (query_text or "").lower()
    if nlu.primary_intent in ("pest_disease", "soil_fertility") or any(hk in q_lower for hk in high_risk_keywords):
        logger.warning(f"High-risk topic detected for session {session_id}. Requiring confirmation.")
        set_session_state(session_id, "awaiting_confirmation", response_text)
        resp = alerts_text + "ይህ እርምጃ ጥንቃቄ ይፈልጋል። ስለ ሁኔታዎ እርግጠኛ ነዎት? (አዎ ወይም አይ)"
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "requires_confirmation", references, nlu.to_dict()

    final_response = alerts_text + normalize_text(response_text)
    log_conversation(phone_number, session_id, "assistant", final_response)
    return final_response, intent, references, nlu.to_dict()


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/ask")
async def process_query(query: Query):
    response_text, intent, references, nlu_out = generate_rag_response(
        query.text, query.phone_number, query.session_id
    )
    out = {"response": response_text, "intent": intent, "nlu": nlu_out}
    if references:
        out["references"] = references
    return out


@app.post("/rag/answer")
async def rag_answer(req: RagAnswerRequest):
    """
    Speed-first RAG endpoint for other services:
    - static retrieval: Postgres+pgvector (rag_kb_*)
    - dynamic retrieval: Postgres alerts/market (dynamic_layer_runtime)
    - response: grounded (no LLM required)
    """
    import rag_pg

    query_text = (req.text or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Empty query")

    nlu = analyze_intent(query_text)
    crop_name = getattr(nlu, "entities", {}).get("crop_en") if nlu else None
    try:
        dyn = build_dynamic_context(req.phone_number, crop_name=crop_name)
    except Exception:
        dyn = ""

    hits, best = rag_pg.retrieve_for_query(query_text, top_k=4)
    answer = compose_grounded_answer_no_llm(query_text, hits) if hits else ""

    if dyn and answer:
        final = f"{dyn}\n\n{answer}"
    elif dyn:
        final = dyn
    else:
        final = answer

    if not final:
        final = "ይቅርታ፣ በአሁኑ ጊዜ ለዚህ ጥያቄ የበቂ መረጃ አልተገኘም።"

    refs: list[dict] = []
    for h in hits[:3]:
        refs.append(
            {
                "document_id": h.get("document_id"),
                "chunk_id": h.get("chunk_id"),
                "title": h.get("title"),
                "source_org": h.get("source_org"),
                "source_url": h.get("source_url"),
                "distance": h.get("distance"),
            }
        )

    return {"response": normalize_text(final), "references": refs, "best_distance": best}


@app.post("/kb/ingest")
async def kb_ingest(
    file: UploadFile = File(...),
    external_document_id: str = Form(""),
    title: str = Form(""),
    source_org: str = Form("admin_dashboard"),
    source_url: str = Form(""),
    language: str = Form("am"),
    status: str = Form("approved"),
):
    """
    Ingest a KB document into Postgres+pgvector tables (rag_kb_documents/rag_kb_chunks).
    Called by logic_service when a dashboard KB doc is approved/reindexed.
    """
    import rag_pg
    import psycopg
    import uuid
    import tempfile

    if not rag_pg.kb_pg_enabled():
        raise HTTPException(status_code=503, detail="Postgres KB is not configured (POSTGRES_URL/psycopg)")

    rag_pg.init_pg_schema()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "kb_document"
    lower = filename.lower()

    text = ""
    try:
        if lower.endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="ignore")
        elif lower.endswith(".pdf"):
            from pdfminer.high_level import extract_text as pdf_extract_text

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                text = pdf_extract_text(tmp.name) or ""
        elif lower.endswith(".docx"):
            import docx

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                document = docx.Document(tmp.name)
                parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
                text = "\n".join(parts)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {exc}")

    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Too little text to index (scanned PDF?)")

    doc_title = (title or filename).strip()
    chunks = rag_pg.chunk_amharic_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created")

    embeddings = rag_pg.embed_texts(chunks)

    with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=False) as conn:
        with conn.cursor() as cur:
            doc_uuid = None
            if external_document_id:
                cur.execute(
                    "SELECT id FROM rag_kb_documents WHERE external_document_id = %s LIMIT 1;",
                    (external_document_id,),
                )
                row = cur.fetchone()
                if row:
                    doc_uuid = row[0]
                    cur.execute("DELETE FROM rag_kb_chunks WHERE document_id = %s;", (doc_uuid,))
                    cur.execute(
                        """
                        UPDATE rag_kb_documents
                        SET title=%s, source_org=%s, source_url=%s, language=%s, status=%s, original_filename=%s, updated_at=NOW()
                        WHERE id=%s;
                        """,
                        (
                            doc_title,
                            source_org or None,
                            source_url or None,
                            language or "am",
                            status or "approved",
                            filename,
                            doc_uuid,
                        ),
                    )

            if not doc_uuid:
                doc_uuid = uuid.uuid4()
                cur.execute(
                    """
                    INSERT INTO rag_kb_documents
                        (id, external_document_id, title, source_org, source_url, language, status, original_filename, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb);
                    """,
                    (
                        doc_uuid,
                        external_document_id or None,
                        doc_title,
                        source_org or None,
                        source_url or None,
                        language or "am",
                        status or "approved",
                        filename,
                    ),
                )

            for idx, emb in enumerate(embeddings):
                lit = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                cur.execute(
                    """
                    INSERT INTO rag_kb_chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s::vector);
                    """,
                    (doc_uuid, idx, chunks[idx], lit),
                )

        conn.commit()

    return {"status": "ok", "document_id": str(doc_uuid), "chunks": len(chunks)}


@app.get("/repeat/{session_id}")
async def repeat_last_response(session_id: str):
    """Returns the last assistant response for a given session (UC-06)."""
    history = get_conversation_history(session_id, limit=10)
    for role, message in reversed(history):
        if role == "assistant":
            return {"response": message}
    return {"response": "ቀዳሚ ምላሽ የለም።"}  # No previous response.


@app.post("/register")
async def register(profile: FarmerProfile):
    register_farmer(profile.phone_number, profile.name, profile.location, profile.preferred_language)
    return {"status": "success", "message": f"Farmer {profile.name} registered successfully."}


@app.get("/profile/{phone_number}")
async def get_profile(phone_number: str):
    profile = get_farmer_profile(phone_number)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.post("/save_call_record")
async def save_call_record(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    phone_number: str = Form(...),
    duration: int = Form(...)
):
    recordings_dir = os.path.join(DATA_DIR, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    file_path = os.path.join(recordings_dir, f"{session_id}.wav")

    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    insert_call_record(session_id, phone_number, file_path, duration)

    if not get_farmer_profile(phone_number):
        register_farmer(phone_number, "Unknown Caller", "Unknown")

    return {"status": "success", "file_path": file_path}


@app.post("/simulate_call")
async def simulate_call(req: E2ERequest):
    """End-to-end test endpoint: text in → logic → TTS → confirms pipeline is live."""
    transcribed_text = req.text_input
    response_text, intent, references, nlu_out = generate_rag_response(
        transcribed_text, req.phone_number, req.session_id
    )

    audio_b64 = None
    try:
        tts_resp = requests.post(TTS_URL, json={"text": response_text}, timeout=30)
        if tts_resp.status_code == 200:
            audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
        else:
            logger.error(f"TTS returned HTTP {tts_resp.status_code}")
    except Exception as e:
        logger.error(f"TTS request failed: {e}")

    payload = {
        "stt_output": transcribed_text,
        "logic_intent": intent,
        "logic_response": response_text,
        "nlu": nlu_out,
        "audio_base64_length": len(audio_b64) if audio_b64 else 0,
    }
    if references:
        payload["references"] = references
    return payload


@app.get("/system_check")
async def system_check():
    """Connectivity health check for all downstream services."""
    results = {}

    try:
        import sqlite3
        from database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.cursor().execute("SELECT 1")
        conn.close()
        results["database"] = "ok"
    except Exception as e:
        results["database"] = f"error: {e}"

    try:
        results["chroma_db"] = "ok" if collection.count() >= 0 else "empty"
    except Exception as e:
        results["chroma_db"] = f"error: {e}"

    try:
        stt_base = (os.environ.get("STT_URL") or "").strip()
        if not stt_base:
            results["stt_service"] = "disabled"
        else:
            stt_check = stt_base.rstrip("/") + "/docs"
            resp = requests.get(stt_check, timeout=3)
            results["stt_service"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        results["stt_service"] = f"error: {e}"

    try:
        tts_base = (os.environ.get("TTS_URL") or "").strip()
        if not tts_base:
            results["tts_service"] = "disabled"
        else:
            tts_check = tts_base.replace("/synthesize", "").rstrip("/") + "/docs"
            resp = requests.get(tts_check, timeout=3)
            results["tts_service"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        results["tts_service"] = f"error: {e}"

    results["rag_threshold"] = RAG_DISTANCE_THRESHOLD
    results["llm_loaded"] = llm is not None

    try:
        import rag_pg

        if rag_pg.kb_pg_enabled():
            rag_pg.init_pg_schema()
            results["postgres_kb"] = "ok"
            results["kb_pg_documents"] = rag_pg.count_documents()
            results["kb_pg_chunks"] = rag_pg.count_approved_chunks()
            results["rag_pg_max_l2"] = RAG_PG_MAX_L2_DISTANCE
        else:
            results["postgres_kb"] = "disabled"
    except Exception as e:
        results["postgres_kb"] = f"error: {e}"

    return results
