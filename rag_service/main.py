from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
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
    get_session_state, insert_call_record, log_interaction_record,
    get_dynamic_knowledge, set_dynamic_knowledge
)
from nlu import analyze_intent, needs_slot_filling
from dynamic_layer_runtime import build_dynamic_context
from farmer_persona import build_personalization_block
from quality_metrics import quality_snapshot
from trust_meta import build_voice_trust_meta, maybe_append_trust_footer
from rag_retrieval import ranked_hits_for_voice_query
import chemical_safety
import response_cache
import voice_guards

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

        def _merged_qa():
            try:
                from merged_ingest import sync_merged_qa

                mrep = sync_merged_qa()
                if mrep.get("ingested"):
                    logger.info("merged.json QA sync: %s", mrep)
            except Exception as exc:
                logger.warning("merged.json QA sync failed: %s", exc)

        threading.Thread(target=_merged_qa, daemon=True).start()
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


class RagDebugContextRequest(BaseModel):
    text: str
    phone_number: str = "Unknown"
    session_id: str = "debug_session"
    retrieve: bool = True


def _require_metrics_token(authorization: Optional[str] = Header(None)) -> None:
    tok = os.getenv("RAG_METRICS_TOKEN", "").strip()
    if not tok:
        return
    if (authorization or "").strip() != f"Bearer {tok}":
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token (RAG_METRICS_TOKEN).")


@app.get("/api/quality/snapshot", dependencies=[Depends(_require_metrics_token)])
def api_quality_snapshot(hours: int = 24):
    """Aggregated interaction + escalation counts for the trust / proof loop."""
    h = max(1, min(int(hours or 24), 168))
    return quality_snapshot(window_hours=h)


def _ops_notify_tokens() -> list[str]:
    out: list[str] = []
    for env in ("OPS_NOTIFY_TOKEN", "RAG_METRICS_TOKEN"):
        v = (os.getenv(env) or "").strip()
        if v and v not in out:
            out.append(v)
    return out


def _require_ops_notify_token(authorization: Optional[str] = Header(None)) -> None:
    toks = _ops_notify_tokens()
    if not toks:
        raise HTTPException(
            status_code=503,
            detail="Set OPS_NOTIFY_TOKEN or RAG_METRICS_TOKEN to call this endpoint.",
        )
    auth = (authorization or "").strip()
    if not any(auth == f"Bearer {t}" for t in toks):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token.")


@app.post("/api/ops/notify", dependencies=[Depends(_require_ops_notify_token)])
def api_ops_notify():
    """
    Push a minimal SLA / backlog payload to ``OPS_ALERT_WEBHOOK_URL`` (Slack incoming
    webhook, PagerDuty, etc.). Intended for cron: call every N minutes; no-op when
    there are zero SLA breaches.
    """
    url = (os.getenv("OPS_ALERT_WEBHOOK_URL") or "").strip()
    if not url:
        raise HTTPException(status_code=503, detail="OPS_ALERT_WEBHOOK_URL is not set.")

    snap = quality_snapshot(window_hours=24)
    breaches = int(snap.get("escalations_pending_over_sla") or 0)
    if breaches <= 0:
        return {"ok": True, "pushed": False, "reason": "no_escalation_sla_breaches"}

    payload = {
        "source": "voice-farmer-advisor-rag",
        "escalations_pending_over_sla": breaches,
        "escalations_pending_total": snap.get("escalations_pending_total"),
        "sla_target_hours": (snap.get("policy") or {}).get("escalation_sla_target_hours"),
        "ops_alerts": snap.get("ops_alerts") or [],
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Webhook returned HTTP {r.status_code}: {r.text[:500]}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Webhook request failed: {exc}") from exc

    return {"ok": True, "pushed": True, "webhook_status": r.status_code}


@app.get("/rag/diagnostics", dependencies=[Depends(_require_metrics_token)])
def rag_diagnostics():
    """Readiness snapshot for KB ingestion, dynamic data, sources, and voice wiring."""
    from pathlib import Path

    def _pdfs(folder: str) -> list[str]:
        p = Path(folder)
        if not p.is_dir():
            return []
        return sorted(x.name for x in p.glob("*.pdf") if x.is_file())

    local_dirs = [
        os.getenv("AUTO_INGEST_KB_DIR", "/app/kb_documents/amharic"),
        *[
            x.strip()
            for x in os.getenv("AUTO_INGEST_KB_DIRS", "").replace(";", ",").split(",")
            if x.strip()
        ],
        "RAG/KB",
        "kb_documents/amharic",
    ]
    seen_dirs: list[str] = []
    local_pdfs: dict[str, list[str]] = {}
    for d in local_dirs:
        if d and d not in seen_dirs:
            seen_dirs.append(d)
            rows = _pdfs(d)
            if rows:
                local_pdfs[d] = rows

    pg = {"enabled": False, "approved_documents": 0, "approved_chunks": 0, "documents": []}
    embedding = {
        "model": os.getenv("KB_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        "path_exists": None,
        "weights_present": None,
    }
    try:
        import rag_pg
        model_path = Path(rag_pg.EMBEDDING_MODEL_NAME)
        embedding["model"] = rag_pg.EMBEDDING_MODEL_NAME
        embedding["path_exists"] = model_path.exists() if model_path.is_absolute() else None
        embedding["weights_present"] = (
            any((model_path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
            if model_path.is_absolute()
            else None
        )
        pg["enabled"] = bool(rag_pg.kb_pg_enabled())
        if rag_pg.kb_pg_enabled():
            pg["approved_documents"] = rag_pg.count_documents()
            pg["approved_chunks"] = rag_pg.count_approved_chunks()
            import psycopg

            with psycopg.connect(rag_pg.POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT original_filename, title, status
                        FROM rag_kb_documents
                        ORDER BY original_filename NULLS LAST, title
                        LIMIT 500;
                        """
                    )
                    pg["documents"] = [
                        {"original_filename": r[0], "title": r[1], "status": r[2]}
                        for r in (cur.fetchall() or [])
                    ]
    except Exception as exc:
        pg["error"] = str(exc)

    ingested_names = {
        str(d.get("original_filename") or "").strip()
        for d in pg.get("documents", [])
        if d.get("status") == "approved"
    }
    all_pdf_names = {name for names in local_pdfs.values() for name in names}
    missing_pdf_names = sorted(name for name in all_pdf_names if name not in ingested_names)

    return {
        "kb_ingestion": {
            "embedding": embedding,
            "local_pdf_count": len(all_pdf_names),
            "local_pdf_dirs": {k: len(v) for k, v in local_pdfs.items()},
            "approved_pg_documents": pg.get("approved_documents"),
            "approved_pg_chunks": pg.get("approved_chunks"),
            "missing_local_pdfs_in_pg_by_filename": missing_pdf_names[:200],
            "auto_ingest_dirs": os.getenv("AUTO_INGEST_KB_DIRS") or os.getenv("AUTO_INGEST_KB_DIR"),
            "auto_ingest_max_files": os.getenv("AUTO_INGEST_MAX_FILES"),
        },
        "dynamic_data": {
            "cache_table": "dynamic_knowledge_cache",
            "weather": {"provider": "Open-Meteo", "cache_ttl_sec": os.getenv("RAG_WEATHER_CACHE_TTL_SEC", "7200")},
            "soil": {"provider": "SoilGrids/ISRIC", "cache_ttl_sec": os.getenv("RAG_SOIL_CACHE_TTL_SEC", str(180 * 24 * 3600))},
            "market": {
                "current_provider": "local market_prices table with mock demo fallback",
                "nmis_live_adapter": False,
                "planned_sources": ["NMIS", "ECX", "ESS", "FAO", "manual CSV/Excel"],
            },
        },
        "voice_pipeline": {
            "asr_to_rag": "vad_service calls POST /rag/answer with ASR transcript",
            "rag_to_tts": "vad_service sends response text to tts_service /synthesize",
            "rag_service_url_env": os.getenv("RAG_SERVICE_URL", "not set in rag-service"),
            "tts_url": TTS_URL,
            "stt_url": STT_URL,
        },
        "nlu": {
            "current": "rule-based multilingual NLU",
            "afroxlmr_loaded": False,
            "afroxlmr_plugin_boundary": "farmer_rag_stack.smart_advisory.classify_intent_and_entities",
        },
        "performance": {
            "response_cache_ttl_sec": os.getenv("RAG_RESPONSE_CACHE_TTL_SEC", "0"),
            "smart_pipeline": os.getenv("RAG_SMART_PIPELINE", "1"),
            "final_backend": os.getenv("RAG_SMART_FINAL_BACKEND", "gemini"),
            "web_mode": os.getenv("RAG_WEB_MODE", "off"),
        },
    }


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


def check_and_deliver_expert_responses(phone_number: str) -> str:
    """Checks if there's an answered escalation for this phone number and returns the formatted response."""
    import rag_pg
    if not rag_pg.kb_pg_enabled():
        return ""
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
                    return f"ቀደም ብለው ስለ '{orig_query}' ላቀረቡት ጥያቄ የባለሙያ ምላሽ አለኝ፡ {expert_resp}\n\nአሁን ደግሞ ስለ አዲሱ ጥያቄዎ ልርዳዎት።"
    except Exception as exc:
        logger.error(f"Failed to check for expert responses: {exc}")
    return ""


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
    expert_delivery = check_and_deliver_expert_responses(phone_number)
    
    current_resp, intent, refs, nlu = _generate_core_rag_response(query_text, phone_number, session_id)
    
    final_text = f"{expert_delivery}\n\n{current_resp}" if expert_delivery else current_resp
    return final_text, intent, refs, nlu


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

    # ── User-requested escalation (FR22) ──────────────────────────────────────
    # A lightweight trigger: if the user explicitly asks for an expert/human.
    # We do this early, before retrieval, so it's deterministic.
    q = (query_text or "").lower()
    user_escalation_phrases = (
        "expert",
        "human",
        "agent",
        "helpdesk",
        "operator",
        "ለባለሙያ",
        "ባለሙያ",
        "ሰው",
        "ወደ ባለሙያ",
        "እርዳታ",
    )
    if any(p in q for p in user_escalation_phrases):
        add_to_escalation(
            query_text,
            "User explicitly requested expert handoff.",
            phone_number=phone_number,
            session_id=session_id,
            reason_code="USER_REQUESTED",
            confidence=nlu.confidence,
            entities=nlu.entities,
        )
        resp = "እሺ፣ ጥያቄዎን ለግብርና ባለሙያ አስተላልፌዋለሁ። በቅርቡ መልስ ያገኛሉ።"
        log_conversation(phone_number, session_id, "assistant", resp)
        log_interaction_record(
            phone_number=phone_number,
            session_id=session_id,
            intent=nlu.primary_intent,
            response_type="escalated_user_requested",
            entities=nlu.entities,
            confidence=nlu.confidence,
        )
        return resp, "escalated_user_requested", [], nlu.to_dict()

    # ── Farmer Profile & Context ──────────────────────────────────────────────
    profile = get_farmer_profile(phone_number)
    farmer_location = (profile or {}).get("location") or "Unknown"
    
    # Identify the relevant region for RAG filtering
    # Priority: 1. NLU extracted region, 2. Profile region
    user_region = nlu.entities.get("region_en")
    if not user_region and profile:
        # Map profile location to a region keyword if possible
        loc = str(profile.get('location', '')).lower()
        if any(k in loc for k in ["highland", "ደጋ"]): user_region = "highland"
        elif any(k in loc for k in ["lowland", "ቆላ"]): user_region = "lowland"
        elif any(k in loc for k in ["midland", "ወይና"]): user_region = "midland"

    intent = nlu.primary_intent
    user_context = build_personalization_block(phone_number, profile)
    if user_region:
        user_context = (user_context or "") + f"የክልል ማጣሪያ / ቦታ፦ {user_region}።\n"

    # ── Active Alerts ─────────────────────────────────────────────────────────
    alerts = get_alerts_for_region(farmer_location)
    alerts_text = f"ማሳሰቢያ: {alerts[0][0]}\n\n" if alerts else ""

    # ── Safety Confirmation State ─────────────────────────────────────────────
    state = get_session_state(session_id)
    if state and state.get("current_state") == "awaiting_confirmation":
        if "አዎ" in query_text or "yes" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = alerts_text + state["pending_action"]
            log_conversation(phone_number, session_id, "assistant", resp)
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent=nlu.primary_intent,
                response_type="confirmed_action" if "አዎ" in query_text or "yes" in query_text.lower() else "safety_confirmation",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "confirmed_action", [], nlu.to_dict()
        elif "አይ" in query_text or "no" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = "እሺ፣ እርምጃው ተሰርዟል። ሌላ ምን ልርዳዎት?"
            log_conversation(phone_number, session_id, "assistant", resp)
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent=nlu.primary_intent,
                response_type="cancelled_action",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "cancelled_action", [], nlu.to_dict()
        else:
            resp = "እባክዎን 'አዎ' ወይም 'አይ' ብለው ያረጋግጡ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent=nlu.primary_intent,
                response_type="safety_confirmation",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "awaiting_confirmation", [], nlu.to_dict()

    # ── Slot Awaiting State ───────────────────────────────────────────────────
    if state and state.get("current_state") == "awaiting_slot":
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
        log_interaction_record(
            phone_number=phone_number,
            session_id=session_id,
            intent=nlu.primary_intent,
            response_type="awaiting_slot",
            entities=nlu.entities,
            confidence=nlu.confidence,
        )
        return clarification, "awaiting_slot", [], nlu.to_dict()

    # ── Complex Query Escalation ──────────────────────────────────────────────
    sensitive_intents = {"pest_disease", "soil_fertility", "crop_production"}
    if intent in sensitive_intents and nlu.confidence < 0.6:
        logger.warning(f"Complex query detected (intent={intent} conf={nlu.confidence}). Escalating.")
        add_to_escalation(
            query_text,
            f"Complex {intent} query with low NLU confidence ({nlu.confidence:.2f}).",
            phone_number=phone_number,
            session_id=session_id,
            reason_code="COMPLEX_QUERY",
            confidence=nlu.confidence,
            entities=nlu.entities,
        )
        resp = "ይህ ጥያቄ ዝርዝር መረጃ ስለሚያስፈልገው ለግብርና ባለሙያ አስተላልፌዋለሁ። በቅርቡ መልስ ያገኛሉ።"
        log_conversation(phone_number, session_id, "assistant", resp)
        log_interaction_record(
            phone_number=phone_number,
            session_id=session_id,
            intent=intent,
            response_type="escalated_complex",
            entities=nlu.entities,
            confidence=nlu.confidence,
        )
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
                log_interaction_record(
                    phone_number=phone_number,
                    session_id=session_id,
                    intent="market_price",
                    response_type="market_price",
                    entities=nlu.entities,
                    confidence=nlu.confidence,
                )
                return resp, "market_price", [], nlu.to_dict()
            else:
                dyn_key = f"market_price_{crop_name}_{farmer_location}"
                cached = get_dynamic_knowledge(dyn_key)
                if cached:
                    log_conversation(phone_number, session_id, "assistant", cached)
                    log_interaction_record(phone_number, session_id, "market_price", "market_price_dynamic", nlu.entities, nlu.confidence)
                    return cached, "market_price_dynamic", [], nlu.to_dict()

                from farmer_rag_stack.rag_tools import web_search
                from farmer_rag_stack.query_llm import run_sync_llm
                from farmer_rag_stack.llm_providers import effective_llm_backend
                
                snippets = web_search.fetch_web_snippets(f"current market price {crop_name} {farmer_location} Ethiopia", max_results=3)
                if snippets:
                    web_text = web_search.format_web_block(snippets)
                    prompt = f"መመሪያ: ከታች ካለው መረጃ የ {crop_name} የአሁኑን የገበያ ዋጋ ፈልግ እና በአጭሩ በአማርኛ ንገረኝ። ዋጋ ከሌለ 'አልተገኘም' በል።\n\nመረጃ:\n{web_text}"
                    msgs = [{"role": "user", "content": prompt}]
                    try:
                        ans, _ = run_sync_llm(effective_llm_backend(), msgs, fast=True)
                        if "አልተገኘም" not in ans and len(ans) > 5:
                            set_dynamic_knowledge(dyn_key, ans)
                            log_conversation(phone_number, session_id, "assistant", ans)
                            log_interaction_record(phone_number, session_id, "market_price", "market_price_dynamic", nlu.entities, nlu.confidence)
                            return ans, "market_price_dynamic", [], nlu.to_dict()
                    except Exception:
                        pass

                resp = f"ለ{crop_name} ዋጋ መረጃ አሁን የለም። ቆይተው ይደውሉ።"
                log_conversation(phone_number, session_id, "assistant", resp)
                log_interaction_record(
                    phone_number=phone_number,
                    session_id=session_id,
                    intent="market_price",
                    response_type="market_price_unavailable",
                    entities=nlu.entities,
                    confidence=nlu.confidence,
                )
                return resp, "market_price_unavailable", [], nlu.to_dict()
        else:
            # Crop not specified
            resp = "ስለ ምን ሰብል ዋጋ ይፈልጋሉ? (ጤፍ፣ ስንዴ፣ ቦሎቄ፣ ወዘተ.)"
            set_session_state(session_id, "awaiting_slot", query_text)
            log_conversation(phone_number, session_id, "assistant", resp)
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent="market_price",
                response_type="awaiting_slot",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "awaiting_slot", [], nlu.to_dict()

    # ── RAG: Postgres+pgvector (preferred) or legacy Chroma ───────────────────
    import rag_pg

    references: list = []
    context: str | None = None
    hits: list[dict] = []
    closest_distance = 999.0
    use_pg = rag_pg.kb_pg_enabled() and rag_pg.count_approved_chunks() > 0
    history_pairs = get_conversation_history(session_id, limit=6)
    conv_msgs = [{"role": r, "content": (m or "").strip()} for r, m in history_pairs if (m or "").strip()]
    from farmer_rag_stack.context_utils import retrieval_query_for
    from farmer_rag_stack.assistant import try_llm_assistant_response
    from farmer_rag_stack.nlu_farmer import augment_retrieval_query_with_nlu, parse_farmer_nlu
    from farmer_rag_stack.retrieval_ranking import rank_pg_hits
    from chroma_retrieve import merge_pg_chroma_hits, retrieve_chroma_mirror_hits

    farmer_nlu = parse_farmer_nlu(query_text)
    retrieval_query = augment_retrieval_query_with_nlu(
        retrieval_query_for(nlu.retrieval_query or query_text, conv_msgs),
        farmer_nlu,
    )

    if use_pg:
        pool = max(12, int(os.environ.get("RAG_PG_RETRIEVE_POOL", "40")))
        hits, closest_distance = rag_pg.retrieve_for_query(
            retrieval_query,
            top_k=pool,
            max_l2_distance=RAG_PG_MAX_L2_DISTANCE,
            region=user_region,
        )

        pg_hits = [h for h in hits if float(h.get("distance", 999)) <= RAG_PG_MAX_L2_DISTANCE]
        chroma_k = max(8, int(os.getenv("RAG_CHROMA_TOP_K", "18").strip() or "18"))
        chroma_hits = retrieve_chroma_mirror_hits(retrieval_query, top_k=chroma_k)
        hits = merge_pg_chroma_hits(pg_hits, chroma_hits)

        if not hits:
            dyn_key = f"soil_kb_{query_text}"
            cached = get_dynamic_knowledge(dyn_key)
            if cached:
                log_conversation(phone_number, session_id, "assistant", cached)
                log_interaction_record(phone_number, session_id, intent, "dynamic_kb_answer", nlu.entities, nlu.confidence)
                return cached, "dynamic_kb_answer", [], nlu.to_dict()

            from farmer_rag_stack.rag_tools import web_search
            from farmer_rag_stack.query_llm import run_sync_llm
            from farmer_rag_stack.llm_providers import effective_llm_backend
            snippets = web_search.fetch_web_snippets(f"Ethiopia agriculture {query_text}", max_results=4)
            if snippets:
                web_text = web_search.format_web_block(snippets)
                prompt = f"መመሪያ: ከታች ካለው የድር መረጃ በመነሳት ለጥያቄው አጭር እና ትክክለኛ ምላሽ በአማርኛ ስጥ። መረጃው የማይጠቅም ከሆነ 'መረጃ የለም' በል።\n\nጥያቄ: {query_text}\n\nመረጃ:\n{web_text}"
                msgs = [{"role": "user", "content": prompt}]
                try:
                    ans, _ = run_sync_llm(effective_llm_backend(), msgs, fast=True)
                    if "መረጃ የለም" not in ans and len(ans) > 10:
                        set_dynamic_knowledge(dyn_key, ans)
                        log_conversation(phone_number, session_id, "assistant", ans)
                        log_interaction_record(phone_number, session_id, intent, "dynamic_kb_answer", nlu.entities, nlu.confidence)
                        return ans, "dynamic_kb_answer", [], nlu.to_dict()
                except Exception:
                    pass

            logger.info(f"No results found for query: {query_text}. Escalating.")
            add_to_escalation(
                query_text,
                "No relevant KB documents found and web fallback yielded no useful info.",
                phone_number=phone_number,
                session_id=session_id,
                reason_code="NO_KB_HITS"
            )
            resp = "ይቅርታ፣ ለዚህ ጥያቄ በቂ መረጃ አልተገኘም። ጥያቄዎን ለግብርና ባለሙያ ልከናል፤ በቅርቡ መልስ ያገኛሉ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent=intent,
                response_type="escalated_no_kb_hits",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "escalated", [], nlu.to_dict()

        def _retrieve_more_boosted(q: str) -> list:
            hh, _ = rag_pg.retrieve_for_query(
                q,
                top_k=pool,
                max_l2_distance=RAG_PG_MAX_L2_DISTANCE,
                region=user_region,
            )
            return [h for h in hh if float(h.get("distance", 999)) <= RAG_PG_MAX_L2_DISTANCE]

        keep = max(4, int(os.environ.get("RAG_PG_FINAL_TOP_K", "6")))
        hits = rank_pg_hits(
            query_text,
            retrieval_query,
            hits,
            farmer_nlu,
            retrieve_more=_retrieve_more_boosted,
            max_hits=max(keep, 8),
        )
        hits = hits[:keep]

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
            log_interaction_record(
                phone_number=phone_number,
                session_id=session_id,
                intent=nlu.primary_intent,
                response_type="escalated_low_confidence",
                entities=nlu.entities,
                confidence=nlu.confidence,
            )
            return resp, "escalated", [], nlu.to_dict()

        context = results["documents"][0][0]
        hits = [{"content": context}]

    intent = nlu.primary_intent

    history = get_conversation_history(session_id, limit=3)
    history_str = "\n".join([f"{h[0]}: {h[1]}" for h in history])

    dyn_block = ""
    try:
        dyn_block = build_dynamic_context(phone_number, crop_name=nlu.entities.get("crop_en")) or ""
    except Exception:
        dyn_block = ""

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
        response_text = None
        if use_pg and hits:
            response_text = try_llm_assistant_response(
                query_text=query_text,
                session_id=session_id,
                hits=hits,
                user_context=user_context,
                alerts_text=alerts_text,
                dynamic_block=dyn_block,
                history_pairs=history_pairs,
            )
        if response_text is None and use_pg and hits:
            response_text = compose_grounded_answer_no_llm(query_text, hits)
            if len(response_text) > 280:
                response_text = response_text[:277] + "..."
        elif response_text is None:
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
        log_interaction_record(
            phone_number=phone_number,
            session_id=session_id,
            intent=nlu.primary_intent,
            response_type="requires_confirmation",
            entities=nlu.entities,
            confidence=nlu.confidence,
        )
        return resp, "requires_confirmation", references, nlu.to_dict()

    final_response = alerts_text + normalize_text(response_text)
    log_conversation(phone_number, session_id, "assistant", final_response)
    log_interaction_record(
        phone_number=phone_number,
        session_id=session_id,
        intent=nlu.primary_intent,
        response_type="rag_answer",
        entities=nlu.entities,
        confidence=nlu.confidence,
    )
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


def _voice_escalation_response(
    *,
    query_text: str,
    phone_number: str,
    session_id: str,
    expert_delivery: str,
    body: str,
    reason_code: str,
    escalation_context: str,
    best_distance: float,
    hits: list,
    t0: float,
    safety: dict | None = None,
    meta_reason: str,
    nlu,
) -> dict:
    add_to_escalation(
        query_text,
        escalation_context,
        phone_number=phone_number,
        session_id=session_id,
        reason_code=reason_code,
        confidence=float(best_distance) if best_distance is not None else None,
        entities=getattr(nlu, "entities", None),
    )
    final = f"{expert_delivery}\n\n{body}" if expert_delivery else body
    final = normalize_text(final)
    log_conversation(phone_number, session_id, "assistant", final)
    log_interaction_record(
        phone_number=phone_number,
        session_id=session_id,
        intent=getattr(nlu, "primary_intent", None),
        response_type="escalated_low_confidence"
        if reason_code == "LOW_CONFIDENCE"
        else "escalated_agrochemical",
        entities=getattr(nlu, "entities", None),
        confidence=getattr(nlu, "confidence", None),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    sla_h = int(os.getenv("ESCALATION_SLA_HOURS", "48") or "48")
    trust = {
        "sources": ["escalation"],
        "kb_chunks_used": len(hits or []),
        "sources_in_prompt": 0,
        "latency_ms": round(latency_ms, 1),
        "escalation_sla_target_hours": sla_h,
        "human_review": True,
        "grounding": "escalation",
    }
    if safety:
        trust["safety"] = safety
    final = maybe_append_trust_footer(final, sources=["escalation"])
    return {
        "response": final,
        "references": [],
        "best_distance": best_distance,
        "trust": trust,
        "meta": {"response_cache": "bypass", "reason": meta_reason},
    }


@app.post("/rag/answer")
async def rag_answer(req: RagAnswerRequest):
    """
    RAG endpoint for other services:
    - static retrieval: Postgres+pgvector (rag_kb_*)
    - dynamic: alerts/market (``build_dynamic_context``)
    - generation: RAG-folder-style assistant (Groq/Gemini/Ollama) when enabled, else chunk composition
    """
    from farmer_rag_stack.assistant import try_llm_assistant_response
    from farmer_rag_stack.smart_advisory import run_smart_advisory

    query_text = (req.text or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Empty query")

    log_conversation(req.phone_number, req.session_id, "user", query_text)

    nlu = analyze_intent(query_text)
    expert_delivery = check_and_deliver_expert_responses(req.phone_number)
    
    crop_name = getattr(nlu, "entities", {}).get("crop_en") if nlu else None
    try:
        dyn = build_dynamic_context(req.phone_number, crop_name=crop_name)
    except Exception:
        dyn = ""

    # Identify region for filtering
    profile = get_farmer_profile(req.phone_number)
    user_region = nlu.entities.get("region_en")
    if not user_region and profile:
        loc = str(profile.get('location', '')).lower()
        if any(k in loc for k in ["highland", "ደጋ"]): user_region = "highland"
        elif any(k in loc for k in ["lowland", "ቆላ"]): user_region = "lowland"
        elif any(k in loc for k in ["midland", "ወይና"]): user_region = "midland"

    hist = get_conversation_history(req.session_id, limit=6)
    hist_pairs = list(hist)

    cache_allowed = not expert_delivery and not (dyn or "").strip()
    cache_key = (
        response_cache.make_rag_cache_key(
            query_text=query_text,
            phone_number=req.phone_number,
            user_region=user_region or "",
        )
        if cache_allowed
        else None
    )
    if cache_key:
        cached = response_cache.get(cache_key)
        if cached:
            out_hit = dict(cached)
            out_hit["meta"] = {"response_cache": "hit"}
            cached_resp = (out_hit.get("response") or "").strip()
            if cached_resp:
                log_conversation(req.phone_number, req.session_id, "assistant", cached_resp)
            return out_hit

    t0 = time.perf_counter()
    max_d = float(os.environ.get("RAG_PG_MAX_L2_DISTANCE", str(RAG_PG_MAX_L2_DISTANCE)))
    try:
        hits, _rq, _farmer_nlu, best = ranked_hits_for_voice_query(
            query_text=query_text,
            nlu=nlu,
            user_region=user_region,
            hist_pairs=hist_pairs,
            max_l2_distance=max_d,
        )
    except (FileNotFoundError, OSError) as exc:
        logger.error("RAG embedding model is missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "embedding_model_missing_or_incomplete",
                "message": str(exc),
                "fix": "Run `python download_models.py`, or set KB_EMBEDDING_MODEL to an existing SentenceTransformer path/model.",
            },
        ) from exc

    agro_max = chemical_safety.agrochemical_max_l2_distance(max_d)
    kb_grounded = voice_guards.kb_grounded_for_voice(hits, best, max_d)
    agro_kb_grounded = voice_guards.kb_grounded_for_voice(hits, best, agro_max)
    is_agro = chemical_safety.is_high_risk_agrochemical_query(query_text)

    if (
        chemical_safety.agrochemical_expert_only_enabled()
        and is_agro
        and not agro_kb_grounded
    ):
        reason = "no_kb_hits" if not hits else "low_kb_confidence"
        return _voice_escalation_response(
            query_text=query_text,
            phone_number=req.phone_number,
            session_id=req.session_id,
            expert_delivery=expert_delivery,
            body=chemical_safety.CANNED_AGROCHEM_ESCALATION_AM,
            reason_code="AGROCHEM_NO_KB",
            escalation_context=(
                f"Agrochemical query; expert-only path ({reason}, best_distance={best:.3f}, max={agro_max})."
            ),
            best_distance=best,
            hits=hits,
            t0=t0,
            safety={"agrochemical_expert_only": True, "reason": reason},
            meta_reason="agrochemical_escalation",
            nlu=nlu,
        )

    if voice_guards.voice_low_conf_escalation_enabled() and not kb_grounded and not is_agro:
        return _voice_escalation_response(
            query_text=query_text,
            phone_number=req.phone_number,
            session_id=req.session_id,
            expert_delivery=expert_delivery,
            body=voice_guards.GENERIC_LOW_CONFIDENCE_ESCALATION_AM,
            reason_code="LOW_CONFIDENCE",
            escalation_context=(
                f"Voice path: no confident KB match (best_distance={best:.3f}, max={max_d})."
            ),
            best_distance=best,
            hits=hits,
            t0=t0,
            meta_reason="low_confidence_escalation",
            nlu=nlu,
        )

    hits = voice_guards.confident_kb_hits(hits, max_d)

    profile_line = build_personalization_block(req.phone_number, profile)
    if user_region:
        profile_line = (profile_line or "") + f"የክልል ማጣሪያ፦ {user_region}።\n"

    answer = ""
    used_llm = False
    used_compose = False
    smart_context = None
    smart_tool_trace = []
    if os.environ.get("RAG_SMART_PIPELINE", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            smart = run_smart_advisory(
                question=query_text,
                phone_number=req.phone_number,
                nlu=nlu,
                profile=profile,
                history_pairs=hist_pairs,
                hits=hits,
                local_market_price_func=get_market_price,
            )
            if smart.answer:
                answer = smart.answer
                used_llm = smart.used_llm
                smart_context = smart.context
                smart_tool_trace = smart.tool_trace
        except Exception as exc:
            logger.warning("Smart advisory pipeline failed; falling back to legacy RAG path: %s", exc)

    if hits:
        if not answer.strip():
            llm_try = (
                try_llm_assistant_response(
                    query_text=query_text,
                    session_id=req.session_id,
                    hits=hits,
                    user_context=profile_line,
                    alerts_text="",
                    dynamic_block=dyn or "",
                    history_pairs=hist_pairs,
                )
                or ""
            )
            if llm_try.strip():
                answer = llm_try
                used_llm = True
    if not answer.strip() and hits:
        answer = compose_grounded_answer_no_llm(query_text, hits)
        if (answer or "").strip():
            used_compose = True

    if dyn and answer:
        final = f"{dyn}\n\n{answer}"
    elif dyn:
        final = dyn
    else:
        final = answer or ""

    if expert_delivery:
        final = f"{expert_delivery}\n\n{final}"

    escalated_empty = False
    if not final:
        escalated_empty = True
        add_to_escalation(
            query_text,
            "Empty response in voice pipeline.",
            phone_number=req.phone_number,
            session_id=req.session_id,
            reason_code="EMPTY_VOICE_RESP"
        )
        final = "ይቅርታ፣ ለዚህ ጥያቄ በቂ መረጃ አልተገኘም። ጥያቄዎን ለግብርና ባለሙያ ልከናል፤ በቅርቡ መልስ ያገኛሉ።"

    voice_cap_raw = os.environ.get("RAG_VOICE_RAG_ANSWER_MAX_CHARS", "600").strip()
    try:
        voice_cap = int(voice_cap_raw) if voice_cap_raw else 600
    except ValueError:
        voice_cap = 600
    if voice_cap > 0 and len(final) > voice_cap:
        final = final[: max(0, voice_cap - 3)].rstrip() + "..."

    final = normalize_text(final)

    latency_ms = (time.perf_counter() - t0) * 1000
    sla_h = int(os.getenv("ESCALATION_SLA_HOURS", "48") or "48")
    src: list[str] = []
    if escalated_empty:
        src.append("escalation")
    else:
        if hits:
            src.append("kb")
        if smart_tool_trace:
            src.append("tools")
        if (dyn or "").strip():
            src.append("dynamic")
        if expert_delivery:
            src.append("expert_delivery")
    trust = build_voice_trust_meta(
        hits=hits,
        used_llm_assistant=used_llm,
        used_chunk_compose=used_compose,
        sources=src,
        escalated_empty=escalated_empty,
        latency_ms=latency_ms,
        sla_target_hours=sla_h,
    )
    final = maybe_append_trust_footer(final, sources=src)

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

    out = {"response": final, "references": refs, "best_distance": best, "trust": trust}
    if smart_tool_trace:
        out["tool_trace"] = smart_tool_trace
    if smart_context and os.environ.get("RAG_RETURN_SMART_CONTEXT", "0").strip().lower() in ("1", "true", "yes", "on"):
        out["smart_context"] = smart_context
    if cache_key and not escalated_empty:
        g = trust.get("grounding") if isinstance(trust, dict) else None
        if hits and g in ("kb_llm", "kb_compose"):
            response_cache.set(cache_key, out)
    if final:
        log_conversation(req.phone_number, req.session_id, "assistant", final)
        log_interaction_record(
            phone_number=req.phone_number,
            session_id=req.session_id,
            intent=getattr(nlu, "primary_intent", None),
            response_type="escalated_empty" if escalated_empty else "rag_answer",
            entities=getattr(nlu, "entities", None),
            confidence=getattr(nlu, "confidence", None),
        )
    return out


@app.post("/rag/debug/context", dependencies=[Depends(_require_metrics_token)])
async def rag_debug_context(req: RagDebugContextRequest):
    """
    Build the exact structured context used by the smart advisory pipeline,
    without calling Gemini. Useful for chat/session tests and cost-free debugging.
    """
    from farmer_rag_stack.smart_advisory import build_smart_context_only

    query_text = (req.text or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Empty query")

    nlu = analyze_intent(query_text)
    profile = get_farmer_profile(req.phone_number)
    user_region = nlu.entities.get("region_en")
    if not user_region and profile:
        loc = str(profile.get("location", "")).lower()
        if any(k in loc for k in ["highland", "ደጋ"]):
            user_region = "highland"
        elif any(k in loc for k in ["lowland", "ቆላ"]):
            user_region = "lowland"
        elif any(k in loc for k in ["midland", "ወይና"]):
            user_region = "midland"

    hist_pairs = list(get_conversation_history(req.session_id, limit=8))
    hits: list[dict] = []
    best = 999.0
    retrieval_query = ""
    if req.retrieve:
        max_d = float(os.environ.get("RAG_PG_MAX_L2_DISTANCE", str(RAG_PG_MAX_L2_DISTANCE)))
        try:
            hits, retrieval_query, _farmer_nlu, best = ranked_hits_for_voice_query(
                query_text=query_text,
                nlu=nlu,
                user_region=user_region,
                hist_pairs=hist_pairs,
                max_l2_distance=max_d,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.error("RAG embedding model is missing: %s", exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "embedding_model_missing_or_incomplete",
                    "message": str(exc),
                    "fix": "Run `python download_models.py`, or set KB_EMBEDDING_MODEL to an existing SentenceTransformer path/model.",
                },
            ) from exc

    context, tool_trace, kb_refs = build_smart_context_only(
        question=query_text,
        phone_number=req.phone_number,
        nlu=nlu,
        profile=profile,
        history_pairs=hist_pairs,
        hits=hits,
        local_market_price_func=get_market_price,
    )
    return {
        "context": context,
        "tool_trace": tool_trace,
        "references": kb_refs[:5],
        "retrieval": {
            "enabled": req.retrieve,
            "query": retrieval_query,
            "best_distance": best,
            "hit_count": len(hits),
            "user_region": user_region,
        },
        "session_history_count": len(hist_pairs),
    }


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
