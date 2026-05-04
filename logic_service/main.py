from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
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
    get_session_state, insert_call_record
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("logic_service")

app = FastAPI()

# Mount admin REST API (used by the frontend microservice)
from admin_api import router as admin_router
app.include_router(admin_router)

# ── Config (externalized) ────────────────────────────────────────────────────
RAG_DISTANCE_THRESHOLD = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "1.2"))
TTS_URL = os.environ.get("TTS_URL", "http://tts_service:8002/synthesize")
STT_URL = os.environ.get("STT_URL", "http://stt_service:8000/transcribe")

# ── LLM Initialization ───────────────────────────────────────────────────────
from langchain_community.llms import LlamaCpp

llm = None
DATA_DIR = os.environ.get("DATA_DIR", "/data")
model_path = os.path.join(DATA_DIR, "models/llama-2-7b-chat.Q4_K_M.gguf")
if os.path.exists(model_path):
    logger.info("Initializing LLaMA-2 model for RAG generation...")
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1, max_tokens=256, top_p=1, n_ctx=2048
    )
else:
    logger.warning("LLaMA-2 model not found at /data/models/. Falling back to direct KB extraction.")


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


# ── Market Price Intent ──────────────────────────────────────────────────────
# Maps Amharic/English crop keywords to DB crop_name
CROP_KEYWORDS = {
    "ጤፍ": "Teff",      "teff": "Teff",
    "ስንዴ": "Wheat",    "wheat": "Wheat",
    "ቦሎቄ": "Maize",    "corn": "Maize",    "maize": "Maize",
    "ገብስ": "Barley",   "barley": "Barley",
    "ቦርኬ": "Sorghum",  "sorghum": "Sorghum",
    "ዳጉሳ": "Sorghum",
    "ሽምብራ": "Chickpea", "chickpea": "Chickpea",
    "ምስር": "Lentil",   "lentil": "Lentil",
    "ቅቤ": "Butter",
    "ቡና": "Coffee",    "coffee": "Coffee",
}

MARKET_KEYWORDS = ["ዋጋ", "ስንት ነው", "ስንት", "ገበያ", "price", "market", "cost", "ብር"]


def detect_market_intent(text: str):
    """Returns (True, crop_name) if a market price intent is detected, else (False, None)."""
    text_lower = text.lower()
    if not any(k in text_lower for k in MARKET_KEYWORDS):
        return False, None
    for keyword, crop_name in CROP_KEYWORDS.items():
        if keyword in text_lower:
            return True, crop_name
    return True, None  # Market intent but no specific crop identified


# ── Slot Filling ─────────────────────────────────────────────────────────────
CROP_ENTITY_WORDS = set(CROP_KEYWORDS.keys())
AGRI_INTENT_KEYWORDS = [
    "ማዳበሪያ", "ፀረ-ተባይ", "ፀረ ተባይ", "ምርት", "ዘር", "መዝራት",
    "fertilizer", "pest", "disease", "plant", "crop", "spray",
    "harvest", "ሰብል", "ለምለም", "ፍሬ", "ቅጠል"
]


def needs_slot_filling(text: str, session_state) -> Optional[str]:
    """
    Returns a clarifying question if a crop-related query lacks crop context.
    Returns None if no slot filling is needed.
    """
    if session_state and session_state.get("current_state") != "active":
        return None  # Already in a special state, don't interrupt

    text_lower = text.lower()
    has_agri_intent = any(k in text_lower for k in AGRI_INTENT_KEYWORDS)
    has_crop_entity = any(k in text_lower for k in CROP_ENTITY_WORDS)

    if has_agri_intent and not has_crop_entity:
        return "ለምን ሰብል ነው ጥያቄዎ? (ስንዴ፣ ጤፍ፣ ቦሎቄ፣ ወዘተ.)"  # Which crop is your question about?
    return None


# ── Core RAG Pipeline ────────────────────────────────────────────────────────
def generate_rag_response(query_text: str, phone_number: str, session_id: str):
    logger.info(f"Processing query for session={session_id} phone={phone_number}: '{query_text}'")
    log_conversation(phone_number, session_id, "user", query_text)

    # ── Language Check ────────────────────────────────────────────────────────
    if query_text.strip() and not is_amharic(query_text):
        resp = "እባክዎ ጥያቄዎን በአማርኛ ይናገሩ።"  # Please ask your question in Amharic.
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "non_amharic"

    # ── Farmer Profile & Context ──────────────────────────────────────────────
    profile = get_farmer_profile(phone_number)
    farmer_location = profile['location'] if profile else "Unknown"
    user_context = f"Farmer Location: {farmer_location}. " if profile else ""

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
            return resp, "confirmed_action"
        elif "አይ" in query_text or "no" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = "እሺ፣ እርምጃው ተሰርዟል። ሌላ ምን ልርዳዎት?"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "cancelled_action"
        else:
            resp = "እባክዎን 'አዎ' ወይም 'አይ' ብለው ያረጋግጡ።"
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "awaiting_confirmation"

    # ── Slot Awaiting State ───────────────────────────────────────────────────
    if state and state["current_state"] == "awaiting_slot":
        # user provided the missing crop/info; resume with enriched query
        original_query = state.get("pending_action", "")
        enriched_query = f"{original_query} {query_text}"
        set_session_state(session_id, "active", None)
        return generate_rag_response(enriched_query, phone_number, session_id)

    # ── Slot Filling Check ────────────────────────────────────────────────────
    clarification = needs_slot_filling(query_text, state)
    if clarification:
        set_session_state(session_id, "awaiting_slot", query_text)
        log_conversation(phone_number, session_id, "assistant", clarification)
        return clarification, "awaiting_slot"

    # ── Market Price Intent ───────────────────────────────────────────────────
    is_market, crop_name = detect_market_intent(query_text)
    if is_market:
        logger.info(f"Market price intent detected. Crop: {crop_name}")
        if crop_name:
            price_data = get_market_price(crop_name, farmer_location) or get_market_price(crop_name)
            if price_data:
                price, unit, updated_at = price_data
                resp = f"የ{crop_name} ዋጋ {price} ብር በ {unit} ነው። (የዋጋ ቀን: {updated_at})"
                log_conversation(phone_number, session_id, "assistant", resp)
                return resp, "market_price"
            else:
                resp = f"ለ{crop_name} ዋጋ መረጃ አሁን የለም። ቆይተው ይደውሉ።"
                log_conversation(phone_number, session_id, "assistant", resp)
                return resp, "market_price_unavailable"
        else:
            # Crop not specified
            resp = "ስለ ምን ሰብል ዋጋ ይፈልጋሉ? (ጤፍ፣ ስንዴ፣ ቦሎቄ፣ ወዘተ.)"
            set_session_state(session_id, "awaiting_slot", query_text)
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "awaiting_slot"

    # ── RAG Vector Retrieval ──────────────────────────────────────────────────
    results = collection.query(query_texts=[query_text], n_results=2)

    if not results["documents"] or not results["documents"][0]:
        distances = [999]
    else:
        distances = results["distances"][0]

    closest_distance = distances[0] if distances else 999

    # ── Escalation Guardrail ──────────────────────────────────────────────────
    if closest_distance > RAG_DISTANCE_THRESHOLD:
        logger.warning(f"Distance {closest_distance:.2f} > threshold {RAG_DISTANCE_THRESHOLD}. Escalating.")
        add_to_escalation(query_text, f"Distance: {closest_distance:.2f}. No confident KB match.")
        resp = "ይቅርታ፣ ይህንን ጥያቄ ሙሉ በሙሉ ልመልስ አልቻልኩም። ለባለሙያ አስተላልፌዋለሁ።"
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "escalated"

    context = results["documents"][0][0]
    intent = results["metadatas"][0][0].get("intent", "unknown")

    history = get_conversation_history(session_id, limit=3)
    history_str = "\n".join([f"{h[0]}: {h[1]}" for h in history])

    # ── LLM or Direct KB Response ─────────────────────────────────────────────
    if llm:
        logger.info("Invoking LLM for grounded response...")
        prompt = (
            f"Context: {user_context}{context}\n"
            f"History: {history_str}\n"
            f"Question: {query_text}\n"
            f"Answer directly in Amharic:"
        )
        response_text = llm(prompt)
    else:
        response_text = context

    # ── High-Risk Safety Interceptor ──────────────────────────────────────────
    high_risk_keywords = ["pest", "chemical", "disease", "fertilizer", "spray",
                          "ፀረ-ተባይ", "ማዳበሪያ", "ርጭት", "ፀረ"]
    if any(hk in intent.lower() for hk in high_risk_keywords) or \
       any(hk in query_text.lower() for hk in high_risk_keywords):
        logger.warning(f"High-risk topic detected for session {session_id}. Requiring confirmation.")
        set_session_state(session_id, "awaiting_confirmation", response_text)
        resp = alerts_text + "ይህ እርምጃ ጥንቃቄ ይፈልጋል። ስለ ሁኔታዎ እርግጠኛ ነዎት? (አዎ ወይም አይ)"
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "requires_confirmation"

    final_response = alerts_text + normalize_text(response_text)
    log_conversation(phone_number, session_id, "assistant", final_response)
    return final_response, intent


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/ask")
async def process_query(query: Query):
    response_text, intent = generate_rag_response(query.text, query.phone_number, query.session_id)
    return {"response": response_text, "intent": intent}


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
    response_text, intent = generate_rag_response(transcribed_text, req.phone_number, req.session_id)

    audio_b64 = None
    try:
        tts_resp = requests.post(TTS_URL, json={"text": response_text}, timeout=30)
        if tts_resp.status_code == 200:
            audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
        else:
            logger.error(f"TTS returned HTTP {tts_resp.status_code}")
    except Exception as e:
        logger.error(f"TTS request failed: {e}")

    return {
        "stt_output": transcribed_text,
        "logic_intent": intent,
        "logic_response": response_text,
        "audio_base64_length": len(audio_b64) if audio_b64 else 0
    }


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
        stt_check = os.environ.get("STT_URL", "http://stt_service:8000") + "/docs"
        resp = requests.get(stt_check, timeout=3)
        results["stt_service"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        results["stt_service"] = f"error: {e}"

    try:
        tts_check = os.environ.get("TTS_URL", "http://tts_service:8002").replace("/synthesize", "") + "/docs"
        resp = requests.get(tts_check, timeout=3)
        results["tts_service"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        results["tts_service"] = f"error: {e}"

    results["rag_threshold"] = RAG_DISTANCE_THRESHOLD
    results["llm_loaded"] = llm is not None

    return results
