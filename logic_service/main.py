from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import random
import logging
import requests
import base64
import time
from database import collection, add_to_escalation, log_conversation, get_conversation_history, get_market_price, register_farmer, get_farmer_profile, get_alerts_for_region, set_session_state, get_session_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("logic_service")

app = FastAPI()

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

# In a full deployment, load LLaMA-2 locally.
from langchain_community.llms import LlamaCpp

llm = None
# Look for model in a mounted data volume to avoid image bloating
model_path = "/data/models/llama-2-7b-chat.Q4_K_M.gguf"
if os.path.exists(model_path):
    logger.info("Initializing LLaMA-2 model for RAG Generation...")
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1, max_tokens=256, top_p=1,
        n_ctx=2048
    )
else:
    logger.warning("LLaMA-2 model not found at /data/models/. Responding with direct KB extraction.")

def generate_rag_response(query_text: str, phone_number: str, session_id: str):
    logger.info(f"Processing query for session {session_id} from {phone_number}")
    log_conversation(phone_number, session_id, "user", query_text)
    
    profile = get_farmer_profile(phone_number)
    farmer_location = profile['location'] if profile else "Unknown"
    user_context = f"Farmer Location: {farmer_location} " if profile else ""
    
    alerts_text = ""
    # Fetch active alerts and prepend
    alerts = get_alerts_for_region(farmer_location)
    if alerts:
        # Prepend just the first high-priority alert for conciseness
        alerts_text = f"ማሳሰቢያ: {alerts[0][0]}\n\n"
        
    # Check Session State for Safety Confirmations
    state = get_session_state(session_id)
    if state and state["current_state"] == "awaiting_confirmation":
        if "አዎ" in query_text or "yes" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = alerts_text + state["pending_action"]
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "confirmed_action"
        elif "አይ" in query_text or "no" in query_text.lower():
            set_session_state(session_id, "active", None)
            resp = "እሺ እርምጃው ተሰርዟል። ሌላ ምን ልርዳዎት?" # Action cancelled. How else can I help?
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "cancelled_action"
        else:
            resp = "እባክዎን እርግጠኛ ከሆኑ 'አዎ'፣ ካልሆኑ 'አይ' በማለት ያረጋግጡ።" # Please confirm by saying Yes or No.
            log_conversation(phone_number, session_id, "assistant", resp)
            return resp, "awaiting_confirmation"

    # Naive Market Data intent extraction
    market_keywords = ["ዋጋ", "ስንት ነው", "ገበያ", "price", "market"]
    if any(k in query_text.lower() for k in market_keywords):
        logger.info("Market price intent detected.")
        if "ጤፍ" in query_text or "teff" in query_text.lower():
            p = get_market_price("Teff")
            if p:
                resp = f"የጤፍ ዋጋ {p[0]} ብር በ {p[1]} ነው።"
                log_conversation(phone_number, session_id, "assistant", resp)
                return resp, "market_price"

    # Retrieve closest match from ChromaDB
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    
    if not results["documents"] or not results["documents"][0]:
        distances = [999]
    else:
        distances = results["distances"][0]
        
    closest_distance = distances[0] if distances else 999
    
    # Guardrail: High risk or ambiguous
    if closest_distance > 1.2:  # Threshold needs tuning
        logger.warning(f"Query '{query_text}' distance {closest_distance} exceeds threshold 1.2. Escalating.")
        add_to_escalation(query_text, "No confident match in Knowledge Base.")
        resp = "ይቅርታ፣ ይህንን ጥያቄ መመለስ አልችልም። ለባለሙያ አስተላልፌዋለሁ።" # Sorry, I can't answer. Escalated to expert.
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "escalated"

    context = results["documents"][0][0]
    intent = results["metadatas"][0][0].get("intent", "unknown")
    
    history = get_conversation_history(session_id, limit=3)
    history_str = "\\n".join([f"{h[0]}: {h[1]}" for h in history])
    
    if llm:
        logger.info("Invoking LLM for generated response...")
        prompt = f"Context: {user_context}{context}\nHistory: {history_str}\nQuestion: {query_text}\nAnswer directly in Amharic:"
        response_text = llm(prompt)
    else:
        response_text = context
        
    # High-Risk Safety Interceptor
    high_risk_keywords = ["pest", "chemical", "disease", "fertilizer", "spray"]
    if any(hk in intent.lower() for hk in high_risk_keywords) or any(hk in query_text.lower() for hk in high_risk_keywords):
        logger.warning(f"High risk topic detected. Enforcing confirmation for session {session_id}")
        set_session_state(session_id, "awaiting_confirmation", response_text)
        resp = alerts_text + "ይህ እርምጃ ጥንቃቄ ይፈልጋል። እርግጠኛ ነዎት? (አዎ ወይም አይ ይበሉ)" # Action requires care. Are you sure? Say yes or no.
        log_conversation(phone_number, session_id, "assistant", resp)
        return resp, "requires_confirmation"
        
    final_response = alerts_text + response_text
    log_conversation(phone_number, session_id, "assistant", final_response)
    return final_response, intent

@app.post("/ask")
async def process_query(query: Query):
    response_text, intent = generate_rag_response(query.text, query.phone_number, query.session_id)
    return {"response": response_text, "intent": intent}

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

@app.post("/simulate_call")
async def simulate_call(req: E2ERequest):
    transcribed_text = req.text_input
    response_text, intent = generate_rag_response(transcribed_text, req.phone_number, req.session_id)
    
    tts_url = os.environ.get("TTS_URL", "http://tts_service:8002/synthesize")
    audio_b64 = None
    try:
        tts_resp = requests.post(tts_url, json={"text": response_text})
        if tts_resp.status_code == 200:
            audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
        else:
            logger.error(f"TTS Error: HTTP {tts_resp.status_code}")
    except Exception as e:
        logger.error(f"TTS Connection Error: {e}")

    return {
        "stt_output": transcribed_text,
        "logic_intent": intent,
        "logic_response": response_text,
        "audio_base64_length": len(audio_b64) if audio_b64 else 0
    }

@app.get("/system_check")
async def system_check():
    results = {}
    
    # 1. DB Check
    try:
        from database import DB_PATH
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.cursor().execute("SELECT 1")
        conn.close()
        results["database"] = "ok"
    except Exception as e:
        results["database"] = f"error: {str(e)}"
        
    # 2. ChromaDB Check
    try:
        if collection.count() >= 0:
            results["chroma_db"] = "ok"
    except Exception as e:
        results["chroma_db"] = f"error: {str(e)}"
        
    # 3. STT Connectivity Check
    try:
        stt_url = os.environ.get("STT_URL", "http://stt_service:8000/docs")
        resp = requests.get(stt_url, timeout=2)
        results["stt_service"] = "ok" if resp.status_code == 200 else f"unexpected status {resp.status_code}"
    except Exception as e:
        results["stt_service"] = f"error: {str(e)}"
        
    # 4. TTS Connectivity Check
    try:
        tts_url = os.environ.get("TTS_URL", "http://tts_service:8002/docs")
        resp = requests.get(tts_url, timeout=2)
        results["tts_service"] = "ok" if resp.status_code == 200 else f"unexpected status {resp.status_code}"
    except Exception as e:
        results["tts_service"] = f"error: {str(e)}"
        
    return results
