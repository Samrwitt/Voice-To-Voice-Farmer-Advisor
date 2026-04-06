from fastapi import FastAPI
from pydantic import BaseModel
import os
import random
import logging
from database import collection, add_to_escalation, log_conversation, get_conversation_history, get_market_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("logic_service")

app = FastAPI()

class Query(BaseModel):
    text: str
    phone_number: str = "Unknown"
    session_id: str = "default_session"

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
        prompt = f"Context: {context}\\nHistory: {history_str}\\nQuestion: {query_text}\\nAnswer directly in Amharic:"
        response_text = llm(prompt)
    else:
        response_text = context
        
    log_conversation(phone_number, session_id, "assistant", response_text)
    return response_text, intent

@app.post("/ask")
async def process_query(query: Query):
    response_text, intent = generate_rag_response(query.text, query.phone_number, query.session_id)
    return {"response": response_text, "intent": intent}
