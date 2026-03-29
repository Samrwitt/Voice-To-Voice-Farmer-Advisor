from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

class Query(BaseModel):
    text: str

# Load Mock KB
with open("mock_kb.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Mock Intent Classification (Usually handled by TF-IDF, embeddings, or LLM)
def classify_intent(text: str) -> str:
    if "አየር" in text or "ዝናብ" in text:
        return "weather"
    elif "በሽታ" in text or "ተባይ" in text:
        return "disease"
    elif "ዋጋ" in text or "ገበያ" in text or "ማዳበሪያ" in text:
        return "market"
    return "unknown"

@app.post("/ask")
async def process_query(query: Query):
    text = query.text
    intent = classify_intent(text)
    
    # Retrieve Amharic RAG answer
    response_text = knowledge_base.get(intent, knowledge_base["unknown"])
    
    # We return the response text to the caller so they can pass it to TTS.
    # Alternatively, the logic service could call TTS directly, but decoupling
    # them via an orchestrator script (client) is cleaner for mocking.
    return {"response": response_text, "intent": intent}
