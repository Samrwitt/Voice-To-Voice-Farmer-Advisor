from fastapi import FastAPI
from pydantic import BaseModel
import os
import random
from database import collection, add_to_escalation

app = FastAPI()

class Query(BaseModel):
    text: str

# In a full deployment, load LLaMA-2 locally.
# from langchain_community.llms import LlamaCpp
# from langchain.prompts import PromptTemplate
# 
# llm = None
# if os.path.exists("./models/llama-2-7b-chat.Q4_K_M.gguf"):
#     llm = LlamaCpp(
#         model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
#         temperature=0.1, max_tokens=256, top_p=1
#     )

def generate_rag_response(query_text: str):
    # Retrieve closest match from ChromaDB
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    
    # Check if there is a match and if the distance is within tolerance (e.g. < 1.5)
    # Cosine distance depends on the embedding model.
    if not results["documents"][0]:
        distances = [999]
    else:
        distances = results["distances"][0]
        
    closest_distance = distances[0]
    
    # Guardrail: High risk or ambiguous
    if closest_distance > 1.2:  # Threshold needs tuning
        add_to_escalation(query_text, "No confident match in Knowledge Base.")
        return "ይቅርታ፣ ይህንን ጥያቄ መመለስ አልችልም። ለባለሙያ አስተላልፌዋለሁ።", "escalated" # Sorry, I can't answer. Escalated to expert.

    context = results["documents"][0][0]
    intent = results["metadatas"][0][0].get("intent", "unknown")
    
    # If LLM is loaded, we would use it here. For now, since LLaMA-2 weights are gigabytes,
    # we simulate the LLM summarization by just returning the matched context directly.
    # if llm:
    #     prompt = f"Context: {context}\nQuestion: {query_text}\nAnswer in Amharic:"
    #     response_text = llm(prompt)
    # else:
    response_text = context
        
    return response_text, intent

@app.post("/ask")
async def process_query(query: Query):
    response_text, intent = generate_rag_response(query.text)
    return {"response": response_text, "intent": intent}
