import os
import httpx
import logging

logger = logging.getLogger("rag_client")

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")

async def get_rag_answer(
    text: str,
    session_id: str,
    phone_number: str = "Unknown",
    asr_meta: dict | None = None,
    utterance_path: str | None = None,
) -> dict:
    """
    Call the RAG service to get a grounded answer for the transcript.
    """
    url = f"{RAG_SERVICE_URL}/rag/answer"
    
    payload = {
        "text": text,
        "session_id": session_id,
        "phone_number": phone_number
    }
    if asr_meta:
        payload["asr"] = asr_meta
    if utterance_path:
        payload["utterance_path"] = utterance_path

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"RAG request failed: {e}")
        return {"response": f"Error calling RAG service: {e}", "references": []}
