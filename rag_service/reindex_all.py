import os
import json
import asyncio
import logging
import uuid
from typing import Dict, Any

import rag_pg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reindex_kb")

KB_FILE_PATH = "/app/kb_documents/amharic/kb_merged_same_structure.jsonl"

REGION_MAP = {
    "መላ ኢትዮጵያ": "ethiopia_all",
    "ደጋ": "highland",
    "ቆላ": "lowland",
    "ወይና ደጋ": "midland",
    "ደጋ እና ወይና ደጋ": "highland", # Approximation
    "ደጋ፣ ወይና ደጋ እና መስኖ አካባቢ": "highland",
    "ወይና ደጋ እና ቆላ": "midland"
}

def map_region(region_am: str) -> str:
    # Try exact match or substring
    for am, en in REGION_MAP.items():
        if am in region_am:
            return en
    return "ethiopia_all"

async def reindex():
    if not rag_pg.kb_pg_enabled():
        logger.error("Postgres KB not enabled. Check POSTGRES_URL.")
        return

    rag_pg.init_pg_schema()
    
    # 1. Clear existing data
    logger.info("Clearing existing RAG KB data...")
    psycopg = rag_pg._load_psycopg()
    with psycopg.connect(rag_pg.POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE rag_kb_chunks CASCADE;")
            cur.execute("TRUNCATE TABLE rag_kb_documents CASCADE;")

    # 2. Read JSONL file
    if not os.path.exists(KB_FILE_PATH):
        logger.error(f"KB file not found at {KB_FILE_PATH}")
        return

    logger.info(f"Indexing data from {KB_FILE_PATH}...")
    
    documents_to_insert = []
    
    with open(KB_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            doc_id = str(uuid.uuid4())
            external_id = item.get("id")
            crop = item.get("crop", "General")
            region_am = item.get("region", "General")
            content = item.get("text_am", "")
            
            if not content:
                continue
                
            region_en = map_region(region_am)
            
            documents_to_insert.append({
                "id": doc_id,
                "external_id": external_id,
                "title": f"KB Item {external_id} ({crop})",
                "content": content,
                "region": region_en,
                "extra": {
                    "crop": crop,
                    "region": region_en,
                    "region_am": region_am,
                    "kb_source": item.get("kb"),
                    "season": item.get("season")
                }
            })

    logger.info(f"Loaded {len(documents_to_insert)} documents. Chunking and embedding...")

    # 3. Process each document
    with psycopg.connect(rag_pg.POSTGRES_URL) as conn:
        for doc in documents_to_insert:
            # Insert document
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_kb_documents (id, external_document_id, title, extra)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (doc["id"], doc["external_id"], doc["title"], json.dumps(doc["extra"]))
                )
            
            # Chunk
            chunks = rag_pg.chunk_amharic_text(doc["content"])
            if not chunks:
                continue
                
            # Embed
            embeddings = rag_pg.embed_texts(chunks)
            
            # Insert chunks
            with conn.cursor() as cur:
                for i, (chunk_text, vec) in enumerate(zip(chunks, embeddings)):
                    cur.execute(
                        """
                        INSERT INTO rag_kb_chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (doc["id"], i, chunk_text, vec)
                    )
            
            logger.info(f"Indexed document {doc['external_id']} with {len(chunks)} chunks.")
        
        conn.commit()

    logger.info("Re-indexing complete!")

if __name__ == "__main__":
    asyncio.run(reindex())
