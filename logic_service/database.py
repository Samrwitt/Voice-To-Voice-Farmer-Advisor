import chromadb
from chromadb.utils import embedding_functions
import json
import sqlite3
import os

# Initialize Vector DB for RAG
chroma_client = chromadb.PersistentClient(path="/data/chroma_db")
# Using a lightweight sentence transformer for embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

collection = chroma_client.get_or_create_collection(name="agronomy_kb", embedding_function=sentence_transformer_ef)

def init_kb():
    if collection.count() == 0:
        # Load mock KB
        if os.path.exists("mock_kb.json"):
            with open("mock_kb.json", "r", encoding="utf-8") as f:
                kb_data = json.load(f)
                
            documents = []
            metadatas = []
            ids = []
            
            for i, (intent, response) in enumerate(kb_data.items()):
                if intent == "unknown":
                    continue
                # Mapping intent keywords to the Amharic response
                documents.append(response)
                metadatas.append({"intent": intent})
                ids.append(str(i))
                
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print("Knowledge Base Initialized.")

# Initialize SQLite for Escalation Queue
def init_db():
    conn = sqlite3.connect("/data/escalation_queue.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalated_queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT NOT NULL,
                  context TEXT,
                  status TEXT DEFAULT 'pending',
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_to_escalation(query: str, context: str):
    conn = sqlite3.connect("/data/escalation_queue.db")
    c = conn.cursor()
    c.execute("INSERT INTO escalated_queries (query, context) VALUES (?, ?)", (query, context))
    conn.commit()
    conn.close()

init_kb()
init_db()
