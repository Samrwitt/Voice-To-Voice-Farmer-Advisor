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

DB_PATH = "/data/advisor.db"

# Initialize SQLite for all entities
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Escalated Queries
    c.execute('''CREATE TABLE IF NOT EXISTS escalated_queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT NOT NULL,
                  context TEXT,
                  status TEXT DEFAULT 'pending',
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Farmers Profile
    c.execute('''CREATE TABLE IF NOT EXISTS farmers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  phone_number TEXT UNIQUE NOT NULL,
                  name TEXT,
                  location TEXT,
                  preferred_language TEXT DEFAULT 'am',
                  registered_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Conversation History
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  phone_number TEXT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  message TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Market Prices
    c.execute('''CREATE TABLE IF NOT EXISTS market_prices
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  crop_name TEXT NOT NULL,
                  region TEXT NOT NULL,
                  price REAL NOT NULL,
                  unit TEXT NOT NULL,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Admin Users
    c.execute('''CREATE TABLE IF NOT EXISTS admin_users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

def add_to_escalation(query: str, context: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO escalated_queries (query, context) VALUES (?, ?)", (query, context))
    conn.commit()
    conn.close()

def log_conversation(phone_number: str, session_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversation_history (phone_number, session_id, role, message) VALUES (?, ?, ?, ?)", 
              (phone_number, session_id, role, message))
    conn.commit()
    conn.close()

def get_conversation_history(session_id: str, limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, message FROM conversation_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit))
    history = c.fetchall()
    conn.close()
    return list(reversed(history))

def get_market_price(crop_name: str, region: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if region:
        c.execute("SELECT price, unit FROM market_prices WHERE crop_name = ? AND region = ? ORDER BY updated_at DESC LIMIT 1", (crop_name, region))
    else:
        c.execute("SELECT price, unit FROM market_prices WHERE crop_name = ? ORDER BY updated_at DESC LIMIT 1", (crop_name,))
    result = c.fetchone()
    conn.close()
    return result

init_kb()
init_db()
