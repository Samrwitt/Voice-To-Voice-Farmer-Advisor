import json
import sqlite3
import os
from datetime import datetime
from typing import Optional, Any

# NOTE:
# This service is now optimized for Postgres+pgvector (rag_pg.py). ChromaDB is
# optional legacy fallback. We keep the symbol `collection` so older code paths
# can import it, but it may be None if chromadb isn't installed.
DATA_DIR = os.environ.get("DATA_DIR", "/data")

collection = None
try:
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore

    chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
    _emb_model = os.environ.get(
        "KB_EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_emb_model
    )
    collection = chroma_client.get_or_create_collection(
        name="agronomy_kb", embedding_function=sentence_transformer_ef
    )
except Exception:
    collection = None

def init_kb():
    if not collection:
        return
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

DB_PATH = os.path.join(DATA_DIR, "advisor.db")

# Initialize SQLite for all entities
def init_db():
    # Ensure the parent directory exists (important for RAG-only/local modes)
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
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

    # Alerts / Forecasts
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  target_region TEXT NOT NULL,
                  alert_message TEXT NOT NULL,
                  severity TEXT DEFAULT 'warning',
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Session States (For Multi-turn / Safety Confirmations)
    c.execute('''CREATE TABLE IF NOT EXISTS session_states
                 (session_id TEXT PRIMARY KEY,
                  current_state TEXT NOT NULL,
                  pending_action TEXT,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Call Records
    c.execute('''CREATE TABLE IF NOT EXISTS call_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT UNIQUE NOT NULL,
                  phone_number TEXT NOT NULL,
                  recording_path TEXT NOT NULL,
                  duration INTEGER NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Dynamic Knowledge Cache (for web search fallbacks)
    c.execute('''CREATE TABLE IF NOT EXISTS dynamic_knowledge_cache
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT UNIQUE NOT NULL,
                  content TEXT NOT NULL,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

POSTGRES_URL = os.environ.get("POSTGRES_URL", "").strip()

def _pg_enabled() -> bool:
    return bool(POSTGRES_URL)


def init_pg_app_tables():
    """
    Create the minimal Postgres tables needed for:
    - conversation history in dashboard
    - structured interaction records (intent/entities/response_type)

    We do NOT depend on SQLAlchemy models here; keep it lightweight.
    """
    if not _pg_enabled():
        return
    try:
        import psycopg
        with psycopg.connect(POSTGRES_URL, autocommit=False) as conn:
            with conn.cursor() as cur:
                # Matches logic_service/models.py table name
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_history (
                      id SERIAL PRIMARY KEY,
                      phone_number TEXT,
                      session_id TEXT NOT NULL,
                      role TEXT NOT NULL,
                      message TEXT NOT NULL,
                      timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversation_history_session ON conversation_history(session_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversation_history_phone ON conversation_history(phone_number);"
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interaction_records (
                      id SERIAL PRIMARY KEY,
                      phone_number TEXT,
                      session_id TEXT,
                      intent TEXT,
                      response_type TEXT,
                      entities JSONB,
                      confidence DOUBLE PRECISION,
                      created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interaction_records_session ON interaction_records(session_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interaction_records_phone ON interaction_records(phone_number);"
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS farmers_kb (
                      id SERIAL PRIMARY KEY,
                      phone_number TEXT UNIQUE NOT NULL,
                      name TEXT,
                      location TEXT,
                      preferred_language TEXT DEFAULT 'am',
                      crops JSONB,
                      farm_size DOUBLE PRECISION,
                      notes TEXT,
                      registered_at TIMESTAMPTZ DEFAULT NOW(),
                      updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                for ddl in (
                    "ALTER TABLE farmers_kb ADD COLUMN IF NOT EXISTS crops JSONB;",
                    "ALTER TABLE farmers_kb ADD COLUMN IF NOT EXISTS farm_size DOUBLE PRECISION;",
                    "ALTER TABLE farmers_kb ADD COLUMN IF NOT EXISTS notes TEXT;",
                    "ALTER TABLE farmers_kb ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();",
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_farmers_kb_phone ON farmers_kb(phone_number);",
                ):
                    cur.execute(ddl)

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS escalations (
                      id SERIAL PRIMARY KEY,
                      query TEXT NOT NULL,
                      context TEXT,
                      phone_number TEXT,
                      session_id TEXT,
                      status TEXT NOT NULL DEFAULT 'pending',
                      reason_code TEXT,
                      confidence DOUBLE PRECISION,
                      entities JSONB,
                      assigned_to_user_id TEXT,
                      assigned_at TIMESTAMPTZ,
                      expert_response TEXT,
                      expert_audio_path TEXT,
                      expert_notes TEXT,
                      transcript_snapshot TEXT,
                      session_recording_path TEXT,
                      answered_at TIMESTAMPTZ,
                      closed_at TIMESTAMPTZ,
                      created_at TIMESTAMPTZ DEFAULT NOW(),
                      updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                for ddl in (
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS expert_audio_path TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS expert_response TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS expert_notes TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS transcript_snapshot TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS session_recording_path TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS assigned_to_user_id TEXT;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS assigned_at TIMESTAMPTZ;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS answered_at TIMESTAMPTZ;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;",
                    "ALTER TABLE escalations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();",
                ):
                    cur.execute(ddl)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_escalations_phone ON escalations(phone_number);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_escalations_session ON escalations(session_id);")
            conn.commit()
    except Exception as exc:
        print(f"[DB] init_pg_app_tables failed: {exc}")


def _merge_escalation_entities(
    entities: dict | None,
    farmer_utterance_path: str | None = None,
) -> dict | None:
    from pathlib import Path

    merged = dict(entities or {})
    if farmer_utterance_path:
        merged["farmer_utterance_basename"] = Path(farmer_utterance_path).name
    return merged or None


def add_to_escalation(
    query: str,
    context: str,
    phone_number: str = None,
    session_id: str = None,
    reason_code: str = None,
    confidence: float = None,
    entities: dict = None,
    farmer_utterance_path: str | None = None,
):
    entities = _merge_escalation_entities(entities, farmer_utterance_path)
    if not POSTGRES_URL:
        # Fallback to legacy SQLite if Postgres is not configured
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO escalated_queries (query, context) VALUES (?, ?)", (query, context))
        conn.commit()
        conn.close()
        return

    try:
        import psycopg
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                transcript_snapshot = _build_transcript_snapshot(cur, session_id, query)
                session_recording_path = _get_session_recording_path(cur, session_id)
                cur.execute(
                    """
                    INSERT INTO escalations 
                    (
                      query, context, phone_number, session_id, reason_code,
                      confidence, entities, transcript_snapshot, session_recording_path,
                      status, created_at, updated_at
                    ) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """,
                    (
                        query,
                        context,
                        phone_number,
                        session_id,
                        reason_code,
                        confidence,
                        json.dumps(entities) if entities else None,
                        transcript_snapshot,
                        session_recording_path,
                        "pending"
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[DB] add_to_escalation (Postgres) failed: {exc}")
        # Fallback to local SQLite so the query isn't lost
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO escalated_queries (query, context) VALUES (?, ?)", (query, f"[PG-FAIL] {context}"))
            conn.commit()
            conn.close()
        except Exception:
            pass


def _build_transcript_snapshot(cur, session_id: str | None, current_query: str) -> str:
    if not session_id:
        return f"user: {current_query}".strip()
    try:
        cur.execute(
            """
            SELECT role, message
            FROM conversation_history
            WHERE session_id = %s
            ORDER BY timestamp ASC, id ASC;
            """,
            (session_id,),
        )
        rows = cur.fetchall() or []
    except Exception:
        rows = []

    lines = [
        f"{role}: {str(message).strip()}"
        for role, message in rows
        if str(message or "").strip()
    ]
    current_line = f"user: {current_query}".strip()
    if current_query and current_line not in lines:
        lines.append(current_line)
    return "\n".join(lines)[-12000:]


def _get_session_recording_path(cur, session_id: str | None) -> str | None:
    if not session_id:
        return None
    try:
        cur.execute(
            "SELECT audio_file_path FROM call_sessions WHERE session_id = %s LIMIT 1;",
            (session_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None

def log_conversation(phone_number: str, session_id: str, role: str, message: str):
    # Prefer Postgres so dashboard can read unified history.
    if _pg_enabled():
        try:
            import psycopg
            with psycopg.connect(POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversation_history (phone_number, session_id, role, message, timestamp)
                        VALUES (%s, %s, %s, %s, NOW());
                        """,
                        (phone_number, session_id, role, message),
                    )
                conn.commit()
            return
        except Exception as exc:
            print(f"[DB] log_conversation (Postgres) failed: {exc} — falling back to SQLite")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_history (phone_number, session_id, role, message) VALUES (?, ?, ?, ?)",
        (phone_number, session_id, role, message),
    )
    conn.commit()
    conn.close()

def get_conversation_history(session_id: str, limit: int = 5):
    if _pg_enabled():
        try:
            import psycopg
            with psycopg.connect(POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, message
                        FROM conversation_history
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s;
                        """,
                        (session_id, limit),
                    )
                    rows = cur.fetchall() or []
                    return list(reversed(rows))
        except Exception as exc:
            print(f"[DB] get_conversation_history (Postgres) failed: {exc} — falling back to SQLite")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, message FROM conversation_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit),
    )
    history = c.fetchall()
    conn.close()
    return list(reversed(history))


def get_recent_conversation_by_phone(
    phone_number: str,
    *,
    limit: int = 4,
    exclude_session_id: str | None = None,
) -> list[tuple[str, str]]:
    """Recent turns for this phone (any session) — used for short voice follow-ups."""
    phone = (phone_number or "").strip()
    if not phone:
        return []
    if _pg_enabled():
        try:
            import psycopg
            with psycopg.connect(POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    if exclude_session_id:
                        cur.execute(
                            """
                            SELECT role, message
                            FROM conversation_history
                            WHERE phone_number = %s AND session_id <> %s
                            ORDER BY timestamp DESC
                            LIMIT %s;
                            """,
                            (phone, exclude_session_id, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT role, message
                            FROM conversation_history
                            WHERE phone_number = %s
                            ORDER BY timestamp DESC
                            LIMIT %s;
                            """,
                            (phone, limit),
                        )
                    rows = cur.fetchall() or []
                    return list(reversed(rows))
        except Exception as exc:
            print(f"[DB] get_recent_conversation_by_phone (Postgres) failed: {exc}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if exclude_session_id:
        c.execute(
            """
            SELECT role, message FROM conversation_history
            WHERE phone_number = ? AND session_id <> ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (phone, exclude_session_id, limit),
        )
    else:
        c.execute(
            """
            SELECT role, message FROM conversation_history
            WHERE phone_number = ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (phone, limit),
        )
    history = c.fetchall()
    conn.close()
    return list(reversed(history))


def get_farmer_memory_context(
    phone_number: str,
    *,
    exclude_session_id: str | None = None,
    limit: int = 6,
) -> str:
    """
    Cross-call personalization memory for FR16/FR17.

    This intentionally summarizes only recent non-sensitive interaction signals:
    intents, response types, extracted entities, and recent user questions.
    """
    if not _pg_enabled():
        return ""
    keys = _phone_lookup_keys(phone_number)
    if not keys:
        return ""
    try:
        import psycopg

        lim = max(1, min(int(limit or 6), 12))
        interactions: list[str] = []
        questions: list[str] = []
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                if exclude_session_id:
                    cur.execute(
                        """
                        SELECT intent, response_type, entities, confidence
                        FROM interaction_records
                        WHERE phone_number = ANY(%s)
                          AND session_id <> %s
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        (keys, exclude_session_id, lim),
                    )
                else:
                    cur.execute(
                        """
                        SELECT intent, response_type, entities, confidence
                        FROM interaction_records
                        WHERE phone_number = ANY(%s)
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        (keys, lim),
                    )
                for intent, response_type, entities, confidence in (cur.fetchall() or []):
                    parts = []
                    if intent:
                        parts.append(f"intent={intent}")
                    if response_type:
                        parts.append(f"response={response_type}")
                    if entities:
                        parts.append(f"entities={entities}")
                    if confidence is not None:
                        try:
                            parts.append(f"conf={float(confidence):.2f}")
                        except (TypeError, ValueError):
                            pass
                    if parts:
                        interactions.append("; ".join(parts))

                if exclude_session_id:
                    cur.execute(
                        """
                        SELECT message
                        FROM conversation_history
                        WHERE phone_number = ANY(%s)
                          AND role = 'user'
                          AND session_id <> %s
                        ORDER BY timestamp DESC
                        LIMIT %s;
                        """,
                        (keys, exclude_session_id, min(lim, 4)),
                    )
                else:
                    cur.execute(
                        """
                        SELECT message
                        FROM conversation_history
                        WHERE phone_number = ANY(%s)
                          AND role = 'user'
                        ORDER BY timestamp DESC
                        LIMIT %s;
                        """,
                        (keys, min(lim, 4)),
                    )
                questions = [
                    str(row[0]).strip()
                    for row in (cur.fetchall() or [])
                    if row and str(row[0]).strip()
                ]
        lines: list[str] = []
        if interactions:
            lines.append("የቀድሞ ጥያቄ/ምላሽ ማጠቃለያ፦ " + " | ".join(interactions[:lim]))
        if questions:
            lines.append("ቀደም ሲል የጠየቁት፦ " + " | ".join(q[:160] for q in questions))
        if not lines:
            return ""
        return "የተጠቃሚ ታሪክ (ለቀጣይ ጥያቄዎች አውድ ብቻ)\n" + "\n".join(lines) + "\n\n"
    except Exception as exc:
        print(f"[DB] get_farmer_memory_context failed: {exc}")
        return ""


def log_interaction_record(
    phone_number: str,
    session_id: str,
    intent: Optional[str],
    response_type: str,
    entities: Optional[dict[str, Any]] = None,
    confidence: Optional[float] = None,
):
    """
    Structured interaction record for FR16 traceability:
    - intent/entities/confidence (best-effort)
    - response_type: market_price | rag_answer | escalated | slot_filling | fallback | etc.
    """
    if not _pg_enabled():
        return
    try:
        import psycopg
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interaction_records
                      (phone_number, session_id, intent, response_type, entities, confidence, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, NOW());
                    """,
                    (
                        phone_number,
                        session_id,
                        intent,
                        response_type,
                        json.dumps(entities) if entities else None,
                        confidence,
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[DB] log_interaction_record failed: {exc}")

    _learn_farmer_memory_from_interaction(phone_number, intent, entities)


def _learn_farmer_memory_from_interaction(
    phone_number: str,
    intent: Optional[str],
    entities: Optional[dict[str, Any]],
) -> None:
    """Best-effort personalization memory learned from each farmer interaction."""
    if not _pg_enabled():
        return
    p = (phone_number or "").strip()
    if not p or p == "Unknown":
        return
    data = entities or {}
    crop = (
        data.get("crop_en")
        or data.get("crop")
        or data.get("crop_type")
        or data.get("commodity")
    )
    location = (
        data.get("location")
        or data.get("location_en")
        or data.get("location_keyword")
        or data.get("region")
        or data.get("region_en")
        or data.get("region_keyword")
    )
    language = data.get("language") or data.get("preferred_language")
    name = data.get("farmer_name") or data.get("name") or data.get("full_name")
    farm_size = data.get("farm_size_ha") or data.get("farm_size")
    learned: list[str] = []
    if intent:
        learned.append(f"last_intent={intent}")
    if crop:
        learned.append(f"crop={crop}")
    if location:
        learned.append(f"location={location}")
    if farm_size:
        learned.append(f"farm_size_ha={farm_size}")
    if name:
        learned.append(f"name={name}")
    if not any([crop, location, language, intent, name, farm_size]):
        return

    try:
        import psycopg

        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT crops, notes
                    FROM farmers_kb
                    WHERE phone_number = %s
                    LIMIT 1;
                    """,
                    (p,),
                )
                row = cur.fetchone()
                existing_crops: list[str] = []
                existing_notes = ""
                if row:
                    raw_crops, existing_notes = row
                    if isinstance(raw_crops, list):
                        existing_crops = [str(x) for x in raw_crops if x]
                    elif raw_crops:
                        try:
                            parsed = json.loads(raw_crops) if isinstance(raw_crops, str) else raw_crops
                            if isinstance(parsed, list):
                                existing_crops = [str(x) for x in parsed if x]
                        except Exception:
                            existing_crops = [str(raw_crops)]

                if crop and str(crop) not in existing_crops:
                    existing_crops.append(str(crop))
                note_line = "; ".join(learned)
                notes = (existing_notes or "").strip()
                if note_line and note_line not in notes:
                    notes = (notes + "\n" + note_line).strip()
                    notes = notes[-2000:]

                cur.execute(
                    """
                    INSERT INTO farmers_kb
                      (phone_number, name, location, preferred_language, crops, farm_size, notes, registered_at, updated_at)
                    VALUES (%s, %s, %s, COALESCE(%s, 'am'), %s::jsonb, %s, %s, NOW(), NOW())
                    ON CONFLICT (phone_number) DO UPDATE SET
                      name = COALESCE(EXCLUDED.name, farmers_kb.name),
                      location = COALESCE(EXCLUDED.location, farmers_kb.location),
                      preferred_language = COALESCE(EXCLUDED.preferred_language, farmers_kb.preferred_language),
                      crops = COALESCE(EXCLUDED.crops, farmers_kb.crops),
                      farm_size = COALESCE(EXCLUDED.farm_size, farmers_kb.farm_size),
                      notes = COALESCE(EXCLUDED.notes, farmers_kb.notes),
                      updated_at = NOW();
                    """,
                    (
                        p,
                        str(name) if name else None,
                        str(location) if location else None,
                        str(language) if language else None,
                        json.dumps(existing_crops, ensure_ascii=False) if existing_crops else None,
                        float(farm_size) if farm_size else None,
                        notes or None,
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[DB] farmer memory update failed: {exc}")

def get_market_price(crop_name: str, region: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if region:
        c.execute(
            "SELECT price, unit, updated_at FROM market_prices WHERE crop_name = ? AND region = ? ORDER BY updated_at DESC LIMIT 1",
            (crop_name, region)
        )
    else:
        c.execute(
            "SELECT price, unit, updated_at FROM market_prices WHERE crop_name = ? ORDER BY updated_at DESC LIMIT 1",
            (crop_name,)
        )
    result = c.fetchone()
    conn.close()
    return result  # (price, unit, updated_at) or None

def get_dynamic_knowledge(query: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT content FROM dynamic_knowledge_cache WHERE query = ?", (query.lower().strip(),))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def set_dynamic_knowledge(query: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO dynamic_knowledge_cache (query, content) VALUES (?, ?) ON CONFLICT(query) DO UPDATE SET content=excluded.content, updated_at=CURRENT_TIMESTAMP",
                  (query.lower().strip(), content))
        conn.commit()
    except Exception as e:
        print(f"Error setting dynamic knowledge: {e}")
    finally:
        conn.close()

def register_farmer(phone_number: str, name: str, location: str, preferred_language: str = 'am'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # SQLite UPSERT equivalent (since sqlite 3.24)
        c.execute("INSERT INTO farmers (phone_number, name, location, preferred_language) VALUES (?, ?, ?, ?) ON CONFLICT(phone_number) DO UPDATE SET name=excluded.name, location=excluded.location, preferred_language=excluded.preferred_language", 
                  (phone_number, name, location, preferred_language))
        conn.commit()
    except Exception as e:
        print(f"Error registering farmer: {e}")
    finally:
        conn.close()

def _phone_lookup_keys(phone_number: str) -> list[str]:
    p = (phone_number or "").strip()
    if not p:
        return []
    keys: list[str] = [p]
    if p.startswith("+"):
        tail = p[1:].strip()
        if tail and tail not in keys:
            keys.append(tail)
    else:
        plus = f"+{p}"
        if plus not in keys:
            keys.append(plus)
    digits = "".join(ch for ch in p if ch.isdigit())
    if len(digits) >= 8 and digits not in keys:
        keys.append(digits)
    return list(dict.fromkeys(keys))


def get_farmer_profile(phone_number: str):
    p = (phone_number or "").strip()
    if not p or p == "Unknown":
        return None
    keys = _phone_lookup_keys(p)
    if not keys:
        return None

    if _pg_enabled():
        try:
            import psycopg

            with psycopg.connect(POSTGRES_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.full_name, fp.location, fp.primary_language, fp.farm_size
                        FROM callers c
                        LEFT JOIN farmer_profiles fp ON fp.caller_id = c.caller_id
                        WHERE c.phone_number = ANY(%s)
                        LIMIT 1;
                        """,
                        (keys,),
                    )
                    row = cur.fetchone()
                    profile: dict | None = None
                    if row:
                        profile = {
                            "phone_number": p,
                            "full_name": row[0],
                            "name": row[0],
                            "location": row[1],
                            "primary_language": row[2],
                            "preferred_language": row[2] or "am",
                            "farm_size": row[3],
                        }
                    try:
                        cur.execute(
                            """
                            SELECT name, location, preferred_language, crops, farm_size, notes
                            FROM farmers_kb
                            WHERE phone_number = ANY(%s)
                            LIMIT 1;
                            """,
                            (keys,),
                        )
                        kb = cur.fetchone()
                    except Exception:
                        kb = None
                    if kb:
                        kb_name, kb_loc, kb_lang, kb_crops, kb_fs, kb_notes = kb
                        if profile is None:
                            profile = {"phone_number": p}
                        if kb_name and not profile.get("name"):
                            profile["name"] = kb_name
                        if kb_loc and not profile.get("location"):
                            profile["location"] = kb_loc
                        if kb_lang:
                            profile["preferred_language"] = kb_lang
                            profile["primary_language"] = kb_lang
                        if kb_crops is not None:
                            profile["crops"] = kb_crops
                        if kb_fs is not None and profile.get("farm_size") is None:
                            profile["farm_size"] = kb_fs
                        if kb_notes:
                            profile["notes"] = kb_notes
                    if profile:
                        return profile
        except Exception as exc:
            print(f"[DB] get_farmer_profile (Postgres) failed: {exc}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT name, location, preferred_language, registered_at FROM farmers WHERE phone_number = ?",
        (p,),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "phone_number": p,
            "name": row[0],
            "location": row[1],
            "preferred_language": row[2],
            "registered_at": row[3],
        }
    return None


def consume_answered_expert_response(phone_number: str) -> dict[str, Any] | None:
    """
    Return the oldest answered escalation for this farmer and mark it closed.

    Legacy on-call delivery uses this when explicitly enabled. The default
    workflow now plays recorded expert answers through outbound callbacks.
    """
    if not _pg_enabled():
        return None
    keys = _phone_lookup_keys(phone_number)
    if not keys:
        return None
    try:
        import psycopg

        with psycopg.connect(POSTGRES_URL, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, query, expert_response, expert_audio_path, expert_notes
                    FROM escalations
                    WHERE phone_number = ANY(%s)
                      AND status = 'answered'
                      AND (expert_response IS NOT NULL OR expert_audio_path IS NOT NULL)
                    ORDER BY answered_at NULLS LAST, created_at ASC
                    LIMIT 1
                    FOR UPDATE;
                    """,
                    (keys,),
                )
                row = cur.fetchone()
                if not row:
                    conn.commit()
                    return None
                ticket_id, query, expert_response, expert_audio_path, expert_notes = row
                cur.execute(
                    """
                    UPDATE escalations
                    SET status = 'closed', closed_at = NOW(), updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (ticket_id,),
                )
            conn.commit()
        return {
            "ticket_id": ticket_id,
            "query": query,
            "text": expert_response or "",
            "audio_path": expert_audio_path or "",
            "expert_notes": expert_notes or "",
        }
    except Exception as exc:
        print(f"[DB] consume_answered_expert_response failed: {exc}")
        return None

def create_alert(target_region: str, alert_message: str, severity: str = "warning"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (target_region, alert_message, severity) VALUES (?, ?, ?)", 
              (target_region, alert_message, severity))
    conn.commit()
    conn.close()

def get_alerts_for_region(region: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT alert_message, severity FROM alerts WHERE target_region = ? OR target_region = 'all' ORDER BY created_at DESC", (region,))
    alerts = c.fetchall()
    conn.close()
    return alerts

def set_session_state(session_id: str, current_state: str, pending_action: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO session_states (session_id, current_state, pending_action) VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET current_state=excluded.current_state, pending_action=excluded.pending_action, updated_at=CURRENT_TIMESTAMP", 
                  (session_id, current_state, pending_action))
        conn.commit()
    except Exception as e:
        print(f"Error setting session state: {e}")
    finally:
        conn.close()

def get_session_state(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT current_state, pending_action FROM session_states WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"current_state": row[0], "pending_action": row[1]}
    return None

def insert_call_record(session_id: str, phone_number: str, recording_path: str, duration: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO call_records (session_id, phone_number, recording_path, duration) VALUES (?, ?, ?, ?)",
                  (session_id, phone_number, recording_path, duration))
        conn.commit()
    except Exception as e:
        print(f"Error inserting call record: {e}")
    finally:
        conn.close()

init_kb()
init_db()
init_pg_app_tables()
