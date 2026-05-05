"""
Logic-service data layer.

Postgres-backed via SQLAlchemy (db.py) and Chroma for vector search.
Public CRUD helpers preserve the original signatures used by main.py so the
RAG pipeline keeps working.
"""
import json
import os
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import desc, func, select

from db import Base, SessionLocal, engine
from models import (
    Alert,
    CallRecord,
    ConversationMessage,
    DashboardUser,
    Escalation,
    FarmerKB,
    MarketPrice,
    SessionState,
)
from auth import hash_password


# ── Chroma (RAG vector DB) ────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/data")
chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
collection = chroma_client.get_or_create_collection(
    name="agronomy_kb",
    embedding_function=sentence_transformer_ef,
)


def init_kb():
    """Seed the Chroma collection from mock_kb.json on first boot."""
    if collection.count() == 0:
        if os.path.exists("mock_kb.json"):
            with open("mock_kb.json", "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            documents, metadatas, ids = [], [], []
            for i, (intent, response) in enumerate(kb_data.items()):
                if intent == "unknown":
                    continue
                documents.append(response)
                metadatas.append({"intent": intent})
                ids.append(str(i))
            if documents:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                print("[KB] Knowledge Base initialised from mock_kb.json")


# ── Postgres schema bootstrap ────────────────────────────────────────────────
def init_db():
    """Create all tables defined on the SQLAlchemy Base."""
    Base.metadata.create_all(bind=engine)


def seed_default_admin():
    """
    Insert a default dashboard admin user if dashboard_users is empty.

    Honours the same env vars used by phone_gateway/bootstrap.py:
      DEFAULT_ADMIN_EMAIL    (default: admin@example.com)
      DEFAULT_ADMIN_PASSWORD (default: admin123)
      DEFAULT_ADMIN_NAME     (default: System Admin)
      CREATE_DEFAULT_ADMIN   (default: true)
      RESET_DEFAULT_ADMIN_PASSWORD (default: false)
    """
    if os.getenv("CREATE_DEFAULT_ADMIN", "true").lower() not in ("true", "1", "yes"):
        return

    email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    name = os.getenv("DEFAULT_ADMIN_NAME", "System Admin")
    reset = os.getenv("RESET_DEFAULT_ADMIN_PASSWORD", "false").lower() in ("true", "1", "yes")

    db = SessionLocal()
    try:
        existing = db.query(DashboardUser).filter(DashboardUser.email == email).first()
        if existing:
            if reset:
                existing.password_hash = hash_password(password)
                existing.role = "admin"
                existing.is_active = True
                db.commit()
                print(f"[BOOTSTRAP] Reset password for default admin {email}.")
            return
        admin = DashboardUser(
            full_name=name,
            email=email,
            password_hash=hash_password(password),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(f"[BOOTSTRAP] Default admin created: {email}")
    except Exception as exc:
        db.rollback()
        print(f"[BOOTSTRAP] Default admin seeding failed: {exc}")
    finally:
        db.close()


# ── Escalations ──────────────────────────────────────────────────────────────
def add_to_escalation(
    query: str,
    context: str,
    phone_number: Optional[str] = None,
    session_id: Optional[str] = None,
):
    db = SessionLocal()
    try:
        esc = Escalation(
            query=query,
            context=context,
            phone_number=phone_number,
            session_id=session_id,
            status="pending",
        )
        db.add(esc)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] add_to_escalation failed: {exc}")
    finally:
        db.close()


# ── Conversation history ─────────────────────────────────────────────────────
def log_conversation(phone_number: str, session_id: str, role: str, message: str):
    db = SessionLocal()
    try:
        db.add(
            ConversationMessage(
                phone_number=phone_number,
                session_id=session_id,
                role=role,
                message=message,
            )
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] log_conversation failed: {exc}")
    finally:
        db.close()


def get_conversation_history(session_id: str, limit: int = 5):
    """Returns a list of (role, message) tuples in chronological order."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ConversationMessage.role, ConversationMessage.message)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(desc(ConversationMessage.timestamp))
            .limit(limit)
            .all()
        )
        return list(reversed([(r[0], r[1]) for r in rows]))
    finally:
        db.close()


# ── Market prices ────────────────────────────────────────────────────────────
def get_market_price(crop_name: str, region: Optional[str] = None):
    """Returns (price, unit, updated_at) for the latest matching row."""
    db = SessionLocal()
    try:
        query = db.query(MarketPrice).filter(MarketPrice.crop_name == crop_name)
        if region:
            query = query.filter(MarketPrice.region == region)
        row = query.order_by(desc(MarketPrice.updated_at)).first()
        if not row:
            return None
        return (row.price, row.unit, row.updated_at)
    finally:
        db.close()


# ── Farmer profiles (RAG-side) ───────────────────────────────────────────────
def register_farmer(
    phone_number: str,
    name: str,
    location: str,
    preferred_language: str = "am",
):
    """Upsert a lightweight farmer profile keyed by phone number."""
    db = SessionLocal()
    try:
        existing = (
            db.query(FarmerKB).filter(FarmerKB.phone_number == phone_number).first()
        )
        if existing:
            existing.name = name
            existing.location = location
            existing.preferred_language = preferred_language
            existing.updated_at = datetime.utcnow()
        else:
            db.add(
                FarmerKB(
                    phone_number=phone_number,
                    name=name,
                    location=location,
                    preferred_language=preferred_language,
                )
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] register_farmer failed: {exc}")
    finally:
        db.close()


def get_farmer_profile(phone_number: str):
    db = SessionLocal()
    try:
        row = (
            db.query(FarmerKB).filter(FarmerKB.phone_number == phone_number).first()
        )
        if not row:
            return None
        return {
            "phone_number": row.phone_number,
            "name": row.name,
            "location": row.location,
            "preferred_language": row.preferred_language,
            "registered_at": row.registered_at.isoformat() if row.registered_at else None,
        }
    finally:
        db.close()


# ── Alerts ───────────────────────────────────────────────────────────────────
def create_alert(target_region: str, alert_message: str, severity: str = "warning"):
    db = SessionLocal()
    try:
        db.add(
            Alert(
                target_region=target_region,
                alert_message=alert_message,
                severity=severity,
            )
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] create_alert failed: {exc}")
    finally:
        db.close()


def get_alerts_for_region(region: str):
    db = SessionLocal()
    try:
        rows = (
            db.query(Alert.alert_message, Alert.severity)
            .filter((Alert.target_region == region) | (Alert.target_region == "all"))
            .order_by(desc(Alert.created_at))
            .all()
        )
        return [(r[0], r[1]) for r in rows]
    finally:
        db.close()


# ── Session state (multi-turn / safety confirmations) ────────────────────────
def set_session_state(
    session_id: str,
    current_state: str,
    pending_action: Optional[str] = None,
):
    db = SessionLocal()
    try:
        existing = (
            db.query(SessionState).filter(SessionState.session_id == session_id).first()
        )
        if existing:
            existing.current_state = current_state
            existing.pending_action = pending_action
            existing.updated_at = datetime.utcnow()
        else:
            db.add(
                SessionState(
                    session_id=session_id,
                    current_state=current_state,
                    pending_action=pending_action,
                )
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] set_session_state failed: {exc}")
    finally:
        db.close()


def get_session_state(session_id: str):
    db = SessionLocal()
    try:
        row = (
            db.query(SessionState).filter(SessionState.session_id == session_id).first()
        )
        if not row:
            return None
        return {
            "current_state": row.current_state,
            "pending_action": row.pending_action,
        }
    finally:
        db.close()


# ── Call records ─────────────────────────────────────────────────────────────
def insert_call_record(
    session_id: str,
    phone_number: str,
    recording_path: str,
    duration: int,
):
    db = SessionLocal()
    try:
        existing = (
            db.query(CallRecord).filter(CallRecord.session_id == session_id).first()
        )
        if existing:
            existing.phone_number = phone_number
            existing.recording_path = recording_path
            existing.duration = duration
        else:
            db.add(
                CallRecord(
                    session_id=session_id,
                    phone_number=phone_number,
                    recording_path=recording_path,
                    duration=duration,
                )
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[DB] insert_call_record failed: {exc}")
    finally:
        db.close()


# ── Module init (called once on FastAPI startup from main.py) ────────────────
init_kb()
