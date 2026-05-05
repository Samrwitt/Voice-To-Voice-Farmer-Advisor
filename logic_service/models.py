"""
SQLAlchemy models for logic_service.

Notes:
- DashboardUser shares the same `dashboard_users` table that phone_gateway
  declares. We keep the column definitions identical so both services can
  read/write the same rows.
- The other tables are owned by logic_service and use unique names to avoid
  colliding with phone_gateway's richer telephony schema.
"""
from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────────────
# Shared with phone_gateway: dashboard_users
# ──────────────────────────────────────────────────────────────────────────────
class DashboardUser(Base):
    __tablename__ = "dashboard_users"

    user_id = Column(String, primary_key=True, default=generate_uuid, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="da")
    # allowed roles: admin | da | expert
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logic service domain
# ──────────────────────────────────────────────────────────────────────────────
class FarmerKB(Base):
    """
    Lightweight farmer profile written by the RAG flow when a call comes in.
    Distinct from phone_gateway's farmer_profiles which is keyed by caller_id.
    """

    __tablename__ = "farmers_kb"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    preferred_language = Column(String, default="am")
    crops = Column(JSON, nullable=True)
    farm_size = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Read-only mirrors of phone_gateway tables (to show caller data in dashboard)
# ──────────────────────────────────────────────────────────────────────────────
class Caller(Base):
    __tablename__ = "callers"

    caller_id = Column(String, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class FarmerProfilePG(Base):
    __tablename__ = "farmer_profiles"

    id = Column(String, primary_key=True, index=True)
    caller_id = Column(String, ForeignKey("callers.caller_id"), unique=True)
    location = Column(String, nullable=True)
    primary_language = Column(String, default="am")
    farm_size = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    phone_number = Column(String, nullable=False, index=True)
    recording_path = Column(String, nullable=True)
    duration = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ConversationMessage(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String, index=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # user | assistant
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    phone_number = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)

    status = Column(String, nullable=False, default="pending", index=True)
    # pending | assigned | answered | closed

    assigned_to_user_id = Column(
        String,
        ForeignKey("dashboard_users.user_id"),
        nullable=True,
        index=True,
    )
    assigned_at = Column(DateTime, nullable=True)

    expert_response = Column(Text, nullable=True)
    answered_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    assignee = relationship("DashboardUser", foreign_keys=[assigned_to_user_id])


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_region = Column(String, nullable=False, index=True)
    alert_message = Column(Text, nullable=False)
    severity = Column(String, default="warning")
    category = Column(String, nullable=True)  # weather | pest | disease | market | other
    scheduled_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_by = Column(String, ForeignKey("dashboard_users.user_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class MarketPrice(Base):
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    crop_name = Column(String, nullable=False, index=True)
    region = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)


class SessionState(Base):
    __tablename__ = "session_states"

    session_id = Column(String, primary_key=True, index=True)
    current_state = Column(String, nullable=False)
    pending_action = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)

    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)

    topic = Column(String, nullable=True, index=True)
    crop = Column(String, nullable=True, index=True)
    region = Column(String, nullable=True, index=True)
    category = Column(String, nullable=True, index=True)
    # agronomy | market | disease | pest | weather | other

    status = Column(String, default="uploaded", index=True)
    # uploaded | approved | rejected

    uploaded_by = Column(String, ForeignKey("dashboard_users.user_id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    approved_by = Column(String, ForeignKey("dashboard_users.user_id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)

    indexing_status = Column(String, default="pending", index=True)
    # pending | indexing | indexed | failed
    indexing_error = Column(Text, nullable=True)
    chroma_doc_count = Column(Integer, default=0)
    last_indexed_at = Column(DateTime, nullable=True)

    chunks = relationship(
        "KBDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class KBDocumentChunk(Base):
    __tablename__ = "kb_document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        String,
        ForeignKey("kb_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chroma_id = Column(String, nullable=False, unique=True, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    status = Column(String, default="indexed")
    indexed_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("KBDocument", back_populates="chunks")


class ServiceError(Base):
    __tablename__ = "service_errors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String, nullable=False, index=True)
    endpoint = Column(String, nullable=True)
    method = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    error = Column(Text, nullable=False)
    request_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
