import uuid
import enum

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Float,
    Integer,
    ForeignKey,
    Enum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class ASRStatus(str, enum.Enum):
    started = "started"
    completed = "completed"
    failed = "failed"


class CallUtterance(Base):
    __tablename__ = "call_utterances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to existing call table
    call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Link to existing call_sessions table
    call_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("call_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Runtime session_id used by VAD/phone gateway
    session_id = Column(String(120), nullable=False, index=True)

    # Utterance/audio metadata
    utterance_index = Column(Integer, nullable=True)
    utterance_path = Column(Text, nullable=True)
    audio_id = Column(String(255), nullable=True, index=True)

    duration_seconds = Column(Float, nullable=True)
    speech_probability = Column(Float, nullable=True)

    # ASR result
    transcript = Column(Text, nullable=True)
    language = Column(String(20), nullable=False, default="am")
    asr_confidence = Column(Float, nullable=True)
    asr_engine = Column(String(100), nullable=True)
    asr_model_version = Column(String(150), nullable=True)

    # ASR processing state
    asr_status = Column(
        Enum(ASRStatus, name="asr_status"),
        nullable=False,
        default=ASRStatus.started,
    )

    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Optional relationships
    call = relationship("Call", back_populates="utterances")
    call_session = relationship("CallSession", back_populates="utterances")