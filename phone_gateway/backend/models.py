from sqlalchemy import (
    Column,
    String,
    DateTime,
    Float,
    ForeignKey,
    Table,
    Integer,
    Boolean,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from backend.database import Base


def generate_uuid():
    return str(uuid.uuid4())


# ============================================================
# Admin Dashboard Authentication Tables
# ============================================================

class DashboardUser(Base):
    __tablename__ = "dashboard_users"

    user_id = Column(
        String,
        primary_key=True,
        default=generate_uuid,
        index=True,
    )

    full_name = Column(String, nullable=False)

    email = Column(
        String,
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash = Column(String, nullable=False)

    role = Column(String, nullable=False, default="da")
    # allowed roles:
    #   admin
    #   da
    #   expert

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)


# ============================================================
# Profile Service Tables
# ============================================================

farmer_crop_association = Table(
    "farmer_crop",
    Base.metadata,
    Column("farmer_id", String, ForeignKey("farmer_profiles.id")),
    Column("crop_id", String, ForeignKey("crops.id")),
)


class Caller(Base):
    __tablename__ = "callers"

    caller_id = Column(String, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)

    calls = relationship("CallSession", back_populates="caller")
    profile = relationship(
        "FarmerProfile",
        back_populates="caller",
        uselist=False,
    )
    preferences = relationship("Preference", back_populates="caller")


class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"

    id = Column(String, primary_key=True, index=True)
    caller_id = Column(
        String,
        ForeignKey("callers.caller_id"),
        unique=True,
    )

    location = Column(String, nullable=True)
    primary_language = Column(String, default="am")
    farm_size = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    caller = relationship("Caller", back_populates="profile")
    crops = relationship(
        "Crop",
        secondary=farmer_crop_association,
        back_populates="farmers",
    )


class Crop(Base):
    __tablename__ = "crops"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    season = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    farmers = relationship(
        "FarmerProfile",
        secondary=farmer_crop_association,
        back_populates="crops",
    )


class Preference(Base):
    __tablename__ = "preferences"

    id = Column(String, primary_key=True, index=True)
    caller_id = Column(String, ForeignKey("callers.caller_id"))

    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    caller = relationship("Caller", back_populates="preferences")


# ============================================================
# Telephony Service Tables
# ============================================================

class CallSession(Base):
    __tablename__ = "call_sessions"

    session_id = Column(String, primary_key=True, index=True)
    caller_id = Column(
        String,
        ForeignKey("callers.caller_id"),
        nullable=True,
    )

    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    audio_file_path = Column(String, nullable=True)
    status = Column(String, default="active")

    caller = relationship("Caller", back_populates="calls")
    audio_files = relationship("AudioFile", back_populates="session")
    dtmf_events = relationship("DTMFEvent", back_populates="session")
    asr_jobs = relationship("ASRJob", back_populates="session")
    tts_jobs = relationship("TTSJob", back_populates="session")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(
        String,
        ForeignKey("call_sessions.session_id"),
    )

    file_path = Column(String, nullable=False)
    type = Column(String, nullable=False)
    # examples:
    #   caller
    #   mixed
    #   agent
    #   utterance
    #   tts_output

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("CallSession", back_populates="audio_files")


class DTMFEvent(Base):
    __tablename__ = "dtmf_events"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(
        String,
        ForeignKey("call_sessions.session_id"),
    )

    key = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("CallSession", back_populates="dtmf_events")


# ============================================================
# ASR Service Tables
# ============================================================

class ASRJob(Base):
    __tablename__ = "asr_jobs"

    id = Column(String, primary_key=True, index=True)

    session_id = Column(
        String,
        ForeignKey("call_sessions.session_id"),
    )

    audio_file_id = Column(
        String,
        ForeignKey("audio_files.id"),
        nullable=True,
    )

    status = Column(String, default="pending")
    engine = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    session = relationship("CallSession", back_populates="asr_jobs")
    transcript = relationship(
        "Transcript",
        back_populates="job",
        uselist=False,
    )


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(String, primary_key=True, index=True)

    asr_job_id = Column(
        String,
        ForeignKey("asr_jobs.id"),
        unique=True,
    )

    text = Column(String, nullable=False)
    language = Column(String, default="am")
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("ASRJob", back_populates="transcript")
    segments = relationship("ASRSegment", back_populates="transcript")


class ASRSegment(Base):
    __tablename__ = "asr_segments"

    id = Column(String, primary_key=True, index=True)

    transcript_id = Column(
        String,
        ForeignKey("transcripts.id"),
    )

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)

    transcript = relationship("Transcript", back_populates="segments")


# ============================================================
# TTS Service Tables
# ============================================================

class TTSJob(Base):
    __tablename__ = "tts_jobs"

    id = Column(String, primary_key=True, index=True)

    session_id = Column(
        String,
        ForeignKey("call_sessions.session_id"),
    )

    text_input = Column(String, nullable=False)
    language = Column(String, default="am")
    status = Column(String, default="pending")

    audio_file_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    session = relationship("CallSession", back_populates="tts_jobs")