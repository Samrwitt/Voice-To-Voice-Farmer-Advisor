"""
Admin REST API for the dashboard frontend.

All endpoints live under /admin and are protected by JWT bearer tokens issued
by `auth.py`. Roles enforced:
    admin  - full access
    da     - Development Agent (read farmers/calls, manage escalations and field reports)
    expert - Agricultural Expert (work assigned escalations, review/approve KB docs)
"""
import csv
import io
import os
import subprocess
import time
import uuid
import wave
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import case, desc, func, text
from sqlalchemy.orm import Session

from alert_utils import normalize_region as _normalize_region
from auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_roles,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from database import collection, ensure_runtime_schema
from db import SessionLocal, get_db
from kb_indexing import index_document, remove_document_from_chroma
from s3_client import is_enabled as s3_enabled, presign_get_url
from models import (
    Alert,
    AlertCallNotification,
    CallRecord,
    CallSessionPG,
    ConversationMessage,
    Caller,
    DashboardUser,
    Escalation,
    FarmerKB,
    FarmerProfilePG,
    KBDocument,
    MarketPrice,
    ServiceError,
)


router = APIRouter(prefix="/admin", tags=["admin"])

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000").rstrip("/")


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None  # legacy alias
    password: str


class CreateUserRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str  # admin | da | expert
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class MarketPriceRequest(BaseModel):
    crop_name: str
    region: str
    price: float
    unit: str


class AlertRequest(BaseModel):
    target_region: str
    alert_message: str
    severity: str = "warning"
    category: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    notify_by_call: bool = True


class KBRequest(BaseModel):
    intent: str
    response: str


class AssignEscalationRequest(BaseModel):
    user_id: str


class EscalationResponseRequest(BaseModel):
    answer: str
    expert_notes: Optional[str] = None


class KBDocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    topic: Optional[str] = None
    crop: Optional[str] = None
    region: Optional[str] = None
    category: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Serialisers
# ──────────────────────────────────────────────────────────────────────────────
def _user_dict(u: DashboardUser) -> dict:
    return {
        "user_id": u.user_id,
        "full_name": u.full_name,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


def _isoformat(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _local_call_recording_candidates(session_id: str, stored_path: Optional[str] = None) -> list[str]:
    candidates: list[str] = []
    recordings_dir = os.getenv("CALL_RECORDINGS_DIR", "/app/recordings")
    if stored_path and not stored_path.startswith("s3://"):
        root, ext = os.path.splitext(stored_path)
        if ext.lower() == ".wav":
            candidates.append(stored_path)
        elif ext.lower() == ".pcm":
            candidates.append(f"{root}.wav")

    candidates.extend(
        [
            os.path.join(recordings_dir, f"{session_id}.wav"),
            os.path.join(recordings_dir, "audio", f"{session_id}.wav"),
        ]
    )
    if stored_path and not stored_path.startswith("s3://"):
        candidates.append(stored_path)
    candidates.extend(
        [
            os.path.join(recordings_dir, f"{session_id}.pcm"),
            os.path.join(recordings_dir, "audio", f"{session_id}.pcm"),
            os.path.join(recordings_dir, session_id),
        ]
    )
    return list(dict.fromkeys(candidates))


def _ensure_playable_wav(candidate: str) -> str:
    if not candidate.lower().endswith(".pcm"):
        return candidate
    wav_path = f"{os.path.splitext(candidate)[0]}.wav"
    if os.path.exists(wav_path):
        return wav_path
    try:
        with open(candidate, "rb") as f:
            pcm_bytes = f.read()
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(os.getenv("SIP_AUDIOSOCKET_SAMPLE_RATE", "8000") or "8000"))
            wf.writeframes(pcm_bytes)
        return wav_path
    except Exception as exc:
        print(f"[CALL AUDIO] PCM to WAV fallback failed for {candidate}: {exc}")
        return candidate


def _first_existing_call_recording(session_id: str, stored_path: Optional[str] = None) -> Optional[str]:
    for candidate in _local_call_recording_candidates(session_id, stored_path):
        if candidate and os.path.exists(candidate):
            return _ensure_playable_wav(candidate)
    return None


def _playable_call_recording_ref(session_id: str, stored_path: Optional[str] = None) -> Optional[str]:
    # Prefer the shared local Docker volume when available. It keeps playback
    # working even when MinIO is not reachable from the browser/proxy path.
    local_path = _first_existing_call_recording(session_id, stored_path)
    if local_path:
        return local_path
    if stored_path and stored_path.startswith("s3://") and s3_enabled():
        return stored_path
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email or req.username
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    user = db.query(DashboardUser).filter(DashboardUser.email == email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is disabled")

    user.last_login_at = datetime.utcnow()
    db.commit()

    token = create_access_token({
        "sub": user.user_id,
        "email": user.email,
        "role": user.role,
    })

    return {
        "token": token,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "role": user.role,
        "username": user.email,
        "email": user.email,
        "user_id": user.user_id,
        "full_name": user.full_name,
    }


@router.post("/logout")
def logout(user: DashboardUser = Depends(get_current_user)):
    # Stateless JWT: client just discards token. Kept for API compatibility.
    return {"status": "ok"}


@router.get("/me")
def me(user: DashboardUser = Depends(get_current_user)):
    return _user_dict(user)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard stats
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/stats")
def get_stats(
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Farmers: prefer callers table (system-of-record)
    total_farmers = db.query(func.count(Caller.caller_id)).scalar() or 0

    # Calls: use phone_gateway call_sessions table (system-of-record)
    total_calls = db.query(func.count(CallSessionPG.session_id)).scalar() or 0
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    calls_today = (
        db.query(func.count(CallSessionPG.session_id))
        .filter(CallSessionPG.start_time >= today_start)
        .scalar()
        or 0
    )
    pending_escalations = (
        db.query(func.count(Escalation.id))
        .filter(Escalation.status.in_(("pending", "assigned")))
        .scalar()
        or 0
    )
    total_alerts = db.query(func.count(Alert.id)).scalar() or 0

    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    rows = (
        db.query(
            func.date_trunc("day", CallSessionPG.start_time).label("day"),
            func.count(CallSessionPG.session_id),
        )
        .filter(CallSessionPG.start_time >= seven_days_ago)
        .group_by("day")
        .order_by("day")
        .all()
    )
    calls_per_day = [
        {"date": (r[0].date().isoformat() if r[0] else ""), "count": r[1]} for r in rows
    ]

    breakdown_rows = (
        db.query(Escalation.status, func.count(Escalation.id)).group_by(Escalation.status).all()
    )
    esc_breakdown = {r[0]: r[1] for r in breakdown_rows}

    try:
        kb_count = collection.count()
    except Exception:
        kb_count = 0

    return {
        "total_farmers": total_farmers,
        "calls_today": calls_today,
        "total_calls": total_calls,
        "pending_escalations": pending_escalations,
        "total_alerts": total_alerts,
        "calls_per_day": calls_per_day,
        "escalation_breakdown": esc_breakdown,
        "kb_count": kb_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Users (admin only)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/users")
def list_users(
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    rows = db.query(DashboardUser).order_by(desc(DashboardUser.created_at)).all()
    return [_user_dict(u) for u in rows]


@router.post("/users")
def create_user(
    req: CreateUserRequest,
    creator: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    if req.role not in ("admin", "da", "expert"):
        raise HTTPException(status_code=400, detail="Invalid role")

    if db.query(DashboardUser).filter(DashboardUser.email == req.email).first():
        raise HTTPException(status_code=400, detail="A user with this email already exists")

    new_user = DashboardUser(
        full_name=req.full_name,
        email=req.email,
        password_hash=hash_password(req.password),
        role=req.role,
        is_active=req.is_active,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return _user_dict(new_user)


@router.put("/users/{user_id}")
def update_user(
    user_id: str,
    req: UpdateUserRequest,
    actor: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.query(DashboardUser).filter(DashboardUser.user_id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if req.full_name is not None:
        target.full_name = req.full_name
    if req.role is not None:
        if req.role not in ("admin", "da", "expert"):
            raise HTTPException(status_code=400, detail="Invalid role")
        target.role = req.role
    if req.is_active is not None:
        if target.user_id == actor.user_id and not req.is_active:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
        target.is_active = req.is_active
    if req.password:
        target.password_hash = hash_password(req.password)

    target.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(target)
    return _user_dict(target)


@router.delete("/users/{user_id}")
def deactivate_user(
    user_id: str,
    actor: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    target = db.query(DashboardUser).filter(DashboardUser.user_id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.user_id == actor.user_id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
    target.is_active = False
    target.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Farmers
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/farmers")
def list_farmers(
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Admin and DA can see all farmers.
    # Experts can see all farmers (read-only) for context, 
    # but the docstring suggests DA is the primary reader.
    # We'll allow Experts to read farmers too as they need context for escalations.
    if user.role not in ("admin", "da", "expert"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Prefer phone_gateway callers as the primary list
    caller_rows = (
        db.query(Caller, FarmerProfilePG)
        .outerjoin(FarmerProfilePG, FarmerProfilePG.caller_id == Caller.caller_id)
        .order_by(desc(Caller.last_seen_at))
        .all()
    )

    # Map FarmerKB by phone number (optional enrichment).
    kb_rows = db.query(FarmerKB).all()
    kb_by_phone = {r.phone_number: r for r in kb_rows}

    results: list[dict] = []
    seen = set()

    for caller, profile in caller_rows:
        kb = kb_by_phone.get(caller.phone_number)
        results.append(
            {
                # keep numeric id optional; UI already tolerates missing
                "id": kb.id if kb else None,
                "phone_number": caller.phone_number,
                "name": (kb.name if kb and kb.name else caller.full_name),
                "location": (
                    (kb.location if kb and kb.location else None)
                    or (profile.location if profile else None)
                ),
                "language": (
                    (kb.preferred_language if kb and kb.preferred_language else None)
                    or (profile.primary_language if profile else None)
                    or "am"
                ),
                "registered_at": _isoformat(
                    (kb.registered_at if kb else None)
                    or (profile.created_at if profile else None)
                    or caller.created_at
                ),
            }
        )
        seen.add(caller.phone_number)

    # Include any FarmerKB rows that don't have a matching caller record
    for kb in kb_rows:
        if kb.phone_number in seen:
            continue
        results.append(
            {
                "id": kb.id,
                "phone_number": kb.phone_number,
                "name": kb.name,
                "location": kb.location,
                "language": kb.preferred_language,
                "registered_at": _isoformat(kb.registered_at),
            }
        )

    return results


@router.get("/farmers/{phone_number}")
def get_farmer(
    phone_number: str,
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.role not in ("admin", "da", "expert"):
        raise HTTPException(status_code=403, detail="Access denied")
    kb = db.query(FarmerKB).filter(FarmerKB.phone_number == phone_number).first()

    caller = db.query(Caller).filter(Caller.phone_number == phone_number).first()
    profile = None
    if caller:
        profile = (
            db.query(FarmerProfilePG)
            .filter(FarmerProfilePG.caller_id == caller.caller_id)
            .first()
        )

    if not kb and not caller and not profile:
        raise HTTPException(status_code=404, detail="Farmer not found")

    return {
        "id": kb.id if kb else None,
        "phone_number": phone_number,
        "name": (kb.name if kb and kb.name else (caller.full_name if caller else None)),
        "location": (
            (kb.location if kb and kb.location else None)
            or (profile.location if profile else None)
        ),
        "language": (
            (kb.preferred_language if kb and kb.preferred_language else None)
            or (profile.primary_language if profile else None)
            or "am"
        ),
        "crops": kb.crops if kb else None,
        "farm_size": (
            kb.farm_size if kb and kb.farm_size is not None else (profile.farm_size if profile else None)
        ),
        "notes": kb.notes if kb else None,
        "registered_at": _isoformat(
            (kb.registered_at if kb else None)
            or (profile.created_at if profile else None)
            or (caller.created_at if caller else None)
        ),
    }


@router.get("/farmers/{phone_number}/calls")
def get_farmer_calls(
    phone_number: str,
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    caller = db.query(Caller).filter(Caller.phone_number == phone_number).first()
    if not caller:
        return []
    rows = (
        db.query(CallSessionPG)
        .filter(CallSessionPG.caller_id == caller.caller_id)
        .order_by(desc(CallSessionPG.start_time))
        .all()
    )
    return [
        {
            "id": r.session_id,
            "session_id": r.session_id,
            "phone_number": phone_number,
            "duration": int(r.duration_seconds) if r.duration_seconds is not None else None,
            "timestamp": _isoformat(r.start_time),
            "recording_path": _playable_call_recording_ref(r.session_id, r.audio_file_path),
        }
        for r in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Calls
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/calls")
def list_calls(
    limit: int = Query(default=100, le=500),
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CallSessionPG, Caller).outerjoin(Caller, Caller.caller_id == CallSessionPG.caller_id)
    
    if user.role == "expert":
        # Experts see calls that have an escalation assigned to them or are pending
        query = query.join(Escalation, Escalation.session_id == CallSessionPG.session_id)\
                     .filter(
                         (Escalation.assigned_to_user_id == user.user_id) | 
                         (Escalation.status == "pending")
                     )
    elif user.role not in ("admin", "da"):
        raise HTTPException(status_code=403, detail="Access denied")

    rows = query.order_by(desc(CallSessionPG.start_time)).limit(limit).all()
    return [
        {
            "id": cs.session_id,
            "session_id": cs.session_id,
            "phone_number": caller.phone_number if caller else None,
            "farmer_name": caller.full_name if caller else None,
            "duration": int(cs.duration_seconds) if cs.duration_seconds is not None else None,
            "timestamp": _isoformat(cs.start_time),
            "recording_path": _playable_call_recording_ref(cs.session_id, cs.audio_file_path),
        }
        for cs, caller in rows
    ]


@router.get("/calls/{session_id}")
def get_call_detail(
    session_id: str,
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.role == "expert":
        # Check if this expert is allowed to see this call
        has_esc = db.query(Escalation).filter(
            Escalation.session_id == session_id,
            (Escalation.assigned_to_user_id == user.user_id) | (Escalation.status == "pending")
        ).first()
        if not has_esc:
            raise HTTPException(status_code=403, detail="You are not assigned to this call session")
    elif user.role not in ("admin", "da"):
        raise HTTPException(status_code=403, detail="Access denied")
    # Primary: call session from phone_gateway
    cs = db.query(CallSessionPG).filter(CallSessionPG.session_id == session_id).first()
    caller = None
    if cs and cs.caller_id:
        caller = db.query(Caller).filter(Caller.caller_id == cs.caller_id).first()

    # Overlay: farmer_kb for extra fields if present
    farmer_kb = None
    if caller and caller.phone_number:
        farmer_kb = db.query(FarmerKB).filter(FarmerKB.phone_number == caller.phone_number).first()
    transcript_rows = (
        db.query(ConversationMessage)
        .filter(ConversationMessage.session_id == session_id)
        .order_by(ConversationMessage.timestamp.asc())
        .all()
    )

    if not cs and not transcript_rows:
        raise HTTPException(status_code=404, detail="Call not found")

    return {
        "session_id": session_id,
        "record": (
            {
                "id": session_id,
                "phone_number": caller.phone_number if caller else None,
                "duration": int(cs.duration_seconds) if cs and cs.duration_seconds is not None else None,
                "timestamp": _isoformat(cs.start_time) if cs else None,
                "recording_path": _playable_call_recording_ref(session_id, cs.audio_file_path if cs else None),
            }
            if cs or caller
            else None
        ),
        "farmer": (
            {
                "phone_number": caller.phone_number,
                "name": (farmer_kb.name if farmer_kb and farmer_kb.name else caller.full_name),
                "location": farmer_kb.location if farmer_kb else None,
                "language": farmer_kb.preferred_language if farmer_kb else "am",
            }
            if caller
            else None
        ),
        "transcript": [
            {
                "role": m.role,
                "message": m.message,
                "timestamp": _isoformat(m.timestamp),
            }
            for m in transcript_rows
        ],
    }


@router.get("/calls/{session_id}/audio")
def get_call_audio_url(
    session_id: str,
    token: Optional[str] = Query(None),
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.role == "expert":
        has_esc = db.query(Escalation).filter(
            Escalation.session_id == session_id,
            (Escalation.assigned_to_user_id == user.user_id) | (Escalation.status == "pending")
        ).first()
        if not has_esc:
            raise HTTPException(status_code=403, detail="Access denied")
    elif user.role not in ("admin", "da"):
        raise HTTPException(status_code=403, detail="Access denied")
    """Streams local call audio when present, otherwise redirects to S3/MinIO."""
    cs = db.query(CallSessionPG).filter(CallSessionPG.session_id == session_id).first()
    stored_path = cs.audio_file_path if cs else None
    if not cs:
        raise HTTPException(status_code=404, detail="Audio not found")

    # 1. Prefer local file serving from the shared recording volume.
    local_path = _first_existing_call_recording(session_id, stored_path)
    if local_path:
        media_type = "audio/wav" if local_path.lower().endswith(".wav") else "application/octet-stream"
        return FileResponse(local_path, media_type=media_type)

    # 2. Try S3/MinIO presigned redirect.
    if stored_path and stored_path.startswith("s3://"):
        if s3_enabled():
            try:
                url = presign_get_url(stored_path, expires_seconds=900)
                if url:
                    return Response(status_code=302, headers={"Location": url})
            except Exception as exc:
                print(f"[CALL AUDIO] S3 presign failed for {session_id}: {exc}")

        raise HTTPException(status_code=404, detail="Audio is stored in S3, but no playable local fallback exists")

    raise HTTPException(status_code=404, detail=f"Audio not found for session {session_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Escalations
# ──────────────────────────────────────────────────────────────────────────────
def _escalation_dict(e: Escalation, db: Session) -> dict:
    assignee = None
    if e.assigned_to_user_id:
        u = db.query(DashboardUser).filter(DashboardUser.user_id == e.assigned_to_user_id).first()
        if u:
            assignee = {"user_id": u.user_id, "full_name": u.full_name, "email": u.email}
    transcript_messages, transcript_text = _get_escalation_transcript(e, db)
    session_record = _get_escalation_session_record(e, db)
    return {
        "id": e.id,
        "query": e.query,
        "context": e.context,
        "phone_number": e.phone_number,
        "session_id": e.session_id,
        "status": e.status,
        "reason_code": e.reason_code,
        "confidence": round(e.confidence, 4) if e.confidence is not None else None,
        "entities": e.entities,
        "assigned_to": assignee,
        "assigned_at": _isoformat(e.assigned_at),
        "expert_response": e.expert_response,
        "expert_notes": e.expert_notes,
        "answered_at": _isoformat(e.answered_at),
        "closed_at": _isoformat(e.closed_at),
        "timestamp": _isoformat(e.created_at),
        "expert_audio_url": _get_expert_audio_link(e, db),
        "transcript": transcript_text,
        "transcript_messages": transcript_messages,
        "session_record": session_record,
        "session_recording_url": session_record.get("audio_url") if session_record else None,
        "session_recording_path": session_record.get("recording_path") if session_record else None,
    }


def _get_escalation_transcript(e: Escalation, db: Session) -> tuple[list[dict], str]:
    rows = []
    if e.session_id:
        rows = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == e.session_id)
            .order_by(ConversationMessage.timestamp.asc(), ConversationMessage.id.asc())
            .all()
        )
    if rows:
        messages = [
            {
                "role": r.role,
                "message": r.message,
                "timestamp": _isoformat(r.timestamp),
            }
            for r in rows
        ]
        text_value = "\n".join(
            f"{m['role']}: {m['message']}"
            for m in messages
            if (m.get("message") or "").strip()
        )
        return messages, text_value

    snapshot = (e.transcript_snapshot or "").strip()
    if not snapshot:
        snapshot = f"user: {e.query}".strip()
    messages = []
    for line in snapshot.splitlines():
        role, _, message = line.partition(":")
        messages.append(
            {
                "role": role.strip() or "user",
                "message": message.strip() if message else line.strip(),
                "timestamp": None,
            }
        )
    return messages, snapshot


def _get_escalation_session_record(e: Escalation, db: Session) -> Optional[dict]:
    if not e.session_id:
        return None
    cs = db.query(CallSessionPG).filter(CallSessionPG.session_id == e.session_id).first()
    stored_path = (cs.audio_file_path if cs and cs.audio_file_path else None) or e.session_recording_path
    recording_path = _playable_call_recording_ref(e.session_id, stored_path)
    return {
        "session_id": e.session_id,
        "status": cs.status if cs else None,
        "duration_seconds": int(cs.duration_seconds) if cs and cs.duration_seconds is not None else None,
        "timestamp": _isoformat(cs.start_time) if cs else None,
        "recording_path": recording_path,
        "audio_url": f"/api/admin/calls/{e.session_id}/audio" if recording_path else None,
    }


def _get_expert_audio_link(e: Escalation, db: Session) -> Optional[str]:
    if not e.expert_audio_path:
        return None
    # If it's an S3 ref, we can't easily presign here without logic, 
    # but the frontend will call /escalations/{id}/audio anyway.
    return f"/api/admin/escalations/{e.id}/audio"


def _farmer_utterance_basename_for_escalation(e: Escalation) -> Optional[str]:
    entities = e.entities if isinstance(e.entities, dict) else {}
    basename = (entities or {}).get("farmer_utterance_basename")
    return str(basename).strip() if basename else None


def _phone_number_for_escalation(e: Escalation, db: Session) -> Optional[str]:
    if e.phone_number:
        return e.phone_number
    if e.session_id:
        row = (
            db.query(CallSessionPG, Caller)
            .outerjoin(Caller, Caller.caller_id == CallSessionPG.caller_id)
            .filter(CallSessionPG.session_id == e.session_id)
            .first()
        )
        if row and row[1] and row[1].phone_number:
            return row[1].phone_number
    return None


def _phone_gateway_base_url() -> str:
    base = (
        os.getenv("PHONE_GATEWAY_EXPERT_CALLBACK_URL")
        or os.getenv("PHONE_GATEWAY_ALERT_CALL_URL")
        or os.getenv("PHONE_GATEWAY_URL", "http://phone-gateway:8000")
    ).rstrip("/")
    if base.endswith("/health"):
        base = base[: -len("/health")]
    return base


def _escalation_session_is_active(e: Escalation, db: Session) -> bool:
    if not e.session_id:
        return False
    cs = db.query(CallSessionPG).filter(CallSessionPG.session_id == e.session_id).first()
    if not cs:
        return False
    return (cs.status or "").lower() == "active" or cs.end_time is None


def _trigger_expert_response_call(
    e: Escalation,
    db: Session,
    *,
    audio_path: Optional[str] = None,
) -> dict:
    if os.getenv("EXPERT_RESPONSE_CALL_ENABLED", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return {"status": "disabled"}

    phone = (_phone_number_for_escalation(e, db) or "").strip()
    farmer_utterance_basename = _farmer_utterance_basename_for_escalation(e)
    callback_message = os.getenv(
        "EXPERT_CALLBACK_INTRO_AM",
        "የባለሙያ መልስ ዝግጁ ነው። መጀመሪያ ጥያቄዎን እናጫውታለን።",
    ).strip()

    if not phone:
        return {"status": "skipped", "reason": "missing_phone_number"}
    if not audio_path:
        return {"status": "skipped", "reason": "missing_audio_response"}
    if _escalation_session_is_active(e, db):
        return {"status": "pending", "reason": "original_call_session_active"}

    try:
        response = requests.post(
            f"{_phone_gateway_base_url()}/api/expert-responses/call",
            json={
                "escalation_id": e.id,
                "phone_number": phone,
                "session_id": e.session_id,
                "expert_audio_path": audio_path,
                "farmer_utterance_basename": farmer_utterance_basename,
                "message": callback_message,
            },
            timeout=8,
        )
        if response.status_code >= 400:
            return {"status": "failed", "error": response.text[:500]}
        return response.json()
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _trigger_expert_response_call_after_session_end(
    escalation_id: int,
    *,
    audio_path: Optional[str] = None,
) -> None:
    max_wait = int(os.getenv("EXPERT_RESPONSE_CALLBACK_WAIT_SECONDS", "300") or "300")
    poll = max(1, int(os.getenv("EXPERT_RESPONSE_CALLBACK_POLL_SECONDS", "5") or "5"))
    deadline = time.monotonic() + max_wait

    while True:
        db = SessionLocal()
        try:
            esc = db.query(Escalation).filter(Escalation.id == escalation_id).first()
            if not esc:
                print(f"[EXPERT CALLBACK] escalation {escalation_id} disappeared")
                return
            if not _escalation_session_is_active(esc, db):
                report = _trigger_expert_response_call(
                    esc,
                    db,
                    audio_path=audio_path,
                )
                print(f"[EXPERT CALLBACK] delayed callback report: {report}")
                return
        finally:
            db.close()

        if time.monotonic() >= deadline:
            print(
                f"[EXPERT CALLBACK] delayed callback timed out for escalation {escalation_id}; "
                "expert response remains saved for callback retry/manual playback"
            )
            return
        time.sleep(poll)


def _queue_or_trigger_expert_response_call(
    e: Escalation,
    db: Session,
    background_tasks: BackgroundTasks,
    *,
    audio_path: Optional[str] = None,
) -> dict:
    if _escalation_session_is_active(e, db):
        background_tasks.add_task(
            _trigger_expert_response_call_after_session_end,
            e.id,
            audio_path=audio_path,
        )
        return {"status": "pending", "reason": "will_call_after_original_session_ends"}
    return _trigger_expert_response_call(e, db, audio_path=audio_path)


@router.get("/escalations")
def list_escalations(
    status: Optional[str] = None,
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_runtime_schema()
    q = db.query(Escalation)
    if status:
        q = q.filter(Escalation.status == status)
    if user.role == "expert":
        # Experts ONLY see escalations assigned to them.
        # (They no longer see pending ones to avoid clutter/privacy issues)
        q = q.filter(Escalation.assigned_to_user_id == user.user_id)
    rows = q.order_by(desc(Escalation.created_at)).all()
    return [_escalation_dict(r, db) for r in rows]


@router.get("/escalations/mine")
def list_my_escalations(
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_runtime_schema()
    rows = (
        db.query(Escalation)
        .filter(Escalation.assigned_to_user_id == user.user_id)
        .order_by(desc(Escalation.created_at))
        .all()
    )
    return [_escalation_dict(r, db) for r in rows]


@router.post("/escalations/{ticket_id}/assign")
def assign_escalation(
    ticket_id: int,
    req: AssignEscalationRequest,
    user: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc:
        raise HTTPException(status_code=404, detail="Escalation not found")
    target = db.query(DashboardUser).filter(DashboardUser.user_id == req.user_id).first()
    if not target or target.role != "expert":
        raise HTTPException(status_code=400, detail="Target user must be an active expert")
    if not target.is_active:
        raise HTTPException(status_code=400, detail="Target expert is deactivated")

    esc.assigned_to_user_id = target.user_id
    esc.assigned_at = datetime.utcnow()
    esc.status = "assigned"
    esc.updated_at = datetime.utcnow()
    db.commit()
    return _escalation_dict(esc, db)


@router.post("/escalations/{ticket_id}/response")
def respond_escalation(
    ticket_id: int,
    req: EscalationResponseRequest,
    background_tasks: BackgroundTasks,
    user: DashboardUser = Depends(require_roles("expert", "admin")),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc:
        raise HTTPException(status_code=404, detail="Escalation not found")
    if user.role == "expert" and esc.assigned_to_user_id != user.user_id:
        raise HTTPException(status_code=403, detail="This case is not assigned to you")

    esc.expert_response = req.answer
    if req.expert_notes is not None:
        esc.expert_notes = req.expert_notes
    esc.answered_at = datetime.utcnow()
    esc.status = "answered"
    esc.updated_at = datetime.utcnow()
    db.commit()
    callback_report = {"status": "skipped", "reason": "audio_response_required_for_callback"}
    out = _escalation_dict(esc, db)
    out["expert_callback"] = callback_report
    return out


@router.post("/escalations/{ticket_id}/audio-response")
async def upload_expert_audio(
    ticket_id: int,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    user: DashboardUser = Depends(require_roles("expert", "admin")),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc:
        raise HTTPException(status_code=404, detail="Escalation not found")
    if user.role == "expert" and esc.assigned_to_user_id != user.user_id:
        raise HTTPException(status_code=403, detail="This case is not assigned to you")

    # Save audio file
    recordings_dir = os.path.join(os.environ.get("DATA_DIR", "/data"), "expert_responses")
    os.makedirs(recordings_dir, exist_ok=True)
    filename = f"esc_{ticket_id}_{uuid.uuid4().hex[:8]}.wav"
    file_path = os.path.join(recordings_dir, filename)

    raw_path = f"{file_path}.upload"
    content = await audio_file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                raw_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                file_path,
            ],
            check=True,
        )
    except Exception as exc:
        print(f"[EXPERT AUDIO] ffmpeg normalization failed, keeping raw upload: {exc}")
        os.replace(raw_path, file_path)
    else:
        try:
            os.remove(raw_path)
        except OSError:
            pass

    # Optional: upload to S3 as a backup, but keep the shared local WAV path in
    # the DB so phone-gateway can play it during the outbound expert callback.
    final_path = file_path
    if s3_enabled():
        from s3_client import upload_file as s3_upload
        try:
            s3_upload(file_path, f"expert_responses/{filename}", content_type="audio/wav")
        except Exception as exc:
            print(f"[EXPERT AUDIO] S3 upload failed, keeping local: {exc}")

    esc.expert_audio_path = final_path
    esc.answered_at = datetime.utcnow()
    esc.status = "answered"
    esc.updated_at = datetime.utcnow()
    db.commit()
    callback_report = _queue_or_trigger_expert_response_call(
        esc,
        db,
        background_tasks,
        audio_path=final_path,
    )
    if callback_report.get("status") == "failed":
        print(f"[EXPERT CALLBACK] audio response call failed: {callback_report}")
    out = _escalation_dict(esc, db)
    out["expert_callback"] = callback_report
    return out


@router.get("/escalations/{ticket_id}/audio")
def get_expert_audio(
    ticket_id: int,
    token: Optional[str] = Query(None),
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc or not esc.expert_audio_path:
        raise HTTPException(status_code=404, detail="Audio response not found")

    # RBAC check
    if user.role == "expert" and esc.assigned_to_user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if esc.expert_audio_path.startswith("s3://"):
        if not s3_enabled():
            raise HTTPException(status_code=503, detail="S3 is not configured")
        url = presign_get_url(esc.expert_audio_path, expires_seconds=900)
        if url:
            return Response(status_code=302, headers={"Location": url})
        raise HTTPException(status_code=404, detail="Audio reference is not in S3")

    if os.path.exists(esc.expert_audio_path):
        return FileResponse(esc.expert_audio_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file missing")


@router.post("/escalations/{ticket_id}/close")
def close_escalation(
    ticket_id: int,
    user: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc:
        raise HTTPException(status_code=404, detail="Escalation not found")
    esc.status = "closed"
    esc.closed_at = datetime.utcnow()
    esc.updated_at = datetime.utcnow()
    db.commit()
    return _escalation_dict(esc, db)


# Backwards-compat shortcut used by the existing /helpdesk page
@router.put("/escalations/{ticket_id}/resolve")
def resolve_escalation(
    ticket_id: int,
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    esc = db.query(Escalation).filter(Escalation.id == ticket_id).first()
    if not esc:
        raise HTTPException(status_code=404, detail="Escalation not found")
    esc.status = "closed" if esc.status != "answered" else "closed"
    esc.closed_at = datetime.utcnow()
    esc.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "ok", "ticket_id": ticket_id}


# ──────────────────────────────────────────────────────────────────────────────
# Market prices
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/market-prices")
def list_market_prices(
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(MarketPrice).order_by(desc(MarketPrice.updated_at)).all()
    return [
        {
            "id": r.id,
            "crop_name": r.crop_name,
            "region": r.region,
            "price": r.price,
            "unit": r.unit,
            "updated_at": _isoformat(r.updated_at),
        }
        for r in rows
    ]


@router.post("/market-prices")
def add_market_price(
    req: MarketPriceRequest,
    _: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    db.add(
        MarketPrice(
            crop_name=req.crop_name,
            region=req.region,
            price=req.price,
            unit=req.unit,
        )
    )
    db.commit()
    return {"status": "ok"}


@router.delete("/market-prices/{price_id}")
def delete_market_price(
    price_id: int,
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    row = db.query(MarketPrice).filter(MarketPrice.id == price_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Market price not found")
    db.delete(row)
    db.commit()
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────────────────────────────────────
def _farmers_for_alert_region(db: Session, target_region: str) -> list[dict]:
    target = _normalize_region(target_region)
    rows = (
        db.query(Caller, FarmerProfilePG, FarmerKB)
        .outerjoin(FarmerProfilePG, FarmerProfilePG.caller_id == Caller.caller_id)
        .outerjoin(FarmerKB, FarmerKB.phone_number == Caller.phone_number)
        .all()
    )
    out: list[dict] = []
    seen: set[str] = set()
    for caller, pg_profile, kb_profile in rows:
        phone = (caller.phone_number or "").strip()
        if not phone or phone in seen:
            continue
        locations = [
            getattr(pg_profile, "location", None),
            getattr(kb_profile, "location", None),
        ]
        if target not in {"", "all"} and not any(
            target in _normalize_region(loc) for loc in locations if loc
        ):
            continue
        seen.add(phone)
        out.append(
            {
                "phone_number": phone,
                "name": caller.full_name,
                "location": next((loc for loc in locations if loc), None),
            }
        )
    return out


def _request_alert_call(
    *,
    alert_id: int,
    phone_number: str,
    target_region: str,
    alert_message: str,
    severity: str,
) -> tuple[str, Optional[str], Optional[str]]:
    phone_gateway_url = (
        os.getenv("PHONE_GATEWAY_ALERT_CALL_URL")
        or os.getenv("PHONE_GATEWAY_URL", "http://phone-gateway:8000")
    ).rstrip("/")
    if phone_gateway_url.endswith("/health"):
        phone_gateway_url = phone_gateway_url[: -len("/health")]
    try:
        response = requests.post(
            f"{phone_gateway_url}/api/alerts/call",
            json={
                "alert_id": alert_id,
                "phone_number": phone_number,
                "target_region": target_region,
                "alert_message": alert_message,
                "severity": severity,
            },
            timeout=8,
        )
        if response.status_code >= 400:
            return "failed", None, response.text[:500]
        payload = response.json()
        return str(payload.get("status") or "queued"), payload.get("call_id"), None
    except Exception as exc:
        return "failed", None, str(exc)


def _enqueue_alert_calls(db: Session, alert: Alert) -> dict:
    recipients = _farmers_for_alert_region(db, alert.target_region)
    max_recipients = int(os.getenv("ALERT_CALL_MAX_RECIPIENTS", "50") or "50")
    recipients = recipients[: max(0, max_recipients)]
    queued = 0
    failed = 0
    for recipient in recipients:
        status, provider_ref, error = _request_alert_call(
            alert_id=alert.id,
            phone_number=recipient["phone_number"],
            target_region=alert.target_region,
            alert_message=alert.alert_message,
            severity=alert.severity,
        )
        row = AlertCallNotification(
            alert_id=alert.id,
            phone_number=recipient["phone_number"],
            target_region=recipient.get("location") or alert.target_region,
            status=status,
            provider_ref=provider_ref,
            error=error,
        )
        db.add(row)
        if status in {"queued", "calling", "sent", "scheduled"}:
            queued += 1
        else:
            failed += 1
    return {"recipients": len(recipients), "queued": queued, "failed": failed}


@router.get("/alerts")
def list_alerts(
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(Alert).order_by(desc(Alert.created_at)).limit(200).all()
    counts = dict(
        db.query(
            AlertCallNotification.alert_id,
            func.count(AlertCallNotification.id),
        )
        .group_by(AlertCallNotification.alert_id)
        .all()
    )
    return [
        {
            "id": r.id,
            "target_region": r.target_region,
            "alert_message": r.alert_message,
            "severity": r.severity,
            "category": r.category,
            "scheduled_at": _isoformat(r.scheduled_at),
            "published_at": _isoformat(r.published_at),
            "created_at": _isoformat(r.created_at),
            "call_notification_count": int(counts.get(r.id, 0)),
        }
        for r in rows
    ]


@router.post("/alerts")
def create_alert_endpoint(
    req: AlertRequest,
    user: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    alert = Alert(
        target_region=req.target_region,
        alert_message=req.alert_message,
        severity=req.severity,
        category=req.category,
        scheduled_at=req.scheduled_at,
        published_at=datetime.utcnow() if not req.scheduled_at else None,
        created_by=user.user_id,
    )
    db.add(alert)
    db.flush()
    call_report = {"recipients": 0, "queued": 0, "failed": 0}
    if req.notify_by_call and not req.scheduled_at:
        call_report = _enqueue_alert_calls(db, alert)
    db.commit()
    return {"status": "ok", "id": alert.id, "call_notifications": call_report}


@router.delete("/alerts/{alert_id}")
def delete_alert(
    alert_id: int,
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(row)
    db.commit()
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Base — legacy Chroma entries + active RAG pgvector documents
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/kb")
def list_kb(
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entries: list[dict] = []

    # Active RAG store: this is what /rag/answer retrieves from.
    try:
        rows = db.execute(
            text(
                """
                SELECT
                  d.id::text,
                  COALESCE(d.title, d.original_filename, d.external_document_id, d.id::text) AS title,
                  d.original_filename,
                  d.source_org,
                  d.status,
                  d.updated_at,
                  COUNT(c.chunk_index) AS chunk_count,
                  LEFT(MIN(c.content), 500) AS sample_content
                FROM rag_kb_documents d
                LEFT JOIN rag_kb_chunks c ON c.document_id = d.id
                GROUP BY d.id, d.title, d.original_filename, d.external_document_id,
                         d.source_org, d.status, d.updated_at
                ORDER BY d.updated_at DESC NULLS LAST, title
                LIMIT 1000;
                """
            )
        ).mappings().all()
        entries.extend(
            {
                "id": f"rag:{row['id']}",
                "intent": row["title"],
                "title": row["title"],
                "category": row["source_org"] or "rag_pgvector",
                "response": row["sample_content"] or "",
                "content": row["sample_content"] or "",
                "source": "rag_pgvector",
                "filename": row["original_filename"],
                "status": row["status"],
                "chunk_count": int(row["chunk_count"] or 0),
                "updated_at": _isoformat(row["updated_at"]),
            }
            for row in rows
        )
    except Exception as exc:
        print(f"[KB] active RAG pgvector list skipped: {exc}")

    # Legacy/manual Chroma entries: small hand-authored intent responses.
    try:
        if collection.count() > 0:
            result = collection.get(include=["documents", "metadatas"])
            entries.extend(
                {
                    "id": f"chroma:{doc_id}",
                    "intent": meta.get("intent", ""),
                    "response": doc,
                    "content": doc,
                    "source": "legacy_chroma",
                }
                for doc_id, doc, meta in zip(
                    result["ids"], result["documents"], result["metadatas"]
                )
            )
    except Exception as exc:
        print(f"[KB] legacy Chroma list skipped: {exc}")

    return entries


@router.post("/kb")
def add_kb(
    req: KBRequest,
    _: DashboardUser = Depends(require_roles("admin")),
):
    doc_id = f"kb_{uuid.uuid4()}"
    collection.add(
        documents=[req.response],
        metadatas=[{"intent": req.intent}],
        ids=[doc_id],
    )
    return {"status": "ok", "id": doc_id}


@router.delete("/kb/{entry_id}")
def delete_kb(
    entry_id: str,
    _: DashboardUser = Depends(require_roles("admin")),
):
    if entry_id.startswith("rag:"):
        raise HTTPException(
            status_code=400,
            detail="RAG pgvector documents are managed from Knowledge Documents, not this legacy delete action.",
        )
    if entry_id.startswith("chroma:"):
        entry_id = entry_id[len("chroma:"):]
    try:
        collection.delete(ids=[entry_id])
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Base — uploaded documents
# ──────────────────────────────────────────────────────────────────────────────
KB_UPLOAD_DIR = os.path.join(os.environ.get("DATA_DIR", "/data"), "kb_uploads")
os.makedirs(KB_UPLOAD_DIR, exist_ok=True)


def _kb_doc_dict(d: KBDocument) -> dict:
    return {
        "id": d.id,
        "filename": d.filename,
        "title": d.title,
        "description": d.description,
        "topic": d.topic,
        "crop": d.crop,
        "region": d.region,
        "category": d.category,
        "status": d.status,
        "indexing_status": d.indexing_status,
        "indexing_error": d.indexing_error,
        "chroma_doc_count": d.chroma_doc_count,
        "uploaded_at": _isoformat(d.uploaded_at),
        "approved_at": _isoformat(d.approved_at),
        "last_indexed_at": _isoformat(d.last_indexed_at),
    }


@router.get("/kb/documents")
def list_kb_documents(
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    rows = db.query(KBDocument).order_by(desc(KBDocument.uploaded_at)).all()
    return [_kb_doc_dict(d) for d in rows]


@router.post("/kb/documents")
async def upload_kb_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    crop: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    user: DashboardUser = Depends(require_roles("admin", "expert")),
    db: Session = Depends(get_db),
):
    doc_id = str(uuid.uuid4())
    safe_name = file.filename or f"{doc_id}.bin"
    storage_path = os.path.join(KB_UPLOAD_DIR, f"{doc_id}_{safe_name}")
    with open(storage_path, "wb") as out:
        out.write(await file.read())

    record = KBDocument(
        id=doc_id,
        filename=safe_name,
        storage_path=storage_path,
        mime_type=file.content_type,
        title=title or safe_name,
        description=description,
        topic=topic,
        crop=crop,
        region=region,
        category=category,
        status="uploaded",
        uploaded_by=user.user_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _kb_doc_dict(record)


@router.put("/kb/documents/{doc_id}")
def update_kb_document(
    doc_id: str,
    req: KBDocumentUpdate,
    _: DashboardUser = Depends(require_roles("admin", "expert")),
    db: Session = Depends(get_db),
):
    doc = db.query(KBDocument).filter(KBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    for field, value in req.dict(exclude_unset=True).items():
        setattr(doc, field, value)
    db.commit()
    return _kb_doc_dict(doc)


@router.post("/kb/documents/{doc_id}/approve")
def approve_kb_document(
    doc_id: str,
    user: DashboardUser = Depends(require_roles("admin", "expert")),
    db: Session = Depends(get_db),
):
    doc = db.query(KBDocument).filter(KBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.status = "approved"
    doc.approved_by = user.user_id
    doc.approved_at = datetime.utcnow()
    db.commit()
    try:
        index_document(db, doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    # Best-effort: also ingest into the RAG service (pgvector static KB).
    try:
        if doc.storage_path and os.path.exists(doc.storage_path):
            with open(doc.storage_path, "rb") as f:
                files = {"file": (doc.filename, f, doc.mime_type or "application/octet-stream")}
                data = {
                    "external_document_id": doc.id,
                    "title": doc.title or doc.filename,
                    "source_org": "admin_dashboard",
                    "source_url": "",
                    "language": "am",
                    "status": "approved",
                }
                requests.post(f"{RAG_SERVICE_URL}/kb/ingest", files=files, data=data, timeout=120)
    except Exception:
        # Don't block approval if RAG service is offline.
        pass
    return _kb_doc_dict(doc)


@router.post("/kb/documents/{doc_id}/reject")
def reject_kb_document(
    doc_id: str,
    user: DashboardUser = Depends(require_roles("admin", "expert")),
    db: Session = Depends(get_db),
):
    doc = db.query(KBDocument).filter(KBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    remove_document_from_chroma(db, doc)
    doc.status = "rejected"
    doc.approved_by = user.user_id
    doc.approved_at = datetime.utcnow()
    doc.indexing_status = "pending"
    doc.chroma_doc_count = 0
    db.commit()
    return _kb_doc_dict(doc)


@router.post("/kb/documents/{doc_id}/reindex")
def reindex_kb_document(
    doc_id: str,
    _: DashboardUser = Depends(require_roles("admin", "expert")),
    db: Session = Depends(get_db),
):
    doc = db.query(KBDocument).filter(KBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        index_document(db, doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}")

    # Best-effort: update the RAG service index as well.
    try:
        if doc.storage_path and os.path.exists(doc.storage_path):
            with open(doc.storage_path, "rb") as f:
                files = {"file": (doc.filename, f, doc.mime_type or "application/octet-stream")}
                data = {
                    "external_document_id": doc.id,
                    "title": doc.title or doc.filename,
                    "source_org": "admin_dashboard",
                    "source_url": "",
                    "language": "am",
                    "status": "approved" if doc.status == "approved" else doc.status,
                }
                requests.post(f"{RAG_SERVICE_URL}/kb/ingest", files=files, data=data, timeout=120)
    except Exception:
        pass
    return _kb_doc_dict(doc)


@router.delete("/kb/documents/{doc_id}")
def delete_kb_document(
    doc_id: str,
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    doc = db.query(KBDocument).filter(KBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    remove_document_from_chroma(db, doc)
    if doc.storage_path and os.path.exists(doc.storage_path):
        try:
            os.remove(doc.storage_path)
        except OSError:
            pass
    db.delete(doc)
    db.commit()
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────────────────────────────────────
def _probe(url: str, timeout: float = 3.0) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        return {
            "url": url,
            "status": "online" if r.status_code < 500 else "degraded",
            "http_status": r.status_code,
        }
    except Exception as exc:
        return {"url": url, "status": "down", "error": str(exc)}


@router.get("/system-status")
def system_status(
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    services = {
        "vad": os.getenv("VAD_HEALTH_URL", "http://vad-service:8010/health"),
        "asr": os.getenv("ASR_HEALTH_URL", "http://asr-service:8001/docs"),
        "tts": os.getenv("TTS_HEALTH_URL", "http://tts-service:8000/docs"),
        "phone_gateway": os.getenv("PHONE_GATEWAY_URL", "http://phone-gateway:8000/health"),
    }
    probed = {name: _probe(url) for name, url in services.items()}

    # Internal checks
    db_status = "ok"
    try:
        from sqlalchemy import text
        from db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    chroma_status = "ok"
    chroma_count = 0
    try:
        chroma_count = collection.count()
    except Exception as exc:
        chroma_status = f"error: {exc}"

    rag_status = {
        "status": "online" if db_status == "ok" else "degraded",
        "chroma_docs": chroma_count,
        "chroma_status": chroma_status,
    }

    recent_errors = (
        db.query(ServiceError).order_by(desc(ServiceError.created_at)).limit(20).all()
    )
    errors = [
        {
            "id": e.id,
            "service": e.service,
            "endpoint": e.endpoint,
            "method": e.method,
            "status_code": e.status_code,
            "error": e.error,
            "created_at": _isoformat(e.created_at),
        }
        for e in recent_errors
    ]

    return {
        "services": {
            "vad": probed["vad"],
            "asr": probed["asr"],
            "tts": probed["tts"],
            "phone_gateway": probed["phone_gateway"],
            "logic_service": {"status": "online", "url": "self"},
            "rag": rag_status,
            "database": {"status": db_status},
        },
        "recent_errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/analytics/summary")
def analytics_summary(
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    # Use phone_gateway's canonical tables (call_sessions/callers) so dashboard
    # reflects actual telephony activity.
    total_calls = db.query(func.count(CallSessionPG.session_id)).scalar() or 0
    calls_30d = (
        db.query(func.count(CallSessionPG.session_id))
        .filter(CallSessionPG.start_time >= thirty_days_ago)
        .scalar()
        or 0
    )
    total_farmers = db.query(func.count(Caller.caller_id)).scalar() or 0
    new_farmers_30d = (
        db.query(func.count(Caller.caller_id))
        .filter(Caller.created_at >= thirty_days_ago)
        .scalar()
        or 0
    )
    open_escalations = (
        db.query(func.count(Escalation.id))
        .filter(Escalation.status.in_(("pending", "assigned")))
        .scalar()
        or 0
    )
    answered_escalations = (
        db.query(func.count(Escalation.id))
        .filter(Escalation.status.in_(("answered", "closed")))
        .scalar()
        or 0
    )
    return {
        "total_calls": total_calls,
        "calls_30d": calls_30d,
        "total_farmers": total_farmers,
        "new_farmers_30d": new_farmers_30d,
        "open_escalations": open_escalations,
        "answered_escalations": answered_escalations,
    }


@router.get("/analytics/common-questions")
def common_questions(
    limit: int = Query(default=10, le=100),
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ConversationMessage.message, func.count(ConversationMessage.id))
        .filter(ConversationMessage.role == "user")
        .group_by(ConversationMessage.message)
        .order_by(desc(func.count(ConversationMessage.id)))
        .limit(limit)
        .all()
    )
    return [{"question": r[0], "count": r[1]} for r in rows]


@router.get("/analytics/calls-breakdown")
def calls_breakdown(
    by: str = Query(default="date"),
    _: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if by == "language":
        rows = (
            db.query(
                func.coalesce(FarmerProfilePG.primary_language, "unknown").label("lang"),
                func.count(CallSessionPG.session_id),
            )
            .join(Caller, Caller.caller_id == CallSessionPG.caller_id)
            .outerjoin(FarmerProfilePG, FarmerProfilePG.caller_id == Caller.caller_id)
            .group_by("lang")
            .order_by(desc(func.count(CallSessionPG.session_id)))
            .all()
        )
        return [{"key": r[0] or "unknown", "count": r[1]} for r in rows]

    if by == "region":
        rows = (
            db.query(
                func.coalesce(FarmerProfilePG.location, "unknown").label("region"),
                func.count(CallSessionPG.session_id),
            )
            .join(Caller, Caller.caller_id == CallSessionPG.caller_id)
            .outerjoin(FarmerProfilePG, FarmerProfilePG.caller_id == Caller.caller_id)
            .group_by("region")
            .order_by(desc(func.count(CallSessionPG.session_id)))
            .all()
        )
        return [{"key": r[0] or "unknown", "count": r[1]} for r in rows]

    # default: by date (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    rows = (
        db.query(
            func.date_trunc("day", CallSessionPG.start_time).label("day"),
            func.count(CallSessionPG.session_id),
        )
        .filter(CallSessionPG.start_time >= thirty_days_ago)
        .group_by("day")
        .order_by("day")
        .all()
    )
    return [
        {"key": (r[0].date().isoformat() if r[0] else ""), "count": r[1]} for r in rows
    ]


@router.get("/analytics/expert-performance")
def expert_performance(
    _: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(
            DashboardUser.user_id,
            DashboardUser.full_name,
            DashboardUser.email,
            func.count(Escalation.id).label("assigned_count"),
            func.sum(
                case((Escalation.status.in_(("answered", "closed")), 1), else_=0)
            ).label("resolved_count"),
        )
        .join(Escalation, Escalation.assigned_to_user_id == DashboardUser.user_id)
        .filter(DashboardUser.role == "expert")
        .group_by(DashboardUser.user_id, DashboardUser.full_name, DashboardUser.email)
        .all()
    )
    return [
        {
            "user_id": r[0],
            "full_name": r[1],
            "email": r[2],
            "assigned": int(r[3] or 0),
            "resolved": int(r[4] or 0),
        }
        for r in rows
    ]


@router.get("/analytics/da-performance")
def da_performance(
    _: DashboardUser = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(
            DashboardUser.user_id,
            DashboardUser.full_name,
            DashboardUser.email,
            func.count(Alert.id).label("alerts_created"),
        )
        .outerjoin(Alert, Alert.created_by == DashboardUser.user_id)
        .filter(DashboardUser.role == "da")
        .group_by(DashboardUser.user_id, DashboardUser.full_name, DashboardUser.email)
        .all()
    )
    return [
        {
            "user_id": r[0],
            "full_name": r[1],
            "email": r[2],
            "alerts_created": int(r[3] or 0),
        }
        for r in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Interaction records (FR16 traceability)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/interaction-records")
def list_interaction_records(
    phone_number: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    user: DashboardUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns structured interaction records written by the voice/RAG pipeline.
    These records are stored in Postgres table `interaction_records`.
    """
    if user.role == "expert":
        # Experts must specify a session_id they are allowed to see
        if not session_id:
            # We could allow them to see all interaction records for their assigned calls, 
            # but usually they access this from the call detail page.
            # To be safe and compliant, we restrict to specific session_id.
            raise HTTPException(status_code=403, detail="Access denied: Expert must specify a session_id")
        
        has_esc = db.query(Escalation).filter(
            Escalation.session_id == session_id,
            (Escalation.assigned_to_user_id == user.user_id) | (Escalation.status == "pending")
        ).first()
        if not has_esc:
            raise HTTPException(status_code=403, detail="Access denied: Not authorized for this session")
    elif user.role not in ("admin", "da"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Best-effort: the table is created by rag_service. If it's missing, return empty.
    where = []
    params: dict = {"limit": limit}
    if phone_number:
        where.append("phone_number = :phone_number")
        params["phone_number"] = phone_number
    if session_id:
        where.append("session_id = :session_id")
        params["session_id"] = session_id
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    try:
        rows = db.execute(
            text(
                f"""
                SELECT
                  id,
                  phone_number,
                  session_id,
                  intent,
                  response_type,
                  entities,
                  confidence,
                  created_at
                FROM interaction_records
                {where_sql}
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            params,
        ).mappings().all()
    except Exception:
        return []

    def _row(r) -> dict:
        return {
            "id": int(r["id"]) if r.get("id") is not None else None,
            "phone_number": r.get("phone_number"),
            "session_id": r.get("session_id"),
            "intent": r.get("intent"),
            "response_type": r.get("response_type"),
            "entities": r.get("entities"),
            "confidence": float(r["confidence"]) if r.get("confidence") is not None else None,
            "created_at": _isoformat(r.get("created_at")),
        }

    return [_row(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# CSV exports
# ──────────────────────────────────────────────────────────────────────────────
EXPORT_RESOURCES = ("calls", "farmers", "escalations", "market-prices", "alerts")


def _csv_response(rows: List[dict], filename: str) -> StreamingResponse:
    if not rows:
        # Still return a header row when there are no rows; fall back to an empty file.
        buf = io.StringIO()
        buf.write("\n")
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/{resource}.csv")
def export_csv(
    resource: str,
    _: DashboardUser = Depends(require_roles("admin", "da")),
    db: Session = Depends(get_db),
):
    if resource not in EXPORT_RESOURCES:
        raise HTTPException(status_code=400, detail=f"Unknown resource: {resource}")

    if resource == "calls":
        rows = db.query(CallRecord).order_by(desc(CallRecord.timestamp)).all()
        data = [
            {
                "id": r.id,
                "session_id": r.session_id,
                "phone_number": r.phone_number,
                "duration": r.duration,
                "timestamp": _isoformat(r.timestamp),
                "recording_path": r.recording_path,
            }
            for r in rows
        ]
    elif resource == "farmers":
        rows = db.query(FarmerKB).order_by(desc(FarmerKB.registered_at)).all()
        data = [
            {
                "id": r.id,
                "phone_number": r.phone_number,
                "name": r.name,
                "location": r.location,
                "language": r.preferred_language,
                "registered_at": _isoformat(r.registered_at),
            }
            for r in rows
        ]
    elif resource == "escalations":
        rows = db.query(Escalation).order_by(desc(Escalation.created_at)).all()
        data = [
            {
                "id": r.id,
                "query": r.query,
                "context": r.context,
                "status": r.status,
                "phone_number": r.phone_number,
                "session_id": r.session_id,
                "assigned_to_user_id": r.assigned_to_user_id,
                "expert_response": r.expert_response,
                "created_at": _isoformat(r.created_at),
                "answered_at": _isoformat(r.answered_at),
                "closed_at": _isoformat(r.closed_at),
            }
            for r in rows
        ]
    elif resource == "market-prices":
        rows = db.query(MarketPrice).order_by(desc(MarketPrice.updated_at)).all()
        data = [
            {
                "id": r.id,
                "crop_name": r.crop_name,
                "region": r.region,
                "price": r.price,
                "unit": r.unit,
                "updated_at": _isoformat(r.updated_at),
            }
            for r in rows
        ]
    else:  # alerts
        rows = db.query(Alert).order_by(desc(Alert.created_at)).all()
        data = [
            {
                "id": r.id,
                "target_region": r.target_region,
                "alert_message": r.alert_message,
                "severity": r.severity,
                "category": r.category,
                "scheduled_at": _isoformat(r.scheduled_at),
                "published_at": _isoformat(r.published_at),
                "created_at": _isoformat(r.created_at),
            }
            for r in rows
        ]

    return _csv_response(data, f"{resource}.csv")
