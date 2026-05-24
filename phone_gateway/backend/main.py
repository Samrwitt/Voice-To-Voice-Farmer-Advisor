import asyncio
import json
import os
import uuid
from pathlib import Path
from datetime import datetime, timedelta

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import httpx

from backend.database import Base, engine, SessionLocal
from backend.models import Caller
from backend.sessions import create_session, end_session
from backend.recorder import AudioRecorder
from backend.callers import create_or_get_caller
from backend.auth.routes import router as auth_router

from backend.monitor_state import (
    start_call_monitor,
    update_audio_stats,
    update_vad_status,
    add_utterance,
    update_utterance_transcript,
    update_utterance_rag,
    update_utterance_tts,
    end_call_monitor,
    add_event,
    get_monitor_state,
    get_monitor_events,
    get_recent_calls,
)
from backend.bootstrap import seed_default_admin


# ============================================================
# Database Init
# ============================================================

Base.metadata.create_all(bind=engine)
seed_default_admin()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Phone Browser Telephony Gateway")

# Auth routes:
# POST /api/auth/login
# GET  /api/auth/me
# POST /api/auth/users
# GET  /api/auth/users
app.include_router(auth_router)

CALL_RECORDING_RETENTION_DAYS = int(os.getenv("CALL_RECORDING_RETENTION_DAYS", "30") or "30")


# ============================================================
# CORS
# Needed because Next.js admin dashboard runs on localhost:3000
# and backend runs on localhost:8000
# ============================================================

ADMIN_DASHBOARD_ORIGIN = os.getenv(
    "ADMIN_DASHBOARD_ORIGIN",
    "http://localhost:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        ADMIN_DASHBOARD_ORIGIN,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Static Frontend
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/static/utterances", StaticFiles(directory="/app/utterances"), name="utterances")

async def _recording_retention_loop() -> None:
    """
    Best-practice privacy control: periodically delete old local call recordings.
    S3/MinIO uploads are controlled separately by bucket lifecycle policies.
    """
    recordings_base = Path(os.getenv("RECORDINGS_DIR", "/app/recordings"))
    audio_dir = recordings_base / "audio"
    while True:
        try:
            if CALL_RECORDING_RETENTION_DAYS > 0 and audio_dir.exists():
                cutoff = datetime.utcnow() - timedelta(days=CALL_RECORDING_RETENTION_DAYS)
                deleted = 0
                for p in audio_dir.glob("*"):
                    try:
                        if not p.is_file():
                            continue
                        mtime = datetime.utcfromtimestamp(p.stat().st_mtime)
                        if mtime < cutoff:
                            p.unlink(missing_ok=True)
                            deleted += 1
                    except Exception:
                        continue
                if deleted:
                    print(f"[RETENTION] Deleted {deleted} old recordings (>{CALL_RECORDING_RETENTION_DAYS}d)", flush=True)
        except Exception as exc:
            print(f"[RETENTION] Cleanup error: {exc}", flush=True)

        # Run daily.
        await asyncio.sleep(24 * 60 * 60)


class TtsSynthesizeBody(BaseModel):
    """Proxy to tts-service for browser playback (same-origin; avoids exposing internal URLs)."""

    text: str


@app.post("/api/tts/synthesize")
async def proxy_tts_synthesize(body: TtsSynthesizeBody):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.post(TTS_SERVICE_URL, json={"text": text})
            upstream.raise_for_status()
        ct = upstream.headers.get("content-type", "audio/wav")
        return Response(content=upstream.content, media_type=ct)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"TTS upstream error: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"TTS unreachable: {exc}") from exc


# ============================================================
# Configuration
# ============================================================

TTS_SERVICE_URL = os.getenv(
    "TTS_SERVICE_URL",
    "http://tts-service:8009/synthesize",
).strip()

VAD_WS_BASE_URL = os.getenv(
    "VAD_WS_BASE_URL",
    "ws://vad-service:8010/ws/vad",
)


# ============================================================
# Schemas
# ============================================================

class CallerRegisterRequest(BaseModel):
    full_name: str
    phone_number: str


# ============================================================
# Helper Functions
# ============================================================

def get_caller_details(caller_id: str | None):
    if not caller_id:
        return None, None

    db = SessionLocal()

    try:
        caller = db.query(Caller).filter(
            Caller.caller_id == caller_id
        ).first()

        if not caller:
            return None, None

        return caller.full_name, caller.phone_number

    finally:
        db.close()


def calculate_pcm16_audio_level(pcm_bytes: bytes) -> float:
    """
    Calculate rough audio level from PCM16 mono audio.
    Returns a value between 0.0 and 1.0.
    """

    if not pcm_bytes:
        return 0.0

    try:
        import array

        samples = array.array("h")
        samples.frombytes(pcm_bytes)

        if not samples:
            return 0.0

        total = 0.0

        for sample in samples:
            normalized = sample / 32768.0
            total += normalized * normalized

        rms = (total / len(samples)) ** 0.5

        # Boost for visual display.
        visual_level = min(1.0, rms * 8)

        return round(visual_level, 4)

    except Exception:
        return 0.0


def build_asr_transcripts_from_events(events: list[dict]) -> list[dict]:
    """
    Extract ASR transcript objects from monitor events.

    monitor_state.add_event() stores events like:
    {
        "time": "...",
        "event_type": "asr_transcript",
        "payload": {
            "event": "asr_transcript",
            "transcript": "...",
            ...
        }
    }

    So this function reads event_type + payload.
    """

    transcripts = []

    for event in events:
        payload = event.get("payload") or {}

        event_type = (
            event.get("event_type")
            or event.get("event")
            or event.get("type")
            or payload.get("event")
            or ""
        )

        is_transcript_event = event_type in (
            "asr_transcript",
            "transcript_ready",
            "transcript_saved",
        )

        if not is_transcript_event:
            continue

        transcript_text = payload.get("transcript") or event.get("transcript")

        if not transcript_text:
            continue

        transcripts.append({
            "session_id": payload.get("session_id") or event.get("session_id"),
            "utterance_path": payload.get("utterance_path") or event.get("utterance_path"),
            "transcript": transcript_text,
            "confidence": payload.get("confidence") or event.get("confidence"),
            "engine": payload.get("engine") or event.get("engine") or "mock",
            "audio_id": payload.get("audio_id") or event.get("audio_id"),
            "timestamp": (
                event.get("time")
                or payload.get("timestamp")
                or payload.get("created_at")
                or event.get("created_at")
                or ""
            ),
            "message": payload.get("message") or "ASR transcription completed",
            "source": "events",
        })

    return transcripts


def build_asr_transcripts_from_active_call(active_call: dict | None) -> list[dict]:
    """
    Extract ASR transcripts from active_call.utterances.

    Current monitor state stores transcripts inside:
        active_call["utterances"][i]["transcript"]

    This converts those into top-level asr_transcripts.
    """

    if not active_call:
        return []

    session_id = active_call.get("session_id")
    transcripts = []

    for utterance in active_call.get("utterances", []):
        transcript_text = utterance.get("transcript")

        if not transcript_text:
            continue

        transcripts.append({
            "session_id": session_id,
            "utterance_path": utterance.get("utterance_path"),
            "transcript": transcript_text,
            "confidence": utterance.get("confidence"),
            "engine": utterance.get("engine", "mock"),
            "audio_id": utterance.get("audio_id"),
            "timestamp": utterance.get("created_at") or "",
            "message": "ASR transcription completed",
            "source": "active_call.utterances",
        })

    return transcripts


def dedupe_asr_transcripts(transcripts: list[dict]) -> list[dict]:
    """
    Remove duplicate transcripts caused by storing:
      - asr_transcript event
      - transcript_saved event
      - active_call.utterances transcript
    """

    seen = set()
    unique = []

    for item in transcripts:
        key = (
            item.get("utterance_path"),
            item.get("transcript"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def save_asr_transcript_to_db(data: dict):
    """
    Save ASR transcript to database.

    This is defensive: if ASRJob/Transcript models are not ready yet,
    the monitor UI should still work.
    """

    session_id = data.get("session_id")
    transcript_text = data.get("transcript")
    confidence = data.get("confidence")

    if not session_id or not transcript_text:
        return

    try:
        from backend.models import ASRJob, Transcript
    except Exception as import_err:
        print(
            f"[DB SKIP] ASRJob/Transcript models not available: {import_err}",
            flush=True,
        )
        return

    db = SessionLocal()

    try:
        job = ASRJob(
            id=str(uuid.uuid4()),
            session_id=session_id,
            engine=data.get("engine"),
            status="completed",
        )

        db.add(job)
        db.flush()

        transcript_record = Transcript(
            id=str(uuid.uuid4()),
            asr_job_id=job.id,
            text=transcript_text,
            confidence=confidence,
        )

        db.add(transcript_record)
        db.commit()

        print(
            f"[DB SAVED] ASR transcript saved. "
            f"session={session_id}, job={job.id}",
            flush=True,
        )

    except Exception as db_err:
        db.rollback()
        print(f"[DB ERROR] Saving transcript failed: {db_err}", flush=True)

    finally:
        db.close()


async def safe_send_to_browser(browser_ws: WebSocket, payload: dict) -> bool:
    """
    Safely forward VAD/ASR event to browser websocket.
    """

    try:
        await browser_ws.send_json(payload)
        return True

    except Exception as exc:
        print(f"[BROWSER SEND FAILED] {exc}", flush=True)
        return False


# ============================================================
# Static Pages
# ============================================================

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/monitor")
def serve_monitor():
    return FileResponse(FRONTEND_DIR / "monitor.html")


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "phone-gateway",
        "call_recording_retention_days": CALL_RECORDING_RETENTION_DAYS,
    }


@app.on_event("startup")
async def _startup_tasks():
    # Start privacy retention loop.
    asyncio.create_task(_recording_retention_loop())


# ============================================================
# Monitor APIs
# ============================================================

@app.get("/api/monitor/state")
def api_monitor_state():
    """
    Main monitor endpoint.

    This guarantees the frontend receives:
      - active_call
      - events
      - asr_transcripts
      - recent_calls

    Important:
    asr_transcripts is built from BOTH:
      1. active_call.utterances that already contain transcript text
      2. monitor events with event_type="asr_transcript"
    """

    state = get_monitor_state()

    if state is None:
        state = {}

    elif not isinstance(state, dict):
        state = {
            "state": state,
        }

    else:
        state = dict(state)

    events = get_monitor_events()
    recent_calls = get_recent_calls()
    active_call = state.get("active_call")

    asr_transcripts = []
    asr_transcripts.extend(
        build_asr_transcripts_from_active_call(active_call)
    )
    asr_transcripts.extend(
        build_asr_transcripts_from_events(events)
    )

    asr_transcripts = dedupe_asr_transcripts(asr_transcripts)

    state["events"] = events
    state["asr_transcripts"] = asr_transcripts[:50]
    state["recent_calls"] = recent_calls

    return state


@app.get("/api/monitor/events")
def api_monitor_events():
    return {
        "events": get_monitor_events(),
    }


@app.get("/api/monitor/calls")
def api_monitor_calls():
    return {
        "calls": get_recent_calls(),
    }


# ============================================================
# Caller Registration
# ============================================================

@app.post("/api/callers/register")
def register_caller(payload: CallerRegisterRequest):
    caller = create_or_get_caller(
        full_name=payload.full_name,
        phone_number=payload.phone_number,
    )

    return {
        "message": "Caller registered",
        "caller": caller,
    }


# ============================================================
# VAD → Browser Event Forwarding
# ============================================================

async def forward_vad_events_to_browser(vad_ws, browser_ws: WebSocket):
    """
    Receive VAD/ASR events from vad-service.

    Responsibilities:
      1. Update monitor state
      2. Store utterance/transcript state
      3. Save ASR transcript to DB when available
      4. Forward useful event to caller browser websocket

    Important:
    ASR events are NOT stored as VAD status.
    This prevents the VAD step from showing "asr_transcript".
    """

    try:
        async for message in vad_ws:
            # Handle binary audio chunks from VAD service
            if isinstance(message, bytes):
                try:
                    await browser_ws.send_bytes(message)
                except Exception as exc:
                    print(f"[BROWSER BINARY SEND FAILED] {exc}", flush=True)
                    break
                continue

            # Handle JSON events
            try:
                data = json.loads(message)
            except Exception:
                data = {
                    "event": "vad_raw_message",
                    "message": message,
                }

            print("[VAD EVENT]", data, flush=True)

            event_name = data.get("event") or "vad_event"

            # ------------------------------------------------------------
            # VAD ready
            # update_vad_status() already stores an event.
            # ------------------------------------------------------------
            if event_name == "vad_ready":
                update_vad_status("vad_ready", data)

            # ------------------------------------------------------------
            # Speech started
            # ------------------------------------------------------------
            elif event_name == "speech_started":
                update_vad_status("speech_started", data)

            # ------------------------------------------------------------
            # Speech ended / utterance saved
            # ------------------------------------------------------------
            elif event_name == "speech_ended":
                update_vad_status("speech_ended", data)

                add_utterance(
                    utterance_path=data.get("utterance_path"),
                    duration_seconds=data.get("duration_seconds"),
                    speech_probability=data.get("speech_probability"),
                )

            # ------------------------------------------------------------
            # ASR started
            # ASR progress is stored as an event only,
            # not as vad_status.
            # ------------------------------------------------------------
            elif event_name == "asr_started":
                add_event("asr_started", data)

            # ------------------------------------------------------------
            # ASR transcript ready
            # Supports both event names:
            #   - asr_transcript
            #   - transcript_ready
            # ------------------------------------------------------------
            elif event_name in ("asr_transcript", "transcript_ready"):
                transcript_text = data.get("transcript")
                utterance_path = data.get("utterance_path")
                confidence = data.get("confidence")

                add_event("asr_transcript", data)

                update_utterance_transcript(
                    utterance_path,
                    transcript_text,
                    confidence,
                )

                save_asr_transcript_to_db(data)
            
            # ------------------------------------------------------------
            # RAG Answer ready
            # ------------------------------------------------------------
            elif event_name == "rag_answer":
                response_text = data.get("response") or data.get("answer")
                utterance_path = data.get("utterance_path")
                references = data.get("references")
                
                if response_text and utterance_path:
                    update_utterance_rag(utterance_path, response_text, references)

            # ------------------------------------------------------------
            # TTS audio ready
            # ------------------------------------------------------------
            elif event_name == "tts_ready":
                tts_url = data.get("tts_url") or data.get("audio_url")
                utterance_path = data.get("utterance_path")

                if tts_url and utterance_path:
                    update_utterance_tts(utterance_path, tts_url)

            elif event_name == "tts_started":
                add_event("tts_started", data)

            # ------------------------------------------------------------
            # ASR error
            # ------------------------------------------------------------
            elif event_name == "asr_error":
                add_event("asr_error", data)

            # ------------------------------------------------------------
            # Any unknown VAD event
            # ------------------------------------------------------------
            else:
                add_event(event_name, data)

            # ------------------------------------------------------------
            # Forward event to browser websocket
            # ------------------------------------------------------------
            sent = await safe_send_to_browser(browser_ws, data)

            if not sent:
                break

    except Exception as exc:
        print("[VAD EVENT FORWARDER CLOSED]", exc, flush=True)

        add_event("vad_forwarder_closed", {
            "error": str(exc),
        })


# ============================================================
# Browser Call WebSocket
# ============================================================

@app.websocket("/ws/call")
async def call_websocket(
    websocket: WebSocket,
    caller_id: str | None = Query(default=None),
    full_name: str | None = Query(default=None),
    phone_number: str | None = Query(default=None),
    audio_format: str = Query(default="pcm16"),
    sample_rate: int = Query(default=16000),
):
    await websocket.accept()

    session = create_session(
        caller_id=caller_id,
        full_name=full_name,
        phone_number=phone_number
    )
    session_id = session["session_id"]
    caller_id = session["caller_id"]

    caller_name, caller_phone = get_caller_details(caller_id)

    recorder = AudioRecorder(session_id=session_id)

    start_call_monitor(
        session_id=session_id,
        caller_id=caller_id,
        caller_name=caller_name,
        caller_phone=caller_phone,
        sample_rate=sample_rate,
        audio_format=audio_format,
    )

    # ── Connect to VAD Service ──
    query_params = {
        "session_id": session_id,
        "sample_rate": sample_rate,
        "phone_number": caller_phone
    }
    vad_url = f"{VAD_WS_BASE_URL}?" + "&".join([f"{k}={v}" for k, v in query_params.items()])

    vad_ws = None
    vad_event_task = None

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "caller_id": caller_id,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "message": "Call session started",
    })

    print(
        f"[CALL STARTED] session={session_id}, "
        f"caller={caller_id}, "
        f"audio_format={audio_format}, "
        f"sample_rate={sample_rate}",
        flush=True,
    )

    chunk_count = 0
    total_audio_bytes = 0

    try:
        vad_ws = await websockets.connect(vad_url)

        print(f"[VAD CONNECTED] {vad_url}", flush=True)

        vad_event_task = asyncio.create_task(
            forward_vad_events_to_browser(vad_ws, websocket)
        )

        while True:
            message = await websocket.receive()

            # ============================================================
            # Binary audio from browser
            # ============================================================

            if "bytes" in message and message["bytes"] is not None:
                audio_chunk = message["bytes"]

                if not audio_chunk:
                    continue

                chunk_count += 1
                total_audio_bytes += len(audio_chunk)

                audio_level = calculate_pcm16_audio_level(audio_chunk)

                update_audio_stats(
                    len(audio_chunk),
                    audio_level=audio_level,
                )

                if chunk_count % 20 == 0:
                    print(
                        f"[AUDIO CHUNKS] session={session_id}, "
                        f"chunks={chunk_count}, "
                        f"bytes={total_audio_bytes}, "
                        f"level={audio_level}",
                        flush=True,
                    )

                recorder.write_chunk(audio_chunk)

                if vad_ws:
                    await vad_ws.send(audio_chunk)

            # ============================================================
            # Text control message from browser
            # ============================================================

            elif "text" in message and message["text"] is not None:
                text = message["text"]

                if text == "END_CALL":
                    break

    except WebSocketDisconnect:
        print(f"[BROWSER DISCONNECTED] session={session_id}", flush=True)

        add_event("browser_disconnected", {
            "session_id": session_id,
        })

    except Exception as exc:
        print(f"[CALL ERROR] session={session_id}, error={exc}", flush=True)

        add_event("call_error", {
            "session_id": session_id,
            "error": str(exc),
        })

        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
            })
        except Exception:
            pass

    finally:
        # ------------------------------------------------------------
        # Tell VAD to finalize last utterance.
        # Do not close immediately before the VAD task has a chance
        # to receive final ASR events.
        # ------------------------------------------------------------

        if vad_ws:
            try:
                await vad_ws.send(json.dumps({
                    "event": "end_session",
                }))
            except Exception:
                pass

        # ------------------------------------------------------------
        # Wait briefly for VAD final events.
        # ------------------------------------------------------------

        if vad_event_task:
            try:
                await asyncio.wait_for(vad_event_task, timeout=5.0)

            except asyncio.TimeoutError:
                vad_event_task.cancel()

                try:
                    await vad_event_task
                except Exception:
                    pass

            except asyncio.CancelledError:
                pass

            except Exception:
                pass

        # ------------------------------------------------------------
        # Close VAD websocket after waiting.
        # ------------------------------------------------------------

        if vad_ws:
            try:
                await vad_ws.close()
            except Exception:
                pass

        # ------------------------------------------------------------
        # Close recorder and session.
        # ------------------------------------------------------------

        audio_file = recorder.close()
        ended_session = end_session(session_id, audio_file=audio_file)

        end_call_monitor(audio_file_path=audio_file)

        print(f"[CALL ENDED] {ended_session}", flush=True)

        try:
            await websocket.send_json({
                "type": "session_ended",
                "session": ended_session,
            })

        except Exception:
            pass