import asyncio
import json
import os
from pathlib import Path

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.database import Base, engine
from backend.models import CallSession, Caller
from backend.sessions import create_session, end_session
from backend.recorder import AudioRecorder
from backend.callers import create_or_get_caller


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Phone Browser Telephony Gateway")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


VAD_WS_BASE_URL = os.getenv(
    "VAD_WS_BASE_URL",
    "ws://vad-service:8010/ws/vad"
)


class CallerRegisterRequest(BaseModel):
    full_name: str
    phone_number: str


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "phone-gateway"}


@app.post("/api/callers/register")
def register_caller(payload: CallerRegisterRequest):
    caller = create_or_get_caller(
        full_name=payload.full_name,
        phone_number=payload.phone_number
    )

    return {
        "message": "Caller registered",
        "caller": caller
    }


async def forward_vad_events_to_browser(vad_ws, browser_ws: WebSocket):
    """
    Receive VAD events from vad-service and forward them to the browser.
    Example events:
    - vad_ready
    - speech_started
    - speech_ended
    """

    try:
        async for message in vad_ws:
            try:
                data = json.loads(message)
            except Exception:
                data = {
                    "event": "vad_raw_message",
                    "message": message
                }

            print("[VAD EVENT]", data)

            try:
                await browser_ws.send_json(data)
            except Exception:
                break

    except Exception as exc:
        print("[VAD EVENT FORWARDER CLOSED]", exc)


@app.websocket("/ws/call")
async def call_websocket(
    websocket: WebSocket,
    caller_id: str | None = Query(default=None),
    audio_format: str = Query(default="pcm16"),
    sample_rate: int = Query(default=16000),
):
    await websocket.accept()

    session = create_session(caller_id=caller_id)
    session_id = session["session_id"]

    recorder = AudioRecorder(session_id=session_id)

    vad_url = (
        f"{VAD_WS_BASE_URL}"
        f"?session_id={session_id}"
        f"&sample_rate={sample_rate}"
    )

    vad_ws = None
    vad_event_task = None

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "caller_id": caller_id,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "message": "Call session started"
    })

    print(
        f"[CALL STARTED] session={session_id}, "
        f"caller={caller_id}, "
        f"audio_format={audio_format}, "
        f"sample_rate={sample_rate}"
    )

    chunk_count = 0
    total_audio_bytes = 0

    try:
        vad_ws = await websockets.connect(vad_url)
        print(f"[VAD CONNECTED] {vad_url}")

        vad_event_task = asyncio.create_task(
            forward_vad_events_to_browser(vad_ws, websocket)
        )

        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_chunk = message["bytes"]

                if audio_chunk:
                    chunk_count += 1
                    total_audio_bytes += len(audio_chunk)

                    if chunk_count % 20 == 0:
                        print(
                            f"[AUDIO CHUNKS] session={session_id}, "
                            f"chunks={chunk_count}, "
                            f"bytes={total_audio_bytes}",
                            flush=True
                        )

                    # Save full call audio locally
                    recorder.write_chunk(audio_chunk)

                    # Forward same PCM16 chunk to VAD service
                    if vad_ws:
                        await vad_ws.send(audio_chunk)

            elif "text" in message:
                text = message["text"]

                if text == "END_CALL":
                    break

    except WebSocketDisconnect:
        print(f"[BROWSER DISCONNECTED] session={session_id}")

    except Exception as exc:
        print(f"[CALL ERROR] session={session_id}, error={exc}")

        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc)
            })
        except Exception:
            pass

    finally:
        if vad_ws:
            try:
                await vad_ws.send(json.dumps({"event": "end_session"}))
                await vad_ws.close()
            except Exception:
                pass

        if vad_event_task:
            vad_event_task.cancel()

        audio_file = recorder.close()
        ended_session = end_session(session_id, audio_file=audio_file)

        print(f"[CALL ENDED] {ended_session}")

        try:
            await websocket.send_json({
                "type": "session_ended",
                "session": ended_session
            })
        except Exception:
            pass