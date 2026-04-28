from pathlib import Path

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


class CallerRegisterRequest(BaseModel):
    full_name: str
    phone_number: str


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


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


@app.websocket("/ws/call")
async def call_websocket(
    websocket: WebSocket,
    caller_id: str | None = Query(default=None)
):
    await websocket.accept()

    session = create_session(caller_id=caller_id)
    session_id = session["session_id"]

    recorder = AudioRecorder(session_id=session_id)

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "caller_id": caller_id,
        "message": "Call session started"
    })

    print(f"[CALL STARTED] session={session_id}, caller={caller_id}")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                recorder.write_chunk(message["bytes"])

            elif "text" in message:
                if message["text"] == "END_CALL":
                    break

    except WebSocketDisconnect:
        print(f"[DISCONNECTED] session={session_id}")

    finally:
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