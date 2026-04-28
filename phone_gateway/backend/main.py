from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.sessions import create_session, end_session
from backend.recorder import AudioRecorder


app = FastAPI(title="Phone Browser Telephony Gateway")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws/call")
async def call_websocket(
    websocket: WebSocket,
    dialed_number: str = Query(...)
):
    await websocket.accept()

    session = create_session(dialed_number=dialed_number)
    session_id = session["session_id"]

    recorder = AudioRecorder(session_id=session_id)

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "dialed_number": dialed_number,
        "message": "Call session started"
    })

    print(f"[CALL STARTED] session={session_id}, number={dialed_number}")

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