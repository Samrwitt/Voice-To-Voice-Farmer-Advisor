from datetime import datetime
from pathlib import Path
import json
import uuid


SESSIONS = {}


def create_session(dialed_number: str) -> dict:
    session_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    session = {
        "session_id": session_id,
        "dialed_number": dialed_number,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "audio_file": None,
        "status": "active",
    }

    SESSIONS[session_id] = session
    return session


def end_session(session_id: str, audio_file: str | None = None) -> dict:
    session = SESSIONS.get(session_id)

    if not session:
        raise ValueError("Session not found")

    end_time = datetime.utcnow()
    start_time = datetime.fromisoformat(session["start_time"])

    session["end_time"] = end_time.isoformat()
    session["duration_seconds"] = round((end_time - start_time).total_seconds(), 2)
    session["audio_file"] = audio_file
    session["status"] = "ended"

    save_session_metadata(session)

    return session


def save_session_metadata(session: dict):
    metadata_dir = Path("recordings/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / f"{session['session_id']}.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)