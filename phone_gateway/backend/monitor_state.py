from datetime import datetime
from threading import Lock


_lock = Lock()

monitor_state = {
    "active_call": None,
    "events": [],
    "recent_calls": [],
}

MAX_EVENTS = 200
MAX_RECENT_CALLS = 50
MAX_WAVEFORM_LEVELS = 80


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def start_call_monitor(
    session_id: str,
    caller_id: str | None,
    caller_name: str | None = None,
    caller_phone: str | None = None,
    sample_rate: int = 16000,
    audio_format: str = "pcm16",
):
    with _lock:
        monitor_state["active_call"] = {
            "session_id": session_id,
            "caller_id": caller_id,
            "caller_name": caller_name,
            "caller_phone": caller_phone,
            "sample_rate": sample_rate,
            "audio_format": audio_format,
            "status": "connected",
            "vad_status": "waiting",
            "started_at": now_iso(),
            "ended_at": None,
            "audio_chunks": 0,
            "audio_bytes": 0,
            "audio_level": 0.0,
            "waveform": [],
            "utterance_count": 0,
            "utterances": [],
            "audio_file_path": None,
            "rag_response": None,
            "tts_url": None,
        }

    add_event("call_started", {
        "session_id": session_id,
        "caller_id": caller_id,
        "caller_name": caller_name,
        "caller_phone": caller_phone,
    })


def update_audio_stats(chunk_bytes: int, audio_level: float = 0.0):
    with _lock:
        active = monitor_state.get("active_call")
        if not active:
            return

        level = max(0.0, min(1.0, float(audio_level)))

        active["audio_chunks"] += 1
        active["audio_bytes"] += chunk_bytes
        active["audio_level"] = level
        active["status"] = "streaming_audio"

        waveform = active.get("waveform", [])
        waveform.append(level)

        if len(waveform) > MAX_WAVEFORM_LEVELS:
            waveform = waveform[-MAX_WAVEFORM_LEVELS:]

        active["waveform"] = waveform


def update_vad_status(status: str, extra: dict | None = None):
    with _lock:
        active = monitor_state.get("active_call")
        if active:
            active["vad_status"] = status

    add_event(status, extra or {})


def add_utterance(
    utterance_path: str | None,
    duration_seconds: float | None,
    speech_probability: float | None,
):
    utterance = None

    with _lock:
        active = monitor_state.get("active_call")
        if not active:
            return

        active["utterance_count"] += 1

        utterance = {
            "index": active["utterance_count"],
            "utterance_path": utterance_path,
            "duration_seconds": duration_seconds,
            "speech_probability": speech_probability,
            "created_at": now_iso(),
        }

        active["utterances"].insert(0, utterance)

    add_event("utterance_saved", utterance)


def update_utterance_transcript(utterance_path: str, transcript: str, confidence: float):
    utterance = None

    with _lock:
        active = monitor_state.get("active_call")
        if not active:
            return

        for u in active.get("utterances", []):
            if u.get("utterance_path") == utterance_path:
                u["transcript"] = transcript
                u["confidence"] = confidence
                utterance = u
                break

    if utterance:
        add_event("transcript_saved", utterance)


def update_utterance_rag(utterance_path: str, response: str, references: list | None = None):
    with _lock:
        active = monitor_state.get("active_call")
        if not active:
            return

        active["rag_response"] = response

        for u in active.get("utterances", []):
            if u.get("utterance_path") == utterance_path:
                u["rag_response"] = response
                u["rag_references"] = references
                break

    add_event("rag_answer", {
        "utterance_path": utterance_path,
        "response": response,
        "references": references,
    })


def update_utterance_tts(utterance_path: str, tts_url: str):
    with _lock:
        active = monitor_state.get("active_call")
        if not active:
            return

        active["tts_url"] = tts_url

        for u in active.get("utterances", []):
            if u.get("utterance_path") == utterance_path:
                u["tts_url"] = tts_url
                break

    add_event("tts_ready", {
        "utterance_path": utterance_path,
        "tts_url": tts_url,
    })


def end_call_monitor(audio_file_path: str | None = None):
    with _lock:
        active = monitor_state.get("active_call")

        if not active:
            return

        active["status"] = "ended"
        active["ended_at"] = now_iso()
        active["audio_file_path"] = audio_file_path

        monitor_state["recent_calls"].insert(0, active.copy())
        monitor_state["recent_calls"] = monitor_state["recent_calls"][:MAX_RECENT_CALLS]

    add_event("call_ended", {
        "session_id": active.get("session_id"),
        "audio_file_path": audio_file_path,
    })


def add_event(event_type: str, payload: dict | None = None):
    with _lock:
        event = {
            "time": now_iso(),
            "event_type": event_type,
            "payload": payload or {},
        }

        monitor_state["events"].insert(0, event)
        monitor_state["events"] = monitor_state["events"][:MAX_EVENTS]


def get_monitor_state():
    with _lock:
        active_call = monitor_state.get("active_call")
        events = list(monitor_state.get("events", []))[:80]
        recent_calls = list(monitor_state.get("recent_calls", []))[:20]

        asr_transcripts = []

        # Build ASR transcripts from active_call.utterances
        if active_call:
            for utterance in active_call.get("utterances", []):
                transcript = utterance.get("transcript")

                if transcript:
                    asr_transcripts.append({
                        "session_id": active_call.get("session_id"),
                        "utterance_path": utterance.get("utterance_path"),
                        "transcript": transcript,
                        "confidence": utterance.get("confidence"),
                        "engine": utterance.get("engine", "mock"),
                        "timestamp": utterance.get("created_at"),
                        "source": "active_call.utterances",
                    })

        # Also build from events just in case
        for event in events:
            event_type = event.get("event_type")
            payload = event.get("payload", {})

            if event_type in ("asr_transcript", "transcript_saved"):
                transcript = payload.get("transcript")

                if transcript:
                    asr_transcripts.append({
                        "session_id": payload.get("session_id") or (
                            active_call.get("session_id") if active_call else None
                        ),
                        "utterance_path": payload.get("utterance_path"),
                        "transcript": transcript,
                        "confidence": payload.get("confidence"),
                        "engine": payload.get("engine", "mock"),
                        "timestamp": event.get("time") or payload.get("created_at"),
                        "source": "events",
                    })

        # Remove duplicates
        seen = set()
        unique_asr_transcripts = []

        for item in asr_transcripts:
            key = (
                item.get("utterance_path"),
                item.get("transcript"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_asr_transcripts.append(item)

        return {
            "active_call": active_call,
            "events": events,
            "recent_calls": recent_calls,
            "asr_transcripts": unique_asr_transcripts[:50],
        }


def get_monitor_events():
    with _lock:
        return list(monitor_state.get("events", []))


def get_recent_calls():
    with _lock:
        return list(monitor_state.get("recent_calls", []))