import asyncio
import audioop
import json
import os
import tempfile
import uuid
import wave
from pathlib import Path
from urllib.parse import urlencode

import httpx
import websockets

from backend.monitor_state import (
    add_event,
    add_utterance,
    end_call_monitor,
    start_call_monitor,
    update_audio_stats,
    update_utterance_rag,
    update_utterance_transcript,
    update_utterance_tts,
    update_vad_status,
)
from backend.recorder import AudioRecorder
from backend.sessions import create_session, end_session
from backend.tts_chunking import chunk_tts_text


AUDIOSOCKET_KIND_HANGUP = 0x00
AUDIOSOCKET_KIND_UUID = 0x01
AUDIOSOCKET_KIND_DTMF = 0x03
AUDIOSOCKET_KIND_ERROR = 0xFF

AUDIOSOCKET_MEDIA_RATES = {
    0x10: 8000,
    0x11: 12000,
    0x12: 16000,
    0x13: 24000,
    0x14: 32000,
    0x15: 44100,
    0x16: 48000,
}

AUDIOSOCKET_RATE_KINDS = {
    sample_rate: kind
    for kind, sample_rate in AUDIOSOCKET_MEDIA_RATES.items()
}

DEFAULT_GREETING_AM = (
    "ሰላም ይሁንልዎ። እኔ የግብርና አማካሪ ነኝ። "
    "በምን ጉዳይ ልርዳዎት እንደምትፈልጉ ይንገሩኝ።"
)
DEFAULT_PROCESSING_ACK_AM = (
    "ጥያቄዎን ተቀብለናል። መልሱን እየዘጋጀን ነው።"
)

_server: asyncio.AbstractServer | None = None
_alert_server: asyncio.AbstractServer | None = None
_alert_call_payloads: dict[str, dict] = {}


def audiosocket_kind_for_sample_rate(sample_rate: int) -> int:
    return AUDIOSOCKET_RATE_KINDS.get(sample_rate, 0x12)


def parse_bool_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def parse_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


SIP_AUDIO_LOG_EVERY = parse_int_env("SIP_AUDIO_LOG_EVERY", 0)


def summarize_sip_event(payload: dict) -> dict:
    event = payload.get("event") or payload.get("type")
    summary = {"event": event}
    for key in ("session_id", "sample_rate", "utterance_path", "confidence", "message"):
        value = payload.get(key)
        if value is not None:
            summary[key] = value

    if payload.get("transcript"):
        summary["transcript"] = str(payload["transcript"])[:80]
    if payload.get("response") or payload.get("answer"):
        summary["response"] = str(payload.get("response") or payload.get("answer"))[:80]

    return summary


async def read_audiosocket_packet(
    reader: asyncio.StreamReader,
) -> tuple[int, bytes]:
    header = await reader.readexactly(3)
    packet_kind = header[0]
    payload_len = int.from_bytes(header[1:3], "big")
    payload = await reader.readexactly(payload_len) if payload_len else b""
    return packet_kind, payload


async def write_audiosocket_packet(
    writer: asyncio.StreamWriter,
    packet_kind: int,
    payload: bytes = b"",
) -> None:
    for offset in range(0, len(payload) or 1, 65535):
        chunk = payload[offset: offset + 65535]
        if not payload:
            chunk = b""
        writer.write(
            bytes([packet_kind])
            + len(chunk).to_bytes(2, "big")
            + chunk
        )
        await writer.drain()
        if not payload:
            break


def decode_call_uuid(payload: bytes) -> str:
    if len(payload) == 16:
        try:
            return str(uuid.UUID(bytes=payload))
        except ValueError:
            pass

    try:
        value = payload.decode("utf-8", errors="ignore").strip()
    except Exception:
        value = ""

    return value or str(uuid.uuid4())


def calculate_pcm16_audio_level(pcm_bytes: bytes) -> float:
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

        return round(min(1.0, ((total / len(samples)) ** 0.5) * 8), 4)
    except Exception:
        return 0.0


async def read_initial_audiosocket_media(
    reader: asyncio.StreamReader,
    peer,
) -> tuple[str, int, tuple[int, bytes] | None]:
    call_leg_id = str(uuid.uuid4())

    while True:
        packet_kind, payload = await asyncio.wait_for(
            read_audiosocket_packet(reader),
            timeout=10.0,
        )

        if packet_kind == AUDIOSOCKET_KIND_UUID:
            call_leg_id = decode_call_uuid(payload)
            continue

        if packet_kind in AUDIOSOCKET_MEDIA_RATES:
            sample_rate = AUDIOSOCKET_MEDIA_RATES[packet_kind]
            return call_leg_id, sample_rate, (packet_kind, payload)

        if packet_kind == AUDIOSOCKET_KIND_HANGUP:
            raise asyncio.IncompleteReadError(partial=b"", expected=0)

        if packet_kind == AUDIOSOCKET_KIND_ERROR:
            raise RuntimeError(f"AudioSocket error before media from {peer}: {payload.hex()}")

        print(
            f"[SIP UNKNOWN INITIAL PACKET] peer={peer}, kind=0x{packet_kind:02x}, "
            f"bytes={len(payload)}",
            flush=True,
        )


class AudioSocketPlaybackSink:
    def __init__(
        self,
        writer: asyncio.StreamWriter,
        default_sample_rate: int,
    ):
        self.writer = writer
        self.default_sample_rate = default_sample_rate
        self.write_lock = asyncio.Lock()

    async def send_json(self, payload: dict) -> None:
        # Monitor state is updated before this point; SIP callers only need audio.
        event = payload.get("event") or payload.get("type")
        if event and parse_bool_env("SIP_EVENT_LOGS", "1"):
            print(f"[SIP EVENT] {summarize_sip_event(payload)}", flush=True)

    async def send_bytes(self, payload: bytes) -> None:
        if not payload:
            return

        packet_kind = audiosocket_kind_for_sample_rate(self.default_sample_rate)
        async with self.write_lock:
            await write_audiosocket_packet(self.writer, packet_kind, payload)


async def safe_send_to_sink(sink: AudioSocketPlaybackSink, payload: dict | bytes) -> bool:
    try:
        if isinstance(payload, bytes):
            await sink.send_bytes(payload)
        else:
            await sink.send_json(payload)
        return True
    except Exception as exc:
        print(f"[SIP SEND FAILED] {exc}", flush=True)
        return False


class SipPlaybackState:
    """
    Tracks any currently running playback task (greeting / ack / answer) so we
    can implement barge-in by cancelling playback when the caller starts talking.
    """

    def __init__(self):
        self.playback_task: asyncio.Task | None = None
        self.playback_label: str | None = None


def _cancel_playback(state: SipPlaybackState, *, reason: str) -> None:
    task = state.playback_task
    if task and not task.done():
        print(f"[BARGE-IN] Cancelling playback ({state.playback_label}) due to {reason}", flush=True)
        task.cancel()


def _should_cancel_for_speech_started(state: SipPlaybackState) -> bool:
    """
    Echo during greeting often triggers speech_started even when caller has not
    spoken yet. By default, do not barge-in cancel greeting for that event.
    """
    cancel_greeting = parse_bool_env("SIP_BARGE_IN_CANCEL_GREETING", "0")
    if state.playback_label == "greeting" and not cancel_greeting:
        return False
    return True


def _start_playback_task(
    state: SipPlaybackState,
    label: str,
    coro,
) -> asyncio.Task:
    _cancel_playback(state, reason=f"starting_{label}")
    task = asyncio.create_task(coro)
    state.playback_task = task
    state.playback_label = label
    return task


async def forward_vad_events_to_sip(vad_ws, sink: AudioSocketPlaybackSink, state: SipPlaybackState):
    processing_ack_played_for: set[str] = set()
    try:
        async for message in vad_ws:
            if isinstance(message, bytes):
                sent = await safe_send_to_sink(sink, message)
                if not sent:
                    break
                continue

            try:
                data = json.loads(message)
            except Exception:
                data = {"event": "vad_raw_message", "message": message}

            event_name = data.get("event") or "vad_event"

            if event_name == "vad_ready":
                update_vad_status("vad_ready", data)
            elif event_name == "speech_started":
                update_vad_status("speech_started", data)
                # Barge-in: if the caller starts speaking, stop any ongoing playback
                if _should_cancel_for_speech_started(state):
                    _cancel_playback(state, reason="speech_started")
            elif event_name == "speech_ended":
                update_vad_status("speech_ended", data)
                add_utterance(
                    utterance_path=data.get("utterance_path"),
                    duration_seconds=data.get("duration_seconds"),
                    speech_probability=data.get("speech_probability"),
                )
                utterance_path = data.get("utterance_path")
                duration_seconds = data.get("duration_seconds")
                try:
                    duration_seconds = float(duration_seconds) if duration_seconds is not None else 0.0
                except (TypeError, ValueError):
                    duration_seconds = 0.0
                min_ack_seconds = float(os.getenv("SIP_PROCESSING_ACK_MIN_UTTERANCE_SEC", "1.2") or "1.2")

                if (
                    utterance_path
                    and utterance_path not in processing_ack_played_for
                    and duration_seconds >= min_ack_seconds
                ):
                    processing_ack_played_for.add(utterance_path)
                    # Cap memory for long calls.
                    if len(processing_ack_played_for) > 50:
                        processing_ack_played_for.clear()
                    _start_playback_task(
                        state,
                        "processing_ack",
                        play_processing_acknowledgement(sink),
                    )
            elif event_name in ("asr_transcript", "transcript_ready"):
                add_event("asr_transcript", data)
                update_utterance_transcript(
                    data.get("utterance_path"),
                    data.get("transcript"),
                    data.get("confidence"),
                )
            elif event_name == "rag_answer":
                add_event("rag_answer", data)
                response_text = data.get("response") or data.get("answer")
                utterance_path = data.get("utterance_path")
                if response_text and utterance_path:
                    update_utterance_rag(
                        utterance_path,
                        response_text,
                        data.get("references"),
                    )
            elif event_name == "tts_ready":
                add_event("tts_ready", data)
                tts_url = data.get("tts_url") or data.get("audio_url")
                utterance_path = data.get("utterance_path")
                if tts_url and utterance_path:
                    update_utterance_tts(utterance_path, tts_url)
            else:
                add_event(event_name, data)

            sent = await safe_send_to_sink(sink, data)
            if not sent:
                break
    except Exception as exc:
        print(f"[SIP VAD FORWARDER CLOSED] {exc}", flush=True)
        add_event("sip_vad_forwarder_closed", {"error": str(exc)})


async def stream_wav_to_audiosocket(
    sink: AudioSocketPlaybackSink,
    wav_bytes: bytes,
) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        with wave.open(tmp_path, "rb") as wf:
            source_sample_rate = wf.getframerate()
            target_sample_rate = sink.default_sample_rate
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            packet_kind = audiosocket_kind_for_sample_rate(target_sample_rate)
            frames_per_chunk = max(1, source_sample_rate // 50)
            rate_state = None

            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break

                if sample_width != 2:
                    frames = audioop.lin2lin(frames, sample_width, 2)

                if channels == 2:
                    frames = audioop.tomono(frames, 2, 0.5, 0.5)
                elif channels != 1:
                    raise ValueError(f"Unsupported TTS channel count: {channels}")

                if source_sample_rate != target_sample_rate:
                    frames, rate_state = audioop.ratecv(
                        frames,
                        2,
                        1,
                        source_sample_rate,
                        target_sample_rate,
                        rate_state,
                    )

                async with sink.write_lock:
                    await write_audiosocket_packet(sink.writer, packet_kind, frames)
                await asyncio.sleep(len(frames) / (2 * target_sample_rate))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


async def play_opening_greeting(sink: AudioSocketPlaybackSink) -> None:
    if not parse_bool_env("SIP_PLAY_OPENING_GREETING", "1"):
        return

    text = os.getenv("SIP_OPENING_GREETING_AM", DEFAULT_GREETING_AM).strip()
    if not text:
        return

    tts_url = os.getenv("TTS_SERVICE_URL", "http://tts-service:8009/synthesize").strip()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(tts_url, json={"text": text})
            response.raise_for_status()
        await stream_wav_to_audiosocket(sink, response.content)
    except Exception as exc:
        print(f"[SIP GREETING FAILED] {exc}", flush=True)
        add_event("sip_greeting_failed", {"error": str(exc)})


async def play_processing_acknowledgement(sink: AudioSocketPlaybackSink) -> None:
    if not parse_bool_env("SIP_PLAY_PROCESSING_ACK", "1"):
        return

    text = os.getenv("SIP_PROCESSING_ACK_AM", DEFAULT_PROCESSING_ACK_AM).strip()
    if not text:
        return

    try:
        await play_alert_message(sink, text)
    except Exception as exc:
        print(f"[SIP PROCESSING ACK FAILED] {exc}", flush=True)
        add_event("sip_processing_ack_failed", {"error": str(exc)})


def register_alert_call_payload(call_id: str, payload: dict) -> None:
    _alert_call_payloads[call_id] = payload


async def play_alert_message(sink: AudioSocketPlaybackSink, text: str) -> None:
    tts_url = os.getenv("TTS_SERVICE_URL", "http://tts-service:8009/synthesize").strip()
    print(f"[CALLBACK TTS] start chars={len((text or '').strip())}", flush=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        for chunk in chunk_tts_text(text):
            response = await client.post(tts_url, json={"text": chunk})
            response.raise_for_status()
            await stream_wav_to_audiosocket(sink, response.content)
            await asyncio.sleep(0.12)
    print("[CALLBACK TTS] complete", flush=True)


async def play_recorded_audio_file(sink: AudioSocketPlaybackSink, audio_path: str) -> None:
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Expert audio file not found: {audio_path}")
    print(
        f"[CALLBACK EXPERT AUDIO] start path={audio_path} size={path.stat().st_size}",
        flush=True,
    )
    await stream_wav_to_audiosocket(sink, path.read_bytes())
    print("[CALLBACK EXPERT AUDIO] complete", flush=True)


async def handle_alert_audiosocket_call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    peer = writer.get_extra_info("peername")
    try:
        call_id = str(uuid.uuid4())
        sample_rate = parse_int_env(
            "SIP_AUDIOSOCKET_SAMPLE_RATE",
            parse_int_env("SIP_ALERT_AUDIOSOCKET_SAMPLE_RATE", 8000),
        )
        packet_kind, payload = await asyncio.wait_for(read_audiosocket_packet(reader), timeout=10.0)
        if packet_kind == AUDIOSOCKET_KIND_UUID:
            call_id = decode_call_uuid(payload)
        elif packet_kind in AUDIOSOCKET_MEDIA_RATES:
            sample_rate = AUDIOSOCKET_MEDIA_RATES[packet_kind]
        elif packet_kind == AUDIOSOCKET_KIND_ERROR:
            raise RuntimeError(f"AudioSocket error before alert playback from {peer}: {payload.hex()}")
        elif packet_kind == AUDIOSOCKET_KIND_HANGUP:
            return

        payload = _alert_call_payloads.pop(call_id, {})
        message = (payload.get("alert_message") or "").strip()
        expert_audio_path = (payload.get("expert_audio_path") or "").strip()
        severity = (payload.get("severity") or "warning").strip()
        if severity == "critical":
            prefix = "አስቸኳይ ማስጠንቀቂያ። "
        elif severity == "expert_response":
            prefix = ""
        else:
            prefix = "የግብርና ማሳሰቢያ። "
        sink = AudioSocketPlaybackSink(writer, default_sample_rate=sample_rate)
        if expert_audio_path:
            intro = (
                message
                or os.getenv(
                    "SIP_EXPERT_RESPONSE_INTRO_AM",
                    "የባለሙያ መልስ ዝግጁ ነው። አሁን እናጫውታለን።",
                )
            )
            add_event("expert_callback_intro_start", {
                "call_id": call_id,
                "phone_number": payload.get("phone_number"),
                "escalation_id": payload.get("escalation_id"),
                "intro_chars": len(intro or ""),
            })
            if intro:
                await play_alert_message(sink, intro)
            add_event("expert_callback_intro_done", {
                "call_id": call_id,
                "phone_number": payload.get("phone_number"),
                "escalation_id": payload.get("escalation_id"),
            })
            add_event("expert_callback_audio_start", {
                "call_id": call_id,
                "phone_number": payload.get("phone_number"),
                "escalation_id": payload.get("escalation_id"),
                "expert_audio_path": expert_audio_path,
            })
            await play_recorded_audio_file(sink, expert_audio_path)
            add_event("expert_callback_audio_done", {
                "call_id": call_id,
                "phone_number": payload.get("phone_number"),
                "escalation_id": payload.get("escalation_id"),
            })
        elif severity == "expert_response":
            raise RuntimeError("Expert callback requires recorded expert audio.")
        elif message:
            await play_alert_message(sink, prefix + message)
        add_event("alert_call_played", {
            "call_id": call_id,
            "phone_number": payload.get("phone_number"),
            "target_region": payload.get("target_region"),
            "kind": payload.get("kind") or "alert",
        })
    except Exception as exc:
        print(f"[ALERT CALL ERROR] peer={peer}, error={exc}", flush=True)
        add_event("alert_call_error", {"peer": str(peer), "error": str(exc)})
    finally:
        try:
            await write_audiosocket_packet(writer, AUDIOSOCKET_KIND_HANGUP)
        except Exception:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def handle_sip_audiosocket_call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    peer = writer.get_extra_info("peername")
    try:
        call_leg_id, sample_rate, first_media_packet = await read_initial_audiosocket_media(
            reader,
            peer,
        )
    except asyncio.TimeoutError:
        print(f"[SIP CALL TIMEOUT] No AudioSocket UUID/media from {peer}", flush=True)
        return
    except asyncio.IncompleteReadError:
        return
    except Exception as exc:
        print(f"[SIP CALL START ERROR] peer={peer}, error={exc}", flush=True)
        return

    sip_caller_phone = (
        os.getenv("SIP_DEFAULT_CALLER_PHONE_NUMBER", "sip:farmeruhamayohannes").strip()
        or f"sip:{call_leg_id}"
    )
    sip_caller_name = os.getenv("SIP_DEFAULT_CALLER_NAME", "SIP Farmer").strip() or "SIP Farmer"

    session = create_session(full_name=sip_caller_name, phone_number=sip_caller_phone)
    session_id = session["session_id"]
    recorder = AudioRecorder(session_id=session_id, sample_rate=sample_rate)
    sink = AudioSocketPlaybackSink(writer, default_sample_rate=sample_rate)

    query_params = urlencode({
        "session_id": session_id,
        "sample_rate": sample_rate,
        "phone_number": sip_caller_phone,
    })
    vad_url = f"{os.getenv('VAD_WS_BASE_URL', 'ws://vad-service:8010/ws/vad')}?{query_params}"

    start_call_monitor(
        session_id=session_id,
        caller_id=session.get("caller_id"),
        caller_name=sip_caller_name,
        caller_phone=sip_caller_phone,
        sample_rate=sample_rate,
        audio_format="audiosocket/pcm16",
    )
    add_event("sip_call_started", {
        "session_id": session_id,
        "call_leg_id": call_leg_id,
        "caller_phone": sip_caller_phone,
        "peer": str(peer),
        "sample_rate": sample_rate,
    })

    vad_ws = None
    vad_event_task = None
    playback_state = SipPlaybackState()
    chunk_count = 0
    total_audio_bytes = 0

    try:
        vad_ws = await websockets.connect(vad_url)
        vad_event_task = asyncio.create_task(forward_vad_events_to_sip(vad_ws, sink, playback_state))
        _start_playback_task(playback_state, "greeting", play_opening_greeting(sink))

        if first_media_packet:
            _, payload = first_media_packet
            await vad_ws.send(payload)
            recorder.write_chunk(payload)

        while True:
            packet_kind, payload = await read_audiosocket_packet(reader)

            if packet_kind == AUDIOSOCKET_KIND_HANGUP:
                break

            if packet_kind == AUDIOSOCKET_KIND_DTMF:
                digit = payload.decode("utf-8", errors="ignore")
                add_event("sip_dtmf", {
                    "session_id": session_id,
                    "digit": digit,
                })
                continue

            if packet_kind == AUDIOSOCKET_KIND_ERROR:
                add_event("sip_audiosocket_error", {
                    "session_id": session_id,
                    "payload": payload.hex(),
                })
                break

            if packet_kind not in AUDIOSOCKET_MEDIA_RATES:
                add_event("sip_unknown_packet", {
                    "session_id": session_id,
                    "packet_kind": packet_kind,
                    "payload_length": len(payload),
                })
                continue

            if payload:
                chunk_count += 1
                total_audio_bytes += len(payload)
                recorder.write_chunk(payload)
                update_audio_stats(
                    len(payload),
                    audio_level=calculate_pcm16_audio_level(payload),
                )
                await vad_ws.send(payload)

                if SIP_AUDIO_LOG_EVERY > 0 and chunk_count % SIP_AUDIO_LOG_EVERY == 0:
                    print(
                        f"[SIP AUDIO] session={session_id}, chunks={chunk_count}, "
                        f"bytes={total_audio_bytes}",
                        flush=True,
                    )
    except asyncio.IncompleteReadError:
        add_event("sip_call_disconnected", {"session_id": session_id})
    except Exception as exc:
        print(f"[SIP CALL ERROR] session={session_id}, error={exc}", flush=True)
        add_event("sip_call_error", {
            "session_id": session_id,
            "error": str(exc),
        })
    finally:
        # Stop any playback still running
        _cancel_playback(playback_state, reason="call_end")
        if playback_state.playback_task:
            try:
                await playback_state.playback_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if vad_ws:
            try:
                await vad_ws.send(json.dumps({"event": "end_session"}))
            except Exception:
                pass

        if vad_event_task:
            try:
                await asyncio.wait_for(vad_event_task, timeout=5.0)
            except asyncio.TimeoutError:
                vad_event_task.cancel()
            except Exception:
                pass

        if vad_ws:
            try:
                await vad_ws.close()
            except Exception:
                pass

        audio_file = recorder.close()
        ended_session = end_session(session_id, audio_file=audio_file)
        end_call_monitor(audio_file_path=audio_file)

        add_event("sip_call_ended", {
            "session_id": session_id,
            "call_leg_id": call_leg_id,
            "duration_seconds": ended_session.get("duration_seconds"),
        })

        try:
            await write_audiosocket_packet(writer, AUDIOSOCKET_KIND_HANGUP)
        except Exception:
            pass

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def start_sip_audiosocket_server() -> asyncio.AbstractServer | None:
    global _server

    if not parse_bool_env("SIP_AUDIOSOCKET_ENABLED", "1"):
        print("[SIP] AudioSocket server disabled", flush=True)
        return None

    host = os.getenv("SIP_AUDIOSOCKET_HOST", "0.0.0.0")
    port = int(os.getenv("SIP_AUDIOSOCKET_PORT", "9092"))

    _server = await asyncio.start_server(
        handle_sip_audiosocket_call,
        host,
        port,
    )

    print(f"[SIP] AudioSocket server listening on {host}:{port}", flush=True)
    return _server


async def stop_sip_audiosocket_server() -> None:
    global _server

    if not _server:
        return

    _server.close()
    await _server.wait_closed()
    _server = None


async def start_alert_audiosocket_server() -> asyncio.AbstractServer | None:
    global _alert_server

    if not parse_bool_env("SIP_ALERT_AUDIOSOCKET_ENABLED", "1"):
        print("[SIP] Alert AudioSocket server disabled", flush=True)
        return None

    host = os.getenv("SIP_AUDIOSOCKET_HOST", "0.0.0.0")
    port = int(os.getenv("SIP_ALERT_AUDIOSOCKET_PORT", "9093"))
    _alert_server = await asyncio.start_server(handle_alert_audiosocket_call, host, port)
    print(f"[SIP] Alert AudioSocket server listening on {host}:{port}", flush=True)
    return _alert_server


async def stop_alert_audiosocket_server() -> None:
    global _alert_server

    if not _alert_server:
        return
    _alert_server.close()
    await _alert_server.wait_closed()
    _alert_server = None
