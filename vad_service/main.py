import json
import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from vad_engine import SileroStreamingVAD
from asr_client import transcribe_utterance_file


app = FastAPI(title="Silero VAD Service")


# ============================================================
# Configuration
# ============================================================

MAX_CONCURRENT_ASR = int(os.getenv("MAX_CONCURRENT_ASR", "2"))
ASR_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ASR)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "silero-vad-service",
        "max_concurrent_asr": MAX_CONCURRENT_ASR,
    }


# ============================================================
# Safe WebSocket Send
# ============================================================

async def safe_send_json(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    payload: dict,
) -> bool:
    """
    Safely send JSON to the WebSocket.

    Multiple coroutines may send to the same WebSocket:
      1. the main VAD loop
      2. background ASR tasks

    The lock prevents simultaneous websocket.send_json() calls.

    Returns:
      True  -> message sent
      False -> client disconnected / socket closed
    """

    try:
        async with send_lock:
            await websocket.send_json(payload)
        return True

    except WebSocketDisconnect:
        return False

    except RuntimeError:
        return False

    except Exception as e:
        print(f"[WS SEND ERROR] {e}", flush=True)
        return False


# ============================================================
# ASR Handling
# ============================================================

async def handle_completed_utterance(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    utterance_path: str,
):
    """
    Runs ASR for one completed utterance.

    This is started with asyncio.create_task(...), so it does not block
    the VAD loop. VAD can continue listening while ASR runs.
    """

    try:
        sent = await safe_send_json(
            websocket,
            send_lock,
            {
                "event": "asr_started",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "message": "ASR is transcribing the saved utterance",
            },
        )

        if not sent:
            print(
                f"[ASR SKIPPED] WebSocket closed before ASR start event. "
                f"session={session_id}, file={utterance_path}",
                flush=True,
            )
            return

        print(
            f"[ASR STARTED] session={session_id}, file={utterance_path}",
            flush=True,
        )

        async with ASR_SEMAPHORE:
            asr_result = await transcribe_utterance_file(
                utterance_path=utterance_path,
                language="am",
            )

        await safe_send_json(
            websocket,
            send_lock,
            {
                "event": "asr_transcript",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "transcript": asr_result.get("transcript"),
                "confidence": asr_result.get("confidence"),
                "engine": asr_result.get("engine"),
                "audio_id": asr_result.get("audio_id"),
                "message": "ASR transcription completed",
            },
        )

        print(
            f"[ASR DONE] session={session_id}, "
            f"file={utterance_path}, "
            f"engine={asr_result.get('engine')}, "
            f"confidence={asr_result.get('confidence')}, "
            f"transcript={asr_result.get('transcript')}",
            flush=True,
        )

    except Exception as e:
        await safe_send_json(
            websocket,
            send_lock,
            {
                "event": "asr_error",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "error": str(e),
                "message": "ASR transcription failed",
            },
        )

        print(
            f"[ASR ERROR] session={session_id}, "
            f"file={utterance_path}, "
            f"error={e}",
            flush=True,
        )


def start_asr_task(
    active_asr_tasks: set,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    utterance_path: str,
):
    """
    Start ASR in the background.

    The task is stored in active_asr_tasks so it remains referenced until
    it completes.
    """

    task = asyncio.create_task(
        handle_completed_utterance(
            websocket=websocket,
            send_lock=send_lock,
            session_id=session_id,
            utterance_path=utterance_path,
        )
    )

    active_asr_tasks.add(task)

    def cleanup_task(done_task: asyncio.Task):
        active_asr_tasks.discard(done_task)

        try:
            exc = done_task.exception()
            if exc:
                print(
                    f"[ASR TASK ERROR] session={session_id}, error={exc}",
                    flush=True,
                )
        except asyncio.CancelledError:
            pass

    task.add_done_callback(cleanup_task)


# ============================================================
# VAD WebSocket Endpoint
# ============================================================

@app.websocket("/ws/vad")
async def vad_websocket(
    websocket: WebSocket,
    session_id: str = Query(...),
    sample_rate: int = Query(default=16000),
):
    await websocket.accept()

    send_lock = asyncio.Lock()
    active_asr_tasks = set()

    output_dir = os.getenv("VAD_UTTERANCES_DIR", "utterances")

    vad = SileroStreamingVAD(
        session_id=session_id,
        sample_rate=sample_rate,
        threshold=0.5,
        min_speech_start_ms=120,
        speech_end_silence_ms=900,
        speech_pad_ms=200,
        output_dir=output_dir,
    )

    await safe_send_json(
        websocket,
        send_lock,
        {
            "event": "vad_ready",
            "session_id": session_id,
            "sample_rate": sample_rate,
            "message": "Silero VAD is ready",
        },
    )

    print(
        f"[VAD READY] session={session_id}, "
        f"sample_rate={sample_rate}, "
        f"output_dir={output_dir}",
        flush=True,
    )

    chunk_count = 0
    total_audio_bytes = 0

    try:
        while True:
            message = await websocket.receive()

            # ============================================================
            # Binary audio message
            # ============================================================

            if "bytes" in message and message["bytes"] is not None:
                pcm_chunk = message["bytes"]

                if pcm_chunk:
                    chunk_count += 1
                    total_audio_bytes += len(pcm_chunk)

                    if chunk_count % 20 == 0:
                        print(
                            f"[VAD AUDIO RECEIVED] session={session_id}, "
                            f"chunks={chunk_count}, "
                            f"bytes={total_audio_bytes}",
                            flush=True,
                        )

                events = vad.process_pcm_chunk(pcm_chunk)

                for event in events:
                    event_name = event.get("event")
                    utterance_path = event.get("utterance_path")

                    await safe_send_json(
                        websocket,
                        send_lock,
                        {
                            "event": event_name,
                            "session_id": session_id,
                            "timestamp": event.get("timestamp"),
                            "utterance_path": utterance_path,
                            "duration_seconds": event.get("duration_seconds"),
                            "speech_probability": event.get("speech_probability"),
                        },
                    )

                    if event_name == "speech_started":
                        print(
                            f"[SPEECH STARTED] session={session_id}",
                            flush=True,
                        )

                    elif event_name == "speech_ended":
                        print(
                            f"[SPEECH ENDED] session={session_id}, "
                            f"file={utterance_path}",
                            flush=True,
                        )

                        if utterance_path:
                            start_asr_task(
                                active_asr_tasks=active_asr_tasks,
                                websocket=websocket,
                                send_lock=send_lock,
                                session_id=session_id,
                                utterance_path=utterance_path,
                            )

            # ============================================================
            # Text control message
            # ============================================================

            elif "text" in message and message["text"] is not None:
                text = message["text"]

                try:
                    data = json.loads(text)

                    if data.get("event") == "reset":
                        vad.reset()

                        await safe_send_json(
                            websocket,
                            send_lock,
                            {
                                "event": "reset_done",
                                "session_id": session_id,
                            },
                        )

                    elif data.get("event") == "end_session":
                        utterance_path = vad.finalize()

                        if utterance_path:
                            await safe_send_json(
                                websocket,
                                send_lock,
                                {
                                    "event": "speech_ended",
                                    "session_id": session_id,
                                    "utterance_path": utterance_path,
                                    "duration_seconds": None,
                                    "message": "Final utterance saved on session end",
                                },
                            )

                            print(
                                f"[SPEECH ENDED ON CLOSE] "
                                f"session={session_id}, "
                                f"file={utterance_path}",
                                flush=True,
                            )

                            start_asr_task(
                                active_asr_tasks=active_asr_tasks,
                                websocket=websocket,
                                send_lock=send_lock,
                                session_id=session_id,
                                utterance_path=utterance_path,
                            )

                        if active_asr_tasks:
                            await asyncio.gather(
                                *active_asr_tasks,
                                return_exceptions=True,
                            )

                        break

                except json.JSONDecodeError:
                    if text == "END":
                        utterance_path = vad.finalize()

                        if utterance_path:
                            await safe_send_json(
                                websocket,
                                send_lock,
                                {
                                    "event": "speech_ended",
                                    "session_id": session_id,
                                    "utterance_path": utterance_path,
                                    "duration_seconds": None,
                                    "message": "Final utterance saved on END message",
                                },
                            )

                            print(
                                f"[SPEECH ENDED ON END] "
                                f"session={session_id}, "
                                f"file={utterance_path}",
                                flush=True,
                            )

                            start_asr_task(
                                active_asr_tasks=active_asr_tasks,
                                websocket=websocket,
                                send_lock=send_lock,
                                session_id=session_id,
                                utterance_path=utterance_path,
                            )

                        if active_asr_tasks:
                            await asyncio.gather(
                                *active_asr_tasks,
                                return_exceptions=True,
                            )

                        break

    except WebSocketDisconnect:
        print(f"[VAD DISCONNECTED] session={session_id}", flush=True)

    finally:
        for task in list(active_asr_tasks):
            if not task.done():
                task.cancel()

        if active_asr_tasks:
            await asyncio.gather(
                *active_asr_tasks,
                return_exceptions=True,
            )

        try:
            await websocket.close()
        except Exception:
            pass

        print(f"[VAD CLOSED] session={session_id}", flush=True)