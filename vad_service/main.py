import json
import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from vad_engine import SileroStreamingVAD
from asr_client import transcribe_utterance_file
from rag_client import get_rag_answer
from tts_client import synthesize_speech


app = FastAPI(title="Silero VAD Service")


# ============================================================
# Configuration
# ============================================================

MAX_CONCURRENT_ASR = int(os.getenv("MAX_CONCURRENT_ASR", "2"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.85"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "400"))
ASR_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ASR)
PLAYBACK_LOCK = asyncio.Lock()


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

async def safe_send(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    payload: dict | bytes,
) -> bool:
    """Safely send JSON or binary to the WebSocket."""
    try:
        async with send_lock:
            if isinstance(payload, bytes):
                await websocket.send_bytes(payload)
            else:
                await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        print(f"[WS SEND ERROR] {e}", flush=True)
        return False


class SessionState:
    def __init__(self):
        self.playback_task = None

# ============================================================
# Playback Handling
# ============================================================

async def play_advisor_response(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    utterance_path: str,
    rag_answer: str,
):
    """
    Synthesizes and streams audio for a RAG answer.
    Locked by PLAYBACK_LOCK to avoid overlapping speech.
    """
    async with PLAYBACK_LOCK:
        try:
            print(f"[TTS STARTING] text_len={len(rag_answer)}, path={utterance_path}", flush=True)
            # ── Sentence-level Streaming ──
            import re
            # Split by Amharic and common sentence delimiters
            sentences = re.split(r'(?<=[።?!])\s+', rag_answer.strip())
            
            print(f"[STREAMING] Total sentences: {len(sentences)}", flush=True)
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                print(f"[SENTENCE {i+1}/{len(sentences)}] Synthesizing: {sentence[:30]}...", flush=True)
                
                # Use a unique path for each sentence to avoid collisions
                sentence_path = f"{utterance_path}_s{i}.wav"
                
                tts_path = await synthesize_speech(
                    text=sentence,
                    utterance_path=sentence_path
                )
                
                if tts_path:
                    try:
                        import wave
                        with wave.open(tts_path, "rb") as wf:
                            # Stream raw frames
                            chunk_size = 1024
                            data = wf.readframes(chunk_size)
                            while data:
                                sent = await safe_send(websocket, send_lock, data)
                                if not sent:
                                    break
                                
                                # Small delay to prevent network congestion
                                await asyncio.sleep(0.01)
                                data = wf.readframes(chunk_size)
                                
                        print(f"[SENTENCE {i+1} DONE] Streamed.", flush=True)
                    except Exception as e:
                        print(f"[SENTENCE {i+1} ERROR] {e}", flush=True)
                
                # Optional: Add a natural pause between sentences
                await asyncio.sleep(0.2)

            print(f"[TTS STREAM DONE] session={session_id}", flush=True)
        except asyncio.CancelledError:
            print(f"[PLAYBACK CANCELLED] session={session_id}", flush=True)
            raise
        except Exception as e:
            print(f"[PLAYBACK ERROR] session={session_id}, error={e}", flush=True)

# ============================================================
# ASR / RAG Handling
# ============================================================

async def handle_completed_utterance(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    utterance_path: str,
    session_state: SessionState,
    phone_number: str = "Unknown",
):
    """
    Runs ASR for one completed utterance.

    This is started with asyncio.create_task(...), so it does not block
    the VAD loop. VAD can continue listening while ASR runs.
    """

    try:
        sent = await safe_send(
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

        await safe_send(
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

        # ── RAG Step ────────────────────────────────────────────────────────
        transcript = asr_result.get("transcript")
        if not transcript:
            return

        await safe_send(
            websocket,
            send_lock,
            {
                "event": "rag_started",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "message": "Retrieving answer from Knowledge Base...",
            },
        )

        rag_result = await get_rag_answer(
            text=transcript,
            session_id=session_id,
            phone_number=phone_number
        )

        rag_answer = rag_result.get("response")

        await safe_send(
            websocket,
            send_lock,
            {
                "event": "rag_answer",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "response": rag_answer,
                "references": rag_result.get("references"),
                "message": "RAG answer generated",
            },
        )

        print(
            f"[RAG DONE] session={session_id}, "
            f"file={utterance_path}, "
            f"response={rag_answer[:50]}...",
            flush=True,
        )

        # ── TTS Step ─────────────────────────────────────────────────────────
        if not rag_answer:
            return

        await safe_send(
            websocket,
            send_lock,
            {
                "event": "tts_started",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "message": "Synthesizing voice response...",
            },
        )

        print(f"[TTS STARTING] text_len={len(rag_answer)}, path={utterance_path}", flush=True)
        # ── Start Playback in background ──
        # We store it in session_state so the VAD loop can cancel it if barge-in occurs.
        session_state.playback_task = asyncio.create_task(
            play_advisor_response(
                websocket=websocket,
                send_lock=send_lock,
                session_id=session_id,
                utterance_path=utterance_path,
                rag_answer=rag_answer,
            )
        )

    except Exception as e:
        import traceback
        print(f"[FATAL ERROR in handle_completed_utterance]: {e}", flush=True)
        traceback.print_exc()
        
        await safe_send(
            websocket,
            send_lock,
            {
                "event": "asr_error",
                "session_id": session_id,
                "utterance_path": utterance_path,
                "error": str(e),
                "message": "Processing failed",
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
    session_state: SessionState,
    phone_number: str = "Unknown",
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
            session_state=session_state,
            phone_number=phone_number,
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
    phone_number: str = Query(default="Unknown"),
):
    await websocket.accept()

    send_lock = asyncio.Lock()
    active_asr_tasks = set()
    session_state = SessionState()

    output_dir = os.getenv("VAD_UTTERANCES_DIR", "utterances")

    vad = SileroStreamingVAD(
        session_id=session_id,
        sample_rate=sample_rate,
        threshold=VAD_THRESHOLD,
        min_speech_start_ms=VAD_MIN_SPEECH_MS,
        speech_end_silence_ms=900,
        speech_pad_ms=200,
        output_dir=output_dir,
    )

    await safe_send(
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

                    await safe_send(
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
                        # Barge-in: Cancel current advisor playback if they were talking
                        if session_state.playback_task and not session_state.playback_task.done():
                            print(f"[BARGE-IN] Interrupting advisor playback", flush=True)
                            session_state.playback_task.cancel()

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
                                session_state=session_state,
                                phone_number=phone_number,
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

                        await safe_send(
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
                            await safe_send(
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
                                session_state=session_state,
                                phone_number=phone_number,
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
                            await safe_send(
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
                                session_state=session_state,
                                phone_number=phone_number,
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