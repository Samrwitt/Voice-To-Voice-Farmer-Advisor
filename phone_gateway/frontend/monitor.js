const MONITOR_ENDPOINT = "/api/monitor/state";
const POLL_MS = Math.max(250, Number.parseInt(String(window.MONITOR_POLL_MS || "650"), 10) || 650);

let _lastMicTarget = 0;
let _lastPlaybackTarget = 0;
let _displayMic = 0;
let _displayPlayback = 0;
let _rafWaveform = 0;

function $(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value ?? "—";
}

function formatBytes(bytes) {
  const n = Number(bytes || 0);

  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;

  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function basename(path) {
  if (!path) return "—";
  return String(path).split("/").pop();
}

function resetStatusBadgeClasses(el) {
  el.classList.remove("online", "offline", "idle", "pending", "ended", "error");
}

/**
 * Header badge: driven only by monitor API state (never a static “no call”).
 */
function updateCallStatusBadge(data) {
  const el = $("systemStatus");
  if (!el) return;

  const call = data && data.active_call;
  const hasSession = Boolean(call && call.session_id);
  const status = (call && call.status) || "";

  resetStatusBadgeClasses(el);

  if (!hasSession) {
    el.classList.add("idle");
    el.textContent = "No call in progress";
    return;
  }
  if (status === "ended") {
    el.classList.add("ended");
    el.textContent = "Last call ended";
    return;
  }
  el.classList.add("online");
  el.textContent = "Live call";
}

function setMonitorApiError() {
  const el = $("systemStatus");
  if (!el) return;
  resetStatusBadgeClasses(el);
  el.classList.add("error");
  el.textContent = "Monitor unavailable";
}

function setStep(id, state, text) {
  const el = $(id);
  if (!el) return;

  el.classList.remove("active", "success", "error");

  if (state) {
    el.classList.add(state);
  }

  const span = el.querySelector("span");
  if (span) span.textContent = text || "Waiting";
}

function effectiveMicLevelFromCall(call) {
  const c = call || {};
  const wf = c.waveform;
  const last = Number(c.audio_level) || 0;
  if (Array.isArray(wf) && wf.length) {
    const tail = wf.slice(-56).map((x) => Number(x) || 0);
    return Math.min(1, Math.max(last, ...tail, 0));
  }
  return Math.min(1, Math.max(0, last));
}

/**
 * Advisor-side energy hint from recent gateway events (TTS streaming).
 */
function playbackLevelFromEvents(events) {
  if (!events || !events.length) return 0;
  const now = Date.now();
  let peak = 0;
  for (const ev of events.slice(0, 40)) {
    const typ = String(ev.event_type || "");
    const t = Date.parse(ev.time || "");
    if (Number.isNaN(t) || now - t > 14000) continue;
    if (typ === "tts_started") peak = Math.max(peak, 0.58);
    else if (typ === "rag_answer" && now - t < 5000) peak = Math.max(peak, 0.12);
  }
  return Math.min(1, peak);
}

function scheduleWaveformDecayLoop() {
  if (_rafWaveform) return;
  const tick = () => {
    _rafWaveform = requestAnimationFrame(tick);
    // Ease displayed levels toward last server targets (smooth between polls).
    const micEase = 0.28;
    const pbEase = 0.22;
    _displayMic += (_lastMicTarget - _displayMic) * micEase;
    _displayPlayback += (_lastPlaybackTarget - _displayPlayback) * pbEase;
    if (_lastPlaybackTarget < 0.02) {
      _displayPlayback *= 0.88;
    }
    if (_lastMicTarget < 0.02) {
      _displayMic *= 0.9;
    }
    updateWaveform(_displayMic, _displayPlayback);
  };
  _rafWaveform = requestAnimationFrame(tick);
}

function updateWaveform(level, playbackLevel = 0) {
  const bars = document.querySelectorAll(".wave-bar");
  const combinedLevel = Math.max(Number(level || 0), Number(playbackLevel || 0));
  const normalized = Math.max(0, Math.min(1, combinedLevel));

  setText("audioLevelValue", `${Math.round(normalized * 100)}%`);

  bars.forEach((bar, index) => {
    const wave = Math.abs(Math.sin(index * 0.55 + Date.now() / 180));
    const height = 8 + normalized * 82 * wave;

    bar.style.height = `${height}%`;
    bar.style.opacity = `${0.25 + normalized * 0.75}`;

    if (playbackLevel > 0.05) {
      bar.style.backgroundColor = "#4facfe";
    } else {
      bar.style.backgroundColor = "#00f2fe";
    }
  });
}

async function loadMonitor() {
  try {
    const response = await fetch(MONITOR_ENDPOINT, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Monitor API failed: ${response.status}`);
    }

    const data = await response.json();

    renderMonitor(data);

  } catch (error) {
    console.error("Monitor load failed:", error);

    setMonitorApiError();
    setStep("stepGateway", "error", "Monitor API not reachable");
  }
}

function renderMonitor(data) {
  const call = data.active_call || {};
  const utterances = call.utterances || [];
  const transcripts = data.asr_transcripts || [];
  const recentCalls = data.recent_calls || [];

  const hasCall = Boolean(call.session_id);
  const hasActiveCall = Boolean(call.session_id && call.status !== "ended");
  const hasAudio = Number(call.audio_chunks || 0) > 0;
  const hasUtterance = utterances.length > 0;
  const hasTranscript = transcripts.length > 0;
  const hasRag = utterances.some(u => u.rag_response);
  const hasTts = utterances.some(u => u.tts_url);

  const vadStatus = call.vad_status || "waiting";

  const hasVad = Boolean(
    vadStatus === "vad_ready" ||
    vadStatus === "speech_started" ||
    vadStatus === "speech_ended" ||
    hasUtterance
  );

  updateCallStatusBadge(data);

  setText("lastUpdated", new Date().toLocaleTimeString());
  setText("sessionId", call.session_id || "—");
  setText("callerName", call.caller_name || "—");
  setText("callerPhone", call.caller_phone || "—");
  setText("callStatus", call.status || "waiting");
  setText("audioFormat", call.audio_format || "PCM16");
  setText("sampleRate", call.sample_rate || "16000");

  setText(
    "callStateText",
    hasActiveCall
      ? vadStatus === "speech_started"
        ? "Caller is speaking — streaming audio to VAD"
        : hasAudio
          ? "Call is live — listening and processing audio"
          : "Call connected — waiting for microphone audio"
      : hasCall
        ? "No active session — last call has ended"
        : "No call connected — open the voice client to start a session"
  );

  setText("audioChunks", call.audio_chunks || 0);
  setText("audioBytes", formatBytes(call.audio_bytes || 0));
  setText("vadStatus", vadStatus);
  setText("utteranceCount", call.utterance_count || utterances.length || 0);

  _lastMicTarget = effectiveMicLevelFromCall(call);
  _lastPlaybackTarget = playbackLevelFromEvents(data.events || []);

  renderUtterances(utterances);
  renderAsrTranscripts(transcripts);
  renderRecentCalls(recentCalls);

  renderPipeline({
    call,
    hasCall,
    hasActiveCall,
    hasAudio,
    hasVad,
    hasUtterance,
    hasTranscript,
    hasRag,
    hasTts,
    transcripts,
    utterances,
  });
}

function renderPipeline({
  call,
  hasCall,
  hasActiveCall,
  hasAudio,
  hasVad,
  hasUtterance,
  hasTranscript,
  transcripts,
  utterances,
}) {
  setStep(
    "stepGateway",
    hasActiveCall ? "success" : hasCall ? "active" : null,
    hasActiveCall ? "Session active" : hasCall ? "Call ended" : "Waiting"
  );

  setStep(
    "stepAudio",
    hasAudio ? "success" : null,
    hasAudio ? "PCM16 streaming" : "No audio"
  );

  setStep(
    "stepVad",
    hasVad ? "success" : null,
    hasVad ? "Speech detected" : "Waiting for speech"
  );

  setStep(
    "stepUtterance",
    hasUtterance ? "success" : null,
    hasUtterance ? `${utterances.length} saved` : "None yet"
  );

  setStep(
    "stepAsr",
    hasTranscript
      ? "success"
      : hasUtterance
        ? "active"
        : null,
    hasTranscript
      ? "Transcription complete"
      : hasUtterance
        ? "Transcribing / waiting for ASR"
        : "Waiting"
  );

  setStep(
    "stepTranscript",
    hasTranscript ? "success" : null,
    hasTranscript
      ? `${transcripts.length} transcript(s) ready`
      : "No transcript yet"
  );

  setStep(
    "stepRag",
    hasRag ? "success" : (hasTranscript ? "active" : null),
    hasRag ? "Answer generated" : (hasTranscript ? "Retrieving from KB..." : "Waiting")
  );

  setStep(
    "stepTts",
    hasTts ? "success" : (hasRag ? "active" : null),
    hasTts ? "Audio synthesized" : (hasRag ? "Synthesizing voice..." : "Waiting")
  );
}

function renderUtterances(utterances) {
  const container = $("utteranceList");
  if (!container) return;

  if (!utterances.length) {
    container.innerHTML = `<div class="empty">No utterance detected yet.</div>`;
    return;
  }

  container.innerHTML = utterances
    .slice(0, 20)
    .map((item) => {
      return `
        <div class="utterance-item">
          <div class="utterance-meta">
            <strong>Utterance ${item.index || ""}</strong>
            <span>${item.created_at || ""}</span>
          </div>

          <small>File: ${basename(item.utterance_path)}</small>
          <small>Duration: ${item.duration_seconds ?? "—"} seconds</small>

          ${
            item.transcript
              ? `<div class="utterance-transcript"><strong>ASR:</strong> ${item.transcript}</div>`
              : ""
          }

          ${
            item.rag_response
              ? `<div class="utterance-rag">
                   <strong>RAG Answer:</strong> ${item.rag_response}
                   ${
                     item.rag_references && item.rag_references.length > 0
                       ? `<div class="rag-sources">
                            <hr style="opacity:0.2; margin: 8px 0;">
                            <small>Sources:</small>
                            ${item.rag_references.map(ref => `
                              <div class="rag-source-item" style="font-size: 12px; margin-bottom: 4px;">
                                📄 <strong>${ref.title || "Untitled Document"}</strong>
                                <span style="opacity:0.7;">(dist: ${Number(ref.distance).toFixed(3)})</span>
                              </div>
                            `).join("")}
                          </div>`
                       : ""
                   }
                 </div>`
              : ""
          }

          ${
            item.tts_url
              ? `<div class="utterance-tts"><a href="${item.tts_url}" target="_blank">🔊 Play TTS Response</a></div>`
              : ""
          }
        </div>
      `;
    })
    .join("");
}

function renderAsrTranscripts(transcripts) {
  const container = $("asrTranscriptLog");

  if (!container) {
    console.error("Missing #asrTranscriptLog in monitor.html");
    return;
  }

  if (!transcripts.length) {
    container.innerHTML = `<div class="empty">No ASR transcripts yet.</div>`;
    return;
  }

  container.innerHTML = transcripts
    .slice(0, 20)
    .map((item) => {
      return `
        <div class="transcript-item">
          <div class="transcript-meta">
            <strong>ASR Transcript</strong>
            <span>${item.timestamp || ""}</span>
          </div>

          <div class="transcript-text">
            ${item.transcript || "—"}
          </div>

          <div class="transcript-extra">
            <small>Confidence: ${item.confidence ?? "—"}</small>
            <small>Engine: ${item.engine || "mock"}</small>
            <small>File: ${basename(item.utterance_path)}</small>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderRecentCalls(calls) {
  const container = $("recentCalls");
  if (!container) return;

  if (!calls.length) {
    container.innerHTML = `<div class="empty">No completed calls yet.</div>`;
    return;
  }

  container.innerHTML = calls
    .slice(0, 10)
    .map((call) => {
      return `
        <div class="recent-call-item">
          <strong>${call.session_id || "Unknown session"}</strong>
          <small>Status: ${call.status || "completed"}</small>
          <small>Audio: ${call.audio_file_path || "—"}</small>
        </div>
      `;
    })
    .join("");
}

scheduleWaveformDecayLoop();
loadMonitor();
setInterval(loadMonitor, POLL_MS);