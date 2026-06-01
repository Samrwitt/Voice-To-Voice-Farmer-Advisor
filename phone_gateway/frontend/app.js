let websocket = null;
let timerInterval = null;
let seconds = 0;
let activeCall = false;

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let captureSinkNode = null;
let playbackStartTime = 0;

const DEFAULT_SERVICE_NUMBER = "8028";
const TARGET_SAMPLE_RATE = 16000;

function createAppAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  try {
    return new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
  } catch (e) {
    return new AudioContextClass();
  }
}

/** Played once per session when the server confirms the call (Amharic). */
const CALL_GREETING_AM =
  "ሰላም ይሁንልዎ። እኔ የግብርና አማካሪ ነኝ። በምን ጉዳይ ልርዳዎት እንደምትፈልጉ ይንገሩኝ።";

let greetingPlayedSessionId = null;

let callerId = localStorage.getItem("caller_id");
let callerName = localStorage.getItem("caller_name");
let callerPhone = localStorage.getItem("caller_phone");

const callerForm = document.getElementById("callerForm");
const dialer = document.getElementById("dialer");
const callerInfo = document.getElementById("callerInfo");

const numberDisplay = document.getElementById("numberDisplay");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const sessionInfo = document.getElementById("sessionInfo");

document.addEventListener("DOMContentLoaded", () => {
  if (numberDisplay && !numberDisplay.value) {
    numberDisplay.value = DEFAULT_SERVICE_NUMBER;
  }

  if (callerId && callerName && callerPhone) {
    showDialer();
  } else {
    showCallerForm();
  }
});

function showCallerForm() {
  if (!callerForm || !dialer) return;

  callerForm.classList.remove("hidden");
  dialer.classList.add("hidden");

  callerForm.style.display = "flex";
  dialer.style.display = "none";
}

function showDialer() {
  if (!callerForm || !dialer) return;

  callerForm.classList.add("hidden");
  dialer.classList.remove("hidden");

  callerForm.style.display = "none";
  dialer.style.display = "flex";

  if (callerInfo) {
    callerInfo.innerText = `${callerName} | ${callerPhone}`;
  }

  if (numberDisplay && !numberDisplay.value) {
    numberDisplay.value = DEFAULT_SERVICE_NUMBER;
  }

  setStatus("Idle");
}

async function registerCaller() {
  const fullNameInput = document.getElementById("fullName");
  const phoneNumberInput = document.getElementById("phoneNumber");

  const fullName = fullNameInput ? fullNameInput.value.trim() : "";
  const phoneNumber = phoneNumberInput ? phoneNumberInput.value.trim() : "";

  if (!fullName || !phoneNumber) {
    alert("Please enter full name and phone number.");
    return;
  }

  try {
    const continueButton = document.querySelector(".continue-btn");

    if (continueButton) {
      continueButton.disabled = true;
      continueButton.innerText = "Please wait...";
    }

    const response = await fetch("/api/callers/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        full_name: fullName,
        phone_number: phoneNumber
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Registration failed:", errorText);
      alert("Failed to register caller.");
      return;
    }

    const data = await response.json();

    if (!data.caller || !data.caller.caller_id) {
      console.error("Unexpected register response:", data);
      alert("Registration response is invalid.");
      return;
    }

    callerId = data.caller.caller_id;
    callerName = data.caller.full_name;
    callerPhone = data.caller.phone_number;

    localStorage.setItem("caller_id", callerId);
    localStorage.setItem("caller_name", callerName);
    localStorage.setItem("caller_phone", callerPhone);

    showDialer();

  } catch (error) {
    console.error("Could not connect to the server:", error);
    alert("Could not connect to the server.");
  } finally {
    const continueButton = document.querySelector(".continue-btn");

    if (continueButton) {
      continueButton.disabled = false;
      continueButton.innerText = "Continue";
    }
  }
}

function clearCaller() {
  if (activeCall) {
    alert("End the current call before changing caller.");
    return;
  }

  localStorage.removeItem("caller_id");
  localStorage.removeItem("caller_name");
  localStorage.removeItem("caller_phone");

  callerId = null;
  callerName = null;
  callerPhone = null;

  const fullNameInput = document.getElementById("fullName");
  const phoneNumberInput = document.getElementById("phoneNumber");

  if (fullNameInput) fullNameInput.value = "";
  if (phoneNumberInput) phoneNumberInput.value = "";

  if (numberDisplay) {
    numberDisplay.value = DEFAULT_SERVICE_NUMBER;
  }

  if (sessionInfo) {
    sessionInfo.innerText = "";
  }

  if (timerEl) {
    timerEl.innerText = "00:00";
  }

  setStatus("Idle");
  showCallerForm();
}

function pressKey(key) {
  if (activeCall || !numberDisplay) return;

  if (numberDisplay.value === DEFAULT_SERVICE_NUMBER) {
    numberDisplay.value = "";
  }

  numberDisplay.value += key;
}

function deleteKey() {
  if (activeCall || !numberDisplay) return;

  numberDisplay.value = numberDisplay.value.slice(0, -1);

  if (!numberDisplay.value) {
    numberDisplay.value = DEFAULT_SERVICE_NUMBER;
  }
}

function setStatus(text) {
  if (statusEl) {
    statusEl.innerText = text;
  }
}

function startTimer() {
  seconds = 0;

  if (timerEl) {
    timerEl.innerText = "00:00";
  }

  stopTimer();

  timerInterval = setInterval(() => {
    seconds++;

    const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
    const secs = String(seconds % 60).padStart(2, "0");

    if (timerEl) {
      timerEl.innerText = `${mins}:${secs}`;
    }
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

async function startCall() {
  // Initialize AudioContext immediately on user gesture to avoid browser blocks
  if (!audioContext) {
    audioContext = createAppAudioContext();
  }

  const dialedNumber = numberDisplay ? numberDisplay.value.trim() : "";

  if (!callerId) {
    alert("Please register caller first.");
    return;
  }

  if (!dialedNumber) {
    alert("Please dial the service number first.");
    return;
  }

  if (activeCall) {
    return;
  }

  try {
    setStatus("Requesting microphone...");

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: TARGET_SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    setStatus("Connecting...");

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    const wsUrl =
      `${protocol}://${window.location.host}/ws/call` +
      `?caller_id=${encodeURIComponent(callerId || "")}` +
      `&full_name=${encodeURIComponent(callerName || "")}` +
      `&phone_number=${encodeURIComponent(callerPhone || "")}` +
      `&audio_format=pcm16` +
      `&sample_rate=${TARGET_SAMPLE_RATE}`;

    websocket = new WebSocket(wsUrl);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = async () => {
      setStatus("In call");
      activeCall = true;
      startTimer();

      await startPCMStreaming(mediaStream);
    };

    websocket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        handleIncomingAudio(event.data);
        return;
      }

      try {
        const data = JSON.parse(event.data);

        if (data.type === "session_started") {
          if (sessionInfo) {
            sessionInfo.innerText = `Session: ${data.session_id}`;
          }
          const sid = data.session_id || "";
          if (sid && greetingPlayedSessionId !== sid) {
            greetingPlayedSessionId = sid;
            void playCallOpeningGreeting();
          }
        }

        if (data.event === "speech_started") {
          setStatus("Speaking...");
        }

        if (data.event === "speech_ended") {
          setStatus("Listening...");
        }

        if (data.type === "session_ended") {
          if (sessionInfo) {
            sessionInfo.innerText = "Call ended.";
          }
        }

      } catch (err) {
        console.log("Message:", event.data);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("Connection error");
    };

    websocket.onclose = (event) => {
      console.log("WebSocket closed:", event.code, event.reason);

      cleanupAudio();

      greetingPlayedSessionId = null;

      setStatus("Ended");
      activeCall = false;
      stopTimer();
    };

  } catch (error) {
    console.error("Microphone error:", error);
    setStatus("Microphone error");
    alert("Could not access microphone. Please allow microphone permission.");
  }
}

async function playCallOpeningGreeting() {
  try {
    setStatus("Advisor greeting…");
    const res = await fetch("/api/tts/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: CALL_GREETING_AM }),
    });
    if (!res.ok) {
      console.warn("Opening greeting TTS failed:", res.status);
      if (activeCall) setStatus("Listening...");
      return;
    }
    const raw = await res.arrayBuffer();
    const copy = raw.slice(0);
    if (!audioContext) {
      audioContext = createAppAudioContext();
    }
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    const decoded = await audioContext.decodeAudioData(copy);
    const src = audioContext.createBufferSource();
    src.buffer = decoded;
    src.connect(audioContext.destination);
    src.onended = () => {
      if (activeCall) setStatus("Listening...");
    };
    src.start();
  } catch (e) {
    console.warn("Opening greeting skipped:", e);
    if (activeCall) setStatus("Listening...");
  }
}

async function startPCMStreaming(stream) {
  if (!audioContext) {
    audioContext = createAppAudioContext();
  }

  sourceNode = audioContext.createMediaStreamSource(stream);

  const bufferSize = 1024;
  processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

  processorNode.onaudioprocess = (event) => {
    if (!activeCall || !websocket || websocket.readyState !== WebSocket.OPEN) {
      return;
    }

    const inputBuffer = event.inputBuffer.getChannelData(0);

    const downsampled = downsampleBuffer(
      inputBuffer,
      audioContext.sampleRate,
      TARGET_SAMPLE_RATE
    );

    const pcm16 = float32ToPCM16(downsampled);

    websocket.send(pcm16.buffer);
  };

  captureSinkNode = audioContext.createGain();
  captureSinkNode.gain.value = 0;

  sourceNode.connect(processorNode);
  // ScriptProcessor must be connected to run, but keep microphone capture inaudible.
  processorNode.connect(captureSinkNode);
  captureSinkNode.connect(audioContext.destination);
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return buffer;
  }

  if (outputSampleRate > inputSampleRate) {
    throw new Error("Output sample rate must be lower than input sample rate.");
  }

  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);

    let accum = 0;
    let count = 0;

    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }

    result[offsetResult] = count > 0 ? accum / count : 0;

    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function handleIncomingAudio(arrayBuffer) {
  if (!audioContext) {
    audioContext = createAppAudioContext();
  }

  // Modern browsers often suspend AudioContext until a user gesture or resume() call.
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }

  setStatus("Advisor is talking...");

  // Convert PCM16 to Float32
  const pcm16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    float32[i] = pcm16[i] / 32768.0;
  }

  const audioBuffer = audioContext.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  // Simple scheduling to avoid clicks
  const currentTime = audioContext.currentTime;
  if (playbackStartTime < currentTime) {
    playbackStartTime = currentTime + 0.05;
  }
  source.start(playbackStartTime);
  playbackStartTime += audioBuffer.duration;

  // Clear "Talking" status after a delay (approx when audio finishes)
  clearTimeout(window.statusTimeout);
  window.statusTimeout = setTimeout(() => {
    if (activeCall) setStatus("Listening...");
  }, 1500);
}

function float32ToPCM16(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    let sample = Math.max(-1, Math.min(1, float32Array[i]));

    if (sample < 0) {
      pcm16[i] = sample * 0x8000;
    } else {
      pcm16[i] = sample * 0x7fff;
    }
  }

  return pcm16;
}

function endCall() {
  if (!activeCall) return;

  setStatus("Ending call...");

  activeCall = false;
  stopTimer();

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send("END_CALL");

    setTimeout(() => {
      websocket.close();
    }, 300);
  }

  cleanupAudio();
}

async function testAudio() {
  try {
    console.log("Starting audio test...");
    if (!audioContext) {
      audioContext = createAppAudioContext();
    }
    
    // Explicitly wait for resume
    if (audioContext.state === "suspended") {
      console.log("Resuming suspended AudioContext...");
      await audioContext.resume();
    }
    
    console.log("Context state:", audioContext.state);
    
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    
    osc.type = "sine";
    osc.frequency.setValueAtTime(440, audioContext.currentTime);
    
    gain.gain.setValueAtTime(0.5, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.5);
    
    osc.connect(gain);
    gain.connect(audioContext.destination);
    
    osc.start();
    osc.stop(audioContext.currentTime + 0.5);
    
    console.log("Audio test beep played");
    alert("You should have heard a short beep. If not, check your speakers/volume.");
    
  } catch (err) {
    console.error("Audio test failed:", err);
    alert("Audio test failed: " + err.message);
  }
}

function cleanupAudio() {
  if (processorNode) {
    try {
      processorNode.disconnect();
    } catch (e) {}
    processorNode = null;
  }

  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (e) {}
    sourceNode = null;
  }

  if (captureSinkNode) {
    try {
      captureSinkNode.disconnect();
    } catch (e) {}
    captureSinkNode = null;
  }

  if (audioContext) {
    try {
      audioContext.close();
    } catch (e) {}
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}