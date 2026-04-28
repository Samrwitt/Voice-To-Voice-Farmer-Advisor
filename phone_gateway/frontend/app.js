let mediaRecorder = null;
let websocket = null;
let timerInterval = null;
let seconds = 0;
let activeCall = false;

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

window.onload = () => {
  if (callerId && callerName && callerPhone) {
    showDialer();
  } else {
    showCallerForm();
  }
};

function showCallerForm() {
  callerForm.classList.remove("hidden");
  dialer.classList.add("hidden");
}

function showDialer() {
  callerForm.classList.add("hidden");
  dialer.classList.remove("hidden");

  callerInfo.innerText = `${callerName} | ${callerPhone}`;
}

async function registerCaller() {
  const fullName = document.getElementById("fullName").value.trim();
  const phoneNumber = document.getElementById("phoneNumber").value.trim();

  if (!fullName || !phoneNumber) {
    alert("Please enter full name and phone number.");
    return;
  }

  try {
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
      alert("Failed to register caller.");
      return;
    }

    const data = await response.json();

    callerId = data.caller.caller_id;
    callerName = data.caller.full_name;
    callerPhone = data.caller.phone_number;

    localStorage.setItem("caller_id", callerId);
    localStorage.setItem("caller_name", callerName);
    localStorage.setItem("caller_phone", callerPhone);

    showDialer();

  } catch (error) {
    console.error(error);
    alert("Could not connect to the server.");
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

  numberDisplay.value = "";
  sessionInfo.innerText = "";
  setStatus("Idle");
  timerEl.innerText = "00:00";

  showCallerForm();
}

function pressKey(key) {
  if (activeCall) return;
  numberDisplay.value += key;
}

function deleteKey() {
  if (activeCall) return;
  numberDisplay.value = numberDisplay.value.slice(0, -1);
}

function setStatus(text) {
  statusEl.innerText = text;
}

function startTimer() {
  seconds = 0;
  timerEl.innerText = "00:00";

  timerInterval = setInterval(() => {
    seconds++;

    const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
    const secs = String(seconds % 60).padStart(2, "0");

    timerEl.innerText = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

async function startCall() {
  const dialedNumber = numberDisplay.value.trim();

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

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true
    });

    setStatus("Connecting...");

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    // We do NOT send dialedNumber to the backend.
    // The dialed number is only used for the phone-call-like UI.
    const wsUrl =
      `${protocol}://${window.location.host}/ws/call` +
      `?caller_id=${encodeURIComponent(callerId)}`;

    websocket = new WebSocket(wsUrl);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
      setStatus("In call");
      activeCall = true;
      startTimer();

      try {
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm"
        });
      } catch (error) {
        console.warn("audio/webm not supported, using default MediaRecorder format.");
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorder.ondataavailable = async (event) => {
        if (
          event.data &&
          event.data.size > 0 &&
          websocket &&
          websocket.readyState === WebSocket.OPEN
        ) {
          const buffer = await event.data.arrayBuffer();
          websocket.send(buffer);
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setStatus("Recording error");
      };

      // Send audio chunks every 500 ms
      mediaRecorder.start(500);
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "session_started") {
          sessionInfo.innerText =
            `Session started: ${data.session_id}`;
        }

        if (data.type === "session_ended") {
          sessionInfo.innerText =
            `Session ended. Audio saved: ${data.session.audio_file_path}`;
        }
      } catch (err) {
        console.log("Message:", event.data);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("WebSocket error");
    };

    websocket.onclose = () => {
      setStatus("Ended");
      activeCall = false;
      stopTimer();

      stream.getTracks().forEach(track => track.stop());
    };

  } catch (error) {
    console.error(error);
    setStatus("Microphone error");
    alert("Could not access microphone. Please allow microphone permission.");
  }
}

function endCall() {
  if (!activeCall) return;

  setStatus("Ending call...");

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send("END_CALL");

    setTimeout(() => {
      websocket.close();
    }, 300);
  }

  activeCall = false;
  stopTimer();
}