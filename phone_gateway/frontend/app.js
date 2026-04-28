let mediaRecorder = null;
let websocket = null;
let timerInterval = null;
let seconds = 0;
let activeCall = false;

const DEFAULT_SERVICE_NUMBER = "8028";

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

  if (fullNameInput) {
    fullNameInput.value = "";
  }

  if (phoneNumberInput) {
    phoneNumberInput.value = "";
  }

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

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true
    });

    setStatus("Connecting...");

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    // The dialed number is only used for UI.
    // We do not send or save the dialed number.
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

      mediaRecorder.start(500);
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "session_started") {
          if (sessionInfo) {
            sessionInfo.innerText = `Session started: ${data.session_id}`;
          }
        }

        if (data.type === "session_ended") {
          if (sessionInfo) {
            sessionInfo.innerText =
              `Session ended. Audio saved: ${data.session.audio_file_path}`;
          }
        }
      } catch (err) {
        console.log("Message:", event.data);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("WebSocket error");
    };

    websocket.onclose = (event) => {
      console.log("WebSocket closed:", event.code, event.reason);

      setStatus("Ended");
      activeCall = false;
      stopTimer();

      stream.getTracks().forEach(track => track.stop());
    };

  } catch (error) {
    console.error("Microphone error:", error);
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