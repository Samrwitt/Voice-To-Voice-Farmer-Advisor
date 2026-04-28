let mediaRecorder = null;
let websocket = null;
let timerInterval = null;
let seconds = 0;
let activeCall = false;

const numberDisplay = document.getElementById("numberDisplay");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const sessionInfo = document.getElementById("sessionInfo");

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
  clearInterval(timerInterval);
}

async function startCall() {
  const dialedNumber = numberDisplay.value.trim();

  if (!dialedNumber) {
    alert("Please dial a number first.");
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
    const wsUrl = `${protocol}://${window.location.host}/ws/call?dialed_number=${encodeURIComponent(dialedNumber)}`;

    websocket = new WebSocket(wsUrl);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
      setStatus("In call");
      activeCall = true;
      startTimer();

      mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm"
      });

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

      mediaRecorder.start(500);
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "session_started") {
          sessionInfo.innerText = `Session started: ${data.session_id}`;
        }

        if (data.type === "session_ended") {
          sessionInfo.innerText = `Session ended. Audio saved: ${data.session.audio_file}`;
        }
      } catch (err) {
        console.log("Message:", event.data);
      }
    };

    websocket.onerror = () => {
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