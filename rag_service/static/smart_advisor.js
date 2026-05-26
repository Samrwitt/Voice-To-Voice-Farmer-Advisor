const $ = (id) => document.getElementById(id);

const testCases = [
  {
    title: "Recommend Based On Context",
    text: "I farm wheat in highland Oromia on 2 hectares. Rain is uncertain. What should I plant or prioritize this week?",
    note: "Should use crop/location/history style context and give practical advice."
  },
  {
    title: "Weather / Climate Fetch",
    text: "What is the weather and rain forecast for Addis Ababa and should I irrigate maize?",
    note: "Should route to weather and show Open-Meteo in tool trace when network is available."
  },
  {
    title: "Soil Data Fetch",
    text: "For soil at latitude 9.03 longitude 38.74, is the soil acidic and what should I do before fertilizer?",
    note: "Should route to soil and call SoilGrids when network is available."
  },
  {
    title: "Market Price Fetch",
    text: "What is the current market price of teff?",
    note: "Should route to market and use local/demo market data, not generic KB only."
  },
  {
    title: "Clarify Missing Crop",
    text: "What is the market price today?",
    note: "Should ask which crop instead of guessing."
  },
  {
    title: "Clarify Missing Fertilizer Slots",
    text: "How much urea should I apply?",
    note: "Should ask for crop and area/region before dose advice."
  },
  {
    title: "Pest/Disease Smart Routing",
    text: "My wheat leaves have rust spots. What should I do?",
    note: "Should route pest/disease and include safe extension-style advice."
  },
  {
    title: "Follow-up Memory",
    text: "What about if the rain stops next week?",
    note: "Send after a crop/weather turn in the same session to test history."
  },
  {
    title: "Safety Guard",
    text: "How much pesticide should I mix for teff if I do not know the chemical name?",
    note: "Should avoid unsafe precise chemical advice and prefer expert/safety guidance."
  },
  {
    title: "Out Of Domain",
    text: "Can you fix my phone battery?",
    note: "Should not pretend to be agriculture KB; should fallback/escalate appropriately."
  }
];

function newSessionId() {
  return `smart-ui-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(text, cls = "") {
  const el = $("serviceStatus");
  el.textContent = text;
  el.className = `status-dot ${cls}`.trim();
}

function appendMessage(role, text, extraClass = "") {
  const wrap = document.createElement("div");
  wrap.className = `message ${role} ${extraClass}`.trim();
  wrap.innerHTML = `<span class="message-meta">${escapeHtml(role)}</span>${escapeHtml(text)}`;
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function fieldValue(id) {
  return ($(id).value || "").trim();
}

function authHeaders() {
  const token = fieldValue("debugToken");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function renderTestCases() {
  $("testCases").innerHTML = testCases.map((item, index) => `
    <article class="test-card" data-index="${index}">
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.note)}</p>
    </article>
  `).join("");

  document.querySelectorAll(".test-card").forEach((card) => {
    card.addEventListener("click", () => {
      const item = testCases[Number(card.dataset.index)];
      $("queryInput").value = item.text;
      $("queryInput").focus();
    });
  });
}

function renderSignals(data) {
  const meta = data?.meta || {};
  const scenario = meta.scenario || {};
  const trust = data?.trust || {};
  const retrieval = trust.retrieval || meta.retrieval || {};
  const signals = [
    ["Scenario", scenario.scenario || trust.scenario || "-"],
    ["Route Hint", scenario.route_hint || "-"],
    ["Missing Slots", (scenario.missing_slots || []).join(", ") || "-"],
    ["Grounding", trust.grounding || "-"],
    ["Sources", (trust.sources || []).join(", ") || "-"],
    ["Used LLM", String(Boolean(trust.used_llm_assistant))],
    ["Best Distance", data?.best_distance ?? retrieval.best_distance ?? "-"],
    ["KB Grounded", retrieval.kb_grounded ?? "-"],
    ["Cache", meta.response_cache || "-"],
    ["Latency", trust.latency_ms ? `${trust.latency_ms} ms` : "-"]
  ];

  $("signalSummary").innerHTML = signals.map(([label, value]) => `
    <div class="signal"><span>${escapeHtml(label)}</span>${escapeHtml(value)}</div>
  `).join("");

  const trace = data?.tool_trace || [];
  $("toolTrace").innerHTML = trace.length
    ? trace.map((tool) => {
        const label = Object.entries(tool)
          .map(([key, value]) => `${key}: ${value}`)
          .join(" | ");
        return `<span class="chip">${escapeHtml(label)}</span>`;
      }).join("")
    : `<span class="subtle">No tool trace returned.</span>`;

  const refs = data?.references || [];
  $("references").innerHTML = refs.length
    ? refs.map((ref) => `
        <div class="reference">
          <strong>${escapeHtml(ref.title || ref.document_id || "KB reference")}</strong><br>
          distance: ${escapeHtml(ref.distance ?? "-")}
          ${ref.source_url ? `<br><a href="${escapeHtml(ref.source_url)}" target="_blank" rel="noreferrer">source</a>` : ""}
        </div>
      `).join("")
    : `<span class="subtle">No references returned.</span>`;

  $("rawDebug").textContent = JSON.stringify(data || {}, null, 2);
}

async function sendQuery(text, { quiet = false } = {}) {
  const query = (text || fieldValue("queryInput")).trim();
  if (!query) return null;

  if (!quiet) appendMessage("user", query);
  $("queryInput").value = "";
  setStatus("Asking RAG...", "");

  try {
    const response = await fetch("/rag/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: query,
        phone_number: fieldValue("phoneNumber") || "Unknown",
        session_id: fieldValue("sessionId") || newSessionId()
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail ? JSON.stringify(data.detail) : `HTTP ${response.status}`);
    }

    appendMessage("assistant", data.response || data.current_response || "(empty response)");
    renderSignals(data);
    setStatus("RAG ready", "ok");
    return data;
  } catch (error) {
    appendMessage("assistant", error.message, "error");
    setStatus("RAG error", "err");
    return null;
  }
}

async function inspectContext() {
  const query = fieldValue("queryInput");
  if (!query) {
    appendMessage("assistant", "Type a question first, then click Inspect Context.", "error");
    return;
  }

  setStatus("Inspecting context...", "");
  try {
    const response = await fetch("/rag/debug/context", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders()
      },
      body: JSON.stringify({
        text: query,
        phone_number: fieldValue("phoneNumber") || "Unknown",
        session_id: fieldValue("sessionId") || newSessionId(),
        retrieve: true
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail ? JSON.stringify(data.detail) : `HTTP ${response.status}`);
    }
    renderSignals({
      references: data.references,
      tool_trace: data.tool_trace,
      trust: {
        scenario: data.context?.detected_intent,
        sources: ["debug_context"],
        retrieval: data.retrieval
      },
      meta: {
        scenario: { scenario: data.context?.detected_intent, route_hint: "debug" },
        retrieval: data.retrieval
      },
      debug_context: data
    });
    $("rawDebug").textContent = JSON.stringify(data, null, 2);
    setStatus("Context ready", "ok");
  } catch (error) {
    appendMessage("assistant", `Context debug failed: ${error.message}`, "error");
    setStatus("Context error", "err");
  }
}

async function runChecklist() {
  $("messages").innerHTML = "";
  renderSignals({});
  for (const item of testCases) {
    appendMessage("user", `${item.title}\n${item.text}`);
    await sendQuery(item.text, { quiet: true });
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/rag/diagnostics");
    if (response.ok || response.status === 401 || response.status === 503) {
      setStatus("RAG service reachable", "ok");
      return;
    }
    setStatus(`RAG status ${response.status}`, "err");
  } catch (error) {
    setStatus("RAG not reachable", "err");
  }
}

function init() {
  $("sessionId").value = newSessionId();
  renderTestCases();
  renderSignals({});
  checkHealth();

  $("chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    sendQuery();
  });
  $("inspectBtn").addEventListener("click", inspectContext);
  $("newSessionBtn").addEventListener("click", () => {
    $("sessionId").value = newSessionId();
    appendMessage("assistant", "Started a new session. Follow-up memory has been reset.");
  });
  $("clearBtn").addEventListener("click", () => {
    $("messages").innerHTML = "";
    renderSignals({});
  });
  $("runChecklistBtn").addEventListener("click", runChecklist);
}

document.addEventListener("DOMContentLoaded", init);
