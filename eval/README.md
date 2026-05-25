# RAG eval & quality (“proof, not vibes”)

## Golden questions

- File: `golden_questions.json` — Amharic questions plus a small **rubric** per case (substrings, max latency, min length).
- Runner: `run_rag_eval.py` — calls `POST {RAG_BASE_URL}/rag/answer` for each case and prints a JSON report. Exit code **1** if any rubric fails (CI-friendly).

### Run locally

```bash
export RAG_BASE_URL=http://127.0.0.1:8004   # host port mapped to rag-service
python3 eval/run_rag_eval.py
```

Optional: `EVAL_PHONE=+2519...` if personalization paths need a real profile.

Tune rubrics as your KB grows (keywords are intentionally loose).

## Quality snapshot (Postgres)

`GET {RAG_BASE_URL}/api/quality/snapshot?hours=24`

- If `RAG_METRICS_TOKEN` is set in the environment, send header:  
  `Authorization: Bearer <same value>`
- Returns counts from `interaction_records` and `escalations`, plus `escalations_pending_over_sla` vs `ESCALATION_SLA_HOURS` (default 48), and structured **`ops_alerts`** for dashboards and alerting.

Policy knobs (read in snapshot JSON `policy`):

- `CALL_RECORDING_RETENTION_DAYS` (default 30) — document for ops; wire deletion jobs separately.
- `ESCALATION_SLA_HOURS` — expert backlog target.

## Voice trust metadata

`POST /rag/answer` now includes a **`trust`** object, e.g.:

- `sources`: `kb`, `dynamic`, `expert_delivery`, `escalation`
- `grounding`: `kb_llm` | `kb_compose` | `dynamic_only` | `none` | `escalation`
- `latency_ms`, `escalation_sla_target_hours`, `human_review`

Optional Amharic disclaimer footer on KB answers: set `RAG_TRUST_FOOTER=1`.

### Agrochemical safety & low-confidence escalation (voice RAG)

When `RAG_AGROCHEM_EXPERT_ONLY=1` (default in `docker-compose`), pesticide / fertilizer / spray / dose questions **without a confident KB match** escalate (`AGROCHEM_NO_KB`). A match requires in-threshold chunk distances **and** `best_distance <= RAG_PG_MAX_L2_DISTANCE` (weak Chroma-only junk with a high PG `best` still escalates).

`RAG_VOICE_LOW_CONF_ESCALATE=1` (default) applies the same distance rule to **non-agrochemical** queries (`LOW_CONFIDENCE`).

`POST /rag/answer` logs user/assistant turns to `conversation_history` so follow-up questions in the same `session_id` appear in `/rag/debug/context` (`session_history_count`, `recent_advisory_records`).

`POST /ask` on **logic_service** forwards the same **`trust`** object when the answer came from `RAG_SERVICE_URL` `/rag/answer`, so browser and JSON clients—not only the VAD WebSocket—see grounding. The VAD service also attaches **`trust`** on the WebSocket `rag_answer` event when present.

## Product & ops

### Farmer escalation status (logic_service)

`POST http://logic-service:8002/product/escalation-status` with JSON `{ "phone_number": "+251...", "session_id": "optional", "limit": 8 }` returns recent tickets **without** the original question text — only `status`, Amharic labels, and timestamps. Use for IVR/USSD “where is my case?” flows.

### Ops alerts in the quality snapshot

`GET /api/quality/snapshot` includes **`ops_alerts`**: SLA breach warnings and a small **info** backlog line when there are pending tickets still inside the SLA window.

### Webhook push (cron)

`POST {RAG_BASE_URL}/api/ops/notify` with header `Authorization: Bearer <OPS_NOTIFY_TOKEN or RAG_METRICS_TOKEN>` posts JSON to **`OPS_ALERT_WEBHOOK_URL`** when `escalations_pending_over_sla > 0`. Otherwise returns `{ "pushed": false }`. Schedule every 10–30 minutes from your job runner.

### Response cache (latency / cost)

When **`RAG_RESPONSE_CACHE_TTL_SEC`** > 0, `/rag/answer` caches **KB-only** replies (no dynamic block, no expert delivery prefix) keyed by normalized query + phone + region. Cache hits include `"meta": { "response_cache": "hit" }`.

## Team API keys (Groq / Gemini rate limits)

Put every teammate's key in the **root** `.env` (see `.env.example`):

```bash
GROQ_API_KEYS=key1,key2,key3,key4,key5
GEMINI_API_KEYS=key1,key2,key3,key4,key5
RAG_LLM_BACKEND=groq
GROQ_GEMINI_FALLBACK=1
```

The RAG service **rotates** keys round-robin per request. On HTTP **429/503**, that key is cooled down (~90s, `RAG_API_KEY_COOLDOWN_SEC`) and the **next** key is used. If all Groq keys are busy, it falls back to the Gemini pool (when `GROQ_GEMINI_FALLBACK=1`).

## Distribution (non-code)

Partnerships (coops, extension, telco) are **go-to-market** work: this folder only automates **measurement** so pilots stay honest.