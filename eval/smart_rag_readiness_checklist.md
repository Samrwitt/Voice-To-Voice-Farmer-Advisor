# Smart RAG Readiness Tests

Run with the stack up:

```bash
docker compose up -d postgres rag-service vad-service asr-service tts-service
export RAG_BASE_URL=http://127.0.0.1:8004
```

If `RAG_METRICS_TOKEN` is set, add `-H "Authorization: Bearer $RAG_METRICS_TOKEN"` to diagnostic calls.

## 1. Low Confidence Escalation

Ask an agrochemical/disease style question that has no KB hit:

```bash
curl -s "$RAG_BASE_URL/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"text":"unknown chemical spray dose for strange crop xyz","phone_number":"test","session_id":"low_conf_1"}' | jq
```

Expected:
- `trust.grounding` is `escalation`, or the answer clearly defers to an expert.
- The service creates an escalation row when Postgres is configured.

## 2. All PDFs Are Ingested

```bash
curl -s "$RAG_BASE_URL/rag/diagnostics" | jq '.kb_ingestion'
```

Expected:
- `local_pdf_count` equals the PDFs found under `RAG/KB` and configured KB folders.
- `missing_local_pdfs_in_pg_by_filename` is empty.
- `approved_pg_chunks` is greater than zero.

Note: startup ingestion now supports `AUTO_INGEST_KB_DIRS=/app/kb_documents/amharic,/app/RAG/KB`.

## 3. Dynamic Data Capture

```bash
curl -s "$RAG_BASE_URL/rag/diagnostics" | jq '.dynamic_data'
```

Expected:
- `dynamic_knowledge_cache` exists as the cache table.
- Weather and soil cache TTLs are visible.
- Market provider status is explicit.

## 4. Reliable Data Sources

```bash
curl -s "$RAG_BASE_URL/rag/diagnostics" | jq '.dynamic_data'
```

Current truth:
- Weather: Open-Meteo.
- Soil: SoilGrids/ISRIC, not CIAT yet.
- Market: local `market_prices` table plus demo fallback, not live NMIS yet.
- NMIS/ECX/ESS/FAO are planned adapter sources.

## 5. ASR Transcript Accepted By RAG

The VAD service calls:

```text
POST /rag/answer
{"text": "<ASR transcript>", "session_id": "...", "phone_number": "..."}
```

Direct test:

```bash
curl -s "$RAG_BASE_URL/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"text":"ስንዴ ለማምረት ምን ያስፈልጋል?","phone_number":"test","session_id":"asr_text_1"}' | jq
```

## 6. RAG Text Sent To TTS

In a live VAD WebSocket session, expect events:
- `asr_transcript`
- `rag_answer`
- `tts_started`

The `rag_answer.response` text is passed to `tts_service /synthesize`.

## 7. Farmer Profile Personalization

Use a phone number that exists in the callers/farmer profile DB:

```bash
curl -s "$RAG_BASE_URL/rag/debug/context" \
  -H "Content-Type: application/json" \
  -d '{"text":"ለእርሻዬ ምን ምክር አለ?","phone_number":"+251900000000","session_id":"profile_1"}' \
  | jq '.context.farmer_profile'
```

Expected:
- Profile fields such as location, farm size, crops, and language appear when present in DB.

## 8. Generic Data Uses Web Search

Enable web fallback for this test if needed:

```bash
RAG_WEB_ALLOW=1
curl -s "$RAG_BASE_URL/rag/debug/context" \
  -H "Content-Type: application/json" \
  -d '{"text":"latest Ethiopia agriculture update today","phone_number":"test","session_id":"web_1"}' \
  | jq '.tool_trace'
```

Expected:
- `web_search` appears only when KB is sparse or the question asks for current/general info.

## 9. NLU / AfroXLM-R

```bash
curl -s "$RAG_BASE_URL/rag/debug/context" \
  -H "Content-Type: application/json" \
  -d '{"text":"የጤፍ ዋጋ ስንት ነው?","phone_number":"test","session_id":"nlu_1","retrieve":false}' \
  | jq '.context.detected_intent,.context.entities'
```

Current truth:
- Rule-based multilingual NLU is active.
- AfroXLM-R is not loaded yet.
- Plug-in boundary: `farmer_rag_stack.smart_advisory.classify_intent_and_entities`.

## 10. Speed

```bash
time curl -s "$RAG_BASE_URL/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"text":"የጤፍ ዋጋ ስንት ነው?","phone_number":"test","session_id":"speed_1"}' >/tmp/rag_speed.json
cat /tmp/rag_speed.json | jq '.trust.latency_ms,.tool_trace'
```

Expected:
- Simple market/demo answers should avoid final LLM and web search.
- Repeated KB-only answers can hit the response cache when enabled.

## 11. Chat Endpoint With Same-Session Context

Send two turns using the same `session_id`:

```bash
curl -s "$RAG_BASE_URL/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"text":"ስለ ስንዴ ንገረኝ","phone_number":"test","session_id":"chat_ctx_1"}' | jq '.response'

curl -s "$RAG_BASE_URL/rag/debug/context" \
  -H "Content-Type: application/json" \
  -d '{"text":"ከዚያ መስኖ ያስፈልገዋል?","phone_number":"test","session_id":"chat_ctx_1"}' \
  | jq '.session_history_count,.context.farmer_profile.recent_advisory_records'
```

Expected:
- `session_history_count` is greater than zero.
- Recent messages appear inside `farmer_profile.recent_advisory_records`.

## 12. Predictive And Recommendation Analytics

```bash
curl -s "$RAG_BASE_URL/rag/debug/context" \
  -H "Content-Type: application/json" \
  -d '{"text":"Will it rain this week and should I irrigate my tomato?","phone_number":"test","session_id":"predict_1"}' \
  | jq '.context.prediction,.tool_trace'
```

Expected:
- `prediction.method` is `rules_v1`.
- Fields include irrigation need, disease risk, fertilizer recommendation, yield risk, market recommendation, and crop suitability.