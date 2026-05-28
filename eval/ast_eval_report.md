# Eval Report (All Runners)

## Scope
- Added concurrent call handling controls for SIP call legs and alert/callback legs.
- Ran all eval runners in `eval/`:
  - `run_asr_eval.py`
  - `run_tts_eval.py`
  - `run_rag_eval.py`
  - `run_component_battery.py`
  - `run_performance_proof.py`

## Concurrent Call Handling Changes
- Updated `phone_gateway/backend/sip_audio.py`:
  - Added bounded concurrency using semaphores.
  - Added caps:
    - `SIP_MAX_CONCURRENT_CALLS` (default `8`)
    - `SIP_MAX_CONCURRENT_ALERT_CALLS` (default `4`)
  - Added active-call/capacity logs for SIP and alert AudioSocket handlers.
- Updated `docker-compose.yml`:
  - Added `SIP_MAX_CONCURRENT_CALLS: "8"` under `phone-gateway`.
  - Added `SIP_MAX_CONCURRENT_ALERT_CALLS: "4"` under `phone-gateway`.

## Results by Eval Runner

### 1) ASR Eval (`python3 eval/run_asr_eval.py`)
- Exit code: `1`
- Cases: `4`
- Passed: `1`
- Failed: `3`
- Mean latency: `2802.1 ms`
- p95 latency: `3762.6 ms`
- Mean RTF: `0.888`
- Mean WER: `0.8476`
- Mean CER: `0.569`
- Key failures:
  - `farmer_advice_medium_tts_loopback`: `CER 0.6604 > 0.55`
  - `safety_message_tts_loopback`: `CER 0.6481 > 0.55`
  - `long_voice_answer_tts_loopback`: `WER 1.0 > 0.9`, `CER 0.7674 > 0.55`

### 2) TTS Eval (`python3 eval/run_tts_eval.py`)
- Exit code: `0`
- Cases: `4`
- Passed: `4`
- Failed: `0`
- Mean latency: `2155.6 ms`
- p95 latency: `3680.9 ms`
- Mean RTF: `0.397`

### 3) RAG Eval (`python3 eval/run_rag_eval.py`)
- Exit code: `1`
- Cases: `12`
- Passed: `11`
- Failed: `1`
- Case pass rate: `91.7%`
- Scenario accuracy: `85.7%`
- Expected grounding accuracy: `100.0%`
- Mean latency: `2032.4 ms`
- p95 latency: `7124.4 ms`
- Avg references: `1.2`
- Failure:
  - `teff_price_intent`: scenario expected `market_price`, got `crop_production`

### 4) Component Battery (`python3 eval/run_component_battery.py`)
- Exit code: `1`
- Total checks: `13`
- Passed: `12`
- Failed: `1`
- Failure:
  - `market_teff`: scenario expected `market_price`, got `crop_production`

### 5) Performance Proof (`python3 eval/run_performance_proof.py`)
- Exit code: `1`
- Cases: `12`
- Passed: `11`
- Failed: `1`
- Case pass rate: `91.7%`
- Scenario accuracy: `85.7%`
- Expected grounding accuracy: `100.0%`
- Mean latency: `1581.1 ms`
- p50 latency: `73.8 ms`
- p95 latency: `6481.1 ms`
- Avg references: `1.2`

## Generated Artifacts
- `eval/asr_performance_latest.json`
- `eval/asr_performance_latest.md`
- `eval/tts_performance_latest.json`
- `eval/tts_performance_latest.md`
- `eval/performance_proof_latest.json`
- `eval/performance_proof_latest.md`

## Summary
- Concurrency controls are now in place for SIP and callback call handling.
- TTS eval is fully passing.
- ASR eval has 3/4 failing on transcript quality thresholds.
- RAG/component/performance runs consistently show one routing miss: market-price intent (`teff_price_intent` / `market_teff`) classified as crop production.
