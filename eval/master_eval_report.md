# Master Evaluation Report

## Scope
- Consolidated all evaluation outputs into one report.
- Includes system update summary plus all eval runner results.

## System Changes Included
- Concurrent SIP call handling added in `phone_gateway/backend/sip_audio.py`:
  - `SIP_MAX_CONCURRENT_CALLS` (default `8`)
  - `SIP_MAX_CONCURRENT_ALERT_CALLS` (default `4`)
  - Capacity/active-call logging for SIP and alert call paths.
- Compose env updated in `docker-compose.yml` with:
  - `SIP_MAX_CONCURRENT_CALLS: "8"`
  - `SIP_MAX_CONCURRENT_ALERT_CALLS: "4"`

## Eval Runs (All)

### 1) ASR Eval (`eval/run_asr_eval.py`)
- Exit code: `1`
- Cases: `4`
- Passed: `1`
- Failed: `3`
- Mean latency: `2802.1 ms`
- p95 latency: `3762.6 ms`
- Mean RTF: `0.888`
- Mean WER: `0.8476`
- Mean CER: `0.569`
- Main failures:
  - `farmer_advice_medium_tts_loopback` (`CER 0.6604 > 0.55`)
  - `safety_message_tts_loopback` (`CER 0.6481 > 0.55`)
  - `long_voice_answer_tts_loopback` (`WER 1.0 > 0.9`, `CER 0.7674 > 0.55`)

### 2) TTS Eval (`eval/run_tts_eval.py`)
- Exit code: `0`
- Cases: `4`
- Passed: `4`
- Failed: `0`
- Mean latency: `2155.6 ms`
- p95 latency: `3680.9 ms`
- Mean RTF: `0.397`

### 3) RAG Eval (`eval/run_rag_eval.py`)
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
  - `teff_price_intent`: expected scenario `market_price`, got `crop_production`

### 4) Component Battery (`eval/run_component_battery.py`)
- Exit code: `1`
- Checks: `13`
- Passed: `12`
- Failed: `1`
- Failure:
  - `market_teff`: expected scenario `market_price`, got `crop_production`

### 5) Performance Proof (`eval/run_performance_proof.py`)
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

## Consolidated Findings
- Concurrency hardening is in place for simultaneous SIP/callback calls.
- TTS path is fully passing current eval criteria.
- ASR quality still misses rubric thresholds on 3/4 loopback cases.
- One recurring routing issue exists across multiple evals:
  - Market-price intent for teff is being classified as crop production.

## Source Artifacts
- `eval/asr_performance_latest.json`
- `eval/asr_performance_latest.md`
- `eval/tts_performance_latest.json`
- `eval/tts_performance_latest.md`
- `eval/performance_proof_latest.json`
- `eval/performance_proof_latest.md`
- `eval/ast_eval_report.md` (previous consolidated report)
