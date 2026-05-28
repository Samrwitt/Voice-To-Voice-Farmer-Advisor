# RAG Performance Proof

- Generated UTC: `2026-05-28T07:37:14.932454+00:00`
- Base URL: `http://127.0.0.1:8004`
- Command: `python3 eval/run_rag_eval.py`

## Summary

| Metric | Value |
|---|---:|
| `cases_passed` | `11` |
| `cases_total` | `12` |
| `case_pass_rate_pct` | `91.7` |
| `scenario_accuracy_pct` | `85.7` |
| `expected_grounding_accuracy_pct` | `100.0` |
| `forbidden_grounding_pass_rate_pct` | `100.0` |
| `latency_ms_mean` | `1581.1` |
| `latency_ms_p50` | `73.8` |
| `latency_ms_p95` | `6481.1` |
| `latency_ms_max` | `6481.1` |
| `avg_references` | `1.2` |

## Case Latency

| Case | OK | Grounding | Scenario | Latency ms | References |
|---|---:|---|---|---:|---:|
| `greeting_amharic` | `True` | `greeting` | `greeting_only` | `60.3` | `0` |
| `wheat_altitude` | `True` | `kb_unknown` | `crop_production` | `5507.1` | `3` |
| `wheat_production_no_escalation` | `True` | `kb_unknown` | `crop_production` | `1105.0` | `3` |
| `fertilizer_general` | `True` | `clarification` | `fertilizer` | `66.1` | `0` |
| `wheat_fertilizer_region_clarification` | `True` | `clarification` | `fertilizer` | `73.7` | `0` |
| `teff_price_intent` | `False` | `kb_unknown` | `crop_production` | `1170.4` | `3` |
| `weather_missing_location_clarifies` | `True` | `clarification` | `weather` | `65.2` | `0` |
| `coffee_pest` | `True` | `kb_unknown` | `pest_disease` | `4233.0` | `3` |
| `compost_benefit` | `True` | `tools` | `fertilizer` | `70.5` | `0` |
| `barley_region` | `True` | `kb_unknown` | `crop_production` | `6481.1` | `3` |
| `urea_timing` | `True` | `clarification` | `fertilizer` | `67.0` | `0` |
| `agrochemical_safety_escalates` | `True` | `escalation` | `safety_agrochemical` | `74.0` | `0` |

## Failures

- teff_price_intent: scenario expected 'market_price' got 'crop_production'
