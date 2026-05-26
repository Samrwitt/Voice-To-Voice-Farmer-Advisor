# RAG Performance Proof

- Generated UTC: `2026-05-26T21:14:23.666959+00:00`
- Base URL: `http://127.0.0.1:8004`
- Command: `python3 eval/run_rag_eval.py`

## Summary

| Metric | Value |
|---|---:|
| `cases_passed` | `12` |
| `cases_total` | `12` |
| `case_pass_rate_pct` | `100.0` |
| `scenario_accuracy_pct` | `100.0` |
| `expected_grounding_accuracy_pct` | `100.0` |
| `forbidden_grounding_pass_rate_pct` | `100.0` |
| `latency_ms_mean` | `3893.0` |
| `latency_ms_p50` | `386.8` |
| `latency_ms_p95` | `12460.4` |
| `latency_ms_max` | `12460.4` |
| `avg_references` | `1.2` |

## Case Latency

| Case | OK | Grounding | Scenario | Latency ms | References |
|---|---:|---|---|---:|---:|
| `greeting_amharic` | `True` | `greeting` | `greeting_only` | `99.6` | `0` |
| `wheat_altitude` | `True` | `kb_llm` | `crop_production` | `12460.4` | `3` |
| `wheat_production_no_escalation` | `True` | `kb_llm` | `crop_production` | `5528.9` | `3` |
| `fertilizer_general` | `True` | `clarification` | `fertilizer` | `176.3` | `0` |
| `wheat_fertilizer_region_clarification` | `True` | `clarification` | `fertilizer` | `206.1` | `0` |
| `teff_price_intent` | `True` | `tools` | `market_price` | `562.7` | `0` |
| `weather_missing_location_clarifies` | `True` | `clarification` | `weather` | `183.7` | `0` |
| `coffee_pest` | `True` | `kb_llm` | `pest_disease` | `11368.3` | `3` |
| `compost_benefit` | `True` | `kb_unknown` | `fertilizer` | `4378.8` | `3` |
| `barley_region` | `True` | `kb_llm` | `crop_production` | `11358.5` | `3` |
| `urea_timing` | `True` | `clarification` | `fertilizer` | `182.3` | `0` |
| `agrochemical_safety_escalates` | `True` | `escalation` | `safety_agrochemical` | `210.9` | `0` |

## Failures

None.
