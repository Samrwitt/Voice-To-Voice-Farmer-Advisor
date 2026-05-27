# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `0`
- Failed: `4`
- Mean latency: `5745.3` ms
- p95 latency: `6623.9` ms
- Mean RTF: `1.389`
- Mean WER: `None`
- Mean CER: `None`
- Mean normalized WER: `None`
- Mean normalized CER: `None`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | norm_wer | norm_cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `False` | `4918.9` | `1.521` | `3.234` | `None` | `None` | `None` | `None` | `None` |  |
| `farmer_advice_medium_tts_loopback` | `False` | `5330.0` | `6.18` | `0.863` | `None` | `None` | `None` | `None` | `None` |  |
| `safety_message_tts_loopback` | `False` | `6734.4` | `7.334` | `0.918` | `None` | `None` | `None` | `None` | `None` |  |
| `long_voice_answer_tts_loopback` | `False` | `5997.9` | `11.108` | `0.54` | `None` | `None` | `None` | `None` | `None` |  |
