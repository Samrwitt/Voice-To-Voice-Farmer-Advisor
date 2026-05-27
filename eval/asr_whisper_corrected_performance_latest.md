# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `1`
- Failed: `3`
- Mean latency: `7795.8` ms
- p95 latency: `10264.0` ms
- Mean RTF: `1.752`
- Mean WER: `0.6976`
- Mean CER: `0.6431`
- Mean normalized WER: `0.6797`
- Mean normalized CER: `0.6402`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | norm_wer | norm_cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `True` | `5517.9` | `1.521` | `3.628` | `0.0` | `0.0` | `0.0` | `0.0` | `0.92` | ሰላም እንዴት ነዎት |
| `farmer_advice_medium_tts_loopback` | `False` | `10601.6` | `6.18` | `1.715` | `0.9333` | `0.9245` | `0.9333` | `0.9245` | `0.638` | እንዲህ ማሳው ዘር |
| `safety_message_tts_loopback` | `False` | `6712.7` | `7.334` | `0.915` | `0.8571` | `0.6481` | `0.7857` | `0.6364` | `0.719` | ጸረ ተባይ መጠቀም በፊት የሚህርቱን መ ማሪያ ያሙቡ |
| `long_voice_answer_tts_loopback` | `False` | `8351.1` | `11.108` | `0.752` | `1.0` | `1.0` | `1.0` | `1.0` | `0.63` | The user wants me to correct an Amharic A |
