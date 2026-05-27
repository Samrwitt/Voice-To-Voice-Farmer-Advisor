# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `1`
- Failed: `3`
- Mean latency: `8126.8` ms
- p95 latency: `9307.4` ms
- Mean RTF: `2.166`
- Mean WER: `0.8643`
- Mean CER: `0.5784`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `True` | `8599.9` | `1.521` | `5.654` | `0.6667` | `0.2` | `0.92` | ሳላም እንዴት ነዎች |
| `farmer_advice_medium_tts_loopback` | `False` | `7403.3` | `6.18` | `1.198` | `0.9333` | `0.6981` | `0.677` | ለእስ እ ንዲህ ማሳቡ ዘር ከመዝራት በፊቱም ባጢሩ ሁኔታ |
| `safety_message_tts_loopback` | `False` | `7071.5` | `7.334` | `0.964` | `0.8571` | `0.6481` | `0.719` | ጸረ ተባይ መጠቀም በፊት የሚህርቱን መ ማሪያ ያሙቡ |
| `long_voice_answer_tts_loopback` | `False` | `9432.3` | `11.108` | `0.849` | `1.0` | `0.7674` | `0.684` | የበ ቆሎ ማሳ ለማ እዘጋጀት መጀ ምሬ አፈዱን በደም ቢያር |
