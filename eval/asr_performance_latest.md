# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `1`
- Failed: `3`
- Mean latency: `2802.1` ms
- p95 latency: `3762.6` ms
- Mean RTF: `0.888`
- Mean WER: `0.8476`
- Mean CER: `0.569`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `True` | `3963.6` | `1.521` | `2.606` | `0.6667` | `0.2` | `0.92` | ሳላም እንዴት ነዎች |
| `farmer_advice_medium_tts_loopback` | `False` | `2623.3` | `6.18` | `0.425` | `0.8667` | `0.6604` | `0.715` | ለስ እንዲህ ማሳቡ ዘር ከመዝራት በፊቱም ባጢሩ ሁኔታ ያ |
| `safety_message_tts_loopback` | `False` | `2289.0` | `7.334` | `0.312` | `0.8571` | `0.6481` | `0.719` | ጸረ ተባይ መጠቀም በፊት የሚህርቱን መ ማሪያ ያሙቡ |
| `long_voice_answer_tts_loopback` | `False` | `2332.3` | `11.108` | `0.21` | `1.0` | `0.7674` | `0.684` | የበ ቆሎ ማሳ ለማ እዘጋጀት መጀ ምሬ አፈዱን በደም ቢያር |
