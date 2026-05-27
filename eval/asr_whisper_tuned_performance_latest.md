# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `0`
- Failed: `4`
- Mean latency: `8977.8` ms
- p95 latency: `11035.2` ms
- Mean RTF: `2.046`
- Mean WER: `0.9024`
- Mean CER: `0.5799`
- Mean normalized WER: `0.8845`
- Mean normalized CER: `0.577`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | norm_wer | norm_cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `False` | `6782.6` | `1.521` | `4.459` | `1.0` | `0.3` | `1.0` | `0.3` | `0.547` | ሳ እላም አንዴት ነዎት |
| `farmer_advice_medium_tts_loopback` | `False` | `8063.4` | `6.18` | `1.305` | `0.8` | `0.6226` | `0.8` | `0.6226` | `0.732` | ለእስንዲህ ማሳውን ዘር ከመዝራት በፊቱም ባጢሩ ሁኔታ ያዘ ጋጁ |
| `safety_message_tts_loopback` | `False` | `11250.6` | `7.334` | `1.534` | `0.8571` | `0.6296` | `0.7857` | `0.6182` | `0.715` | ጸረ ተባይ መጠቀም በፊት የሚህርቱን መ ማሪያ ያእሙቡ |
| `long_voice_answer_tts_loopback` | `False` | `9814.5` | `11.108` | `0.884` | `0.9524` | `0.7674` | `0.9524` | `0.7674` | `0.665` | የበጐሎ ማሳ ለማ እዘጋጀት መጀ ምሬ አፈዱን በ ደም ቢያርሱ |
