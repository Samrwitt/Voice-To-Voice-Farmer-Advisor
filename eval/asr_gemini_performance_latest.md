# ASR Performance Report

- Base URL: `http://127.0.0.1:8001`
- Cases: `4`
- Passed: `4`
- Failed: `0`
- Mean latency: `4069.6` ms
- p95 latency: `4950.5` ms
- Mean RTF: `0.972`
- Mean WER: `0.1048`
- Mean CER: `0.0368`
- Mean normalized WER: `0.0691`
- Mean normalized CER: `0.0229`

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | audio_sec | rtf | wer | cer | norm_wer | norm_cer | confidence | transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_greeting_tts_loopback` | `True` | `3419.4` | `1.521` | `2.248` | `0.0` | `0.0` | `0.0` | `0.0` | `0.92` | ሰላም እንዴት ነዎት |
| `farmer_advice_medium_tts_loopback` | `True` | `4595.6` | `6.18` | `0.744` | `0.1333` | `0.0566` | `0.1333` | `0.0566` | `0.816` | ስንዴ ማሳውን ዘር ከመዝራት በፊት መሬቱን በጥሩ ሁኔታ ያዘጋጁ። አፈር እርጥበት ካለው የተሻለ ውጤት ይሰጣል። |
| `safety_message_tts_loopback` | `True` | `3250.1` | `7.334` | `0.443` | `0.1429` | `0.0556` | `0.0` | `0.0` | `0.79` | ጸረ ተባይ ከመጠቀምዎ በፊት የምርቱን መመሪያ ያንብቡ። ጉአንት ይጠቀሙ እና የአካባቢ ግብርና ባለሙያን ያማክሩ |
| `long_voice_answer_tts_loopback` | `True` | `5013.1` | `11.108` | `0.451` | `0.1429` | `0.0349` | `0.1429` | `0.0349` | `0.666` | በቆሎ ማሳ ለማዘጋጀት መጀመሪያ አፈሩን በደንብ ያርሱ። የተሻለ ውጤት ለማግኘት ዘሩን ትክክለኛ ርቀት ይዝሩ። እንክርዳድን በጊዜ |
