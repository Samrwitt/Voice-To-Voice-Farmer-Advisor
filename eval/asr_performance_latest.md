# ASR Performance Report

- Generated: `2026-05-27T16:20:12.633175+00:00`
- Base URL: `http://127.0.0.1:8001`
- Engine config: `whisper_local` / runtime `whisper_local`
- Cases: `4`
- Passed: `0`
- Failed: `4`
- LLM fix triggered: `0` / `4`
- Mean latency: `117.1` ms (engine `None` ms)
- p95 latency: `139.6` ms
- Mean RTF: `0.026`
- Mean WER: `None` (normalized `None`)
- Mean CER: `None` (normalized `None`)

These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.

| id | ok | latency_ms | wer | cer | conf | fix | errors |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `short_greeting_tts_loopback` | `False` | `86.8` | `None` | `None` | `None` | `-` | HTTP 500: {'detail': 'Library libcublas.so.12 is not found o |
| `farmer_advice_medium_tts_loopback` | `False` | `106.6` | `None` | `None` | `None` | `-` | HTTP 500: {'detail': 'Library libcublas.so.12 is not found o |
| `safety_message_tts_loopback` | `False` | `134.4` | `None` | `None` | `None` | `-` | HTTP 500: {'detail': 'Library libcublas.so.12 is not found o |
| `long_voice_answer_tts_loopback` | `False` | `140.5` | `None` | `None` | `None` | `-` | HTTP 500: {'detail': 'Library libcublas.so.12 is not found o |

## Transcripts

### `short_greeting_tts_loopback`
- Reference: ሰላም፣ እንዴት ነዎት?
- Raw: None
- Domain: None
- Final: 

### `farmer_advice_medium_tts_loopback`
- Reference: ለስንዴ ማሳዎ ዘር ከመዝራት በፊት መሬቱን በጥሩ ሁኔታ ያዘጋጁ። አፈር እርጥበት ካለው የተሻለ ውጤት ይሰጣል።
- Raw: None
- Domain: None
- Final: 

### `safety_message_tts_loopback`
- Reference: ፀረ ተባይ ከመጠቀምዎ በፊት የምርቱን መመሪያ ያንብቡ፣ ጓንት ይጠቀሙ፣ እና የአካባቢ ግብርና ባለሙያን ያማክሩ።
- Raw: None
- Domain: None
- Final: 

### `long_voice_answer_tts_loopback`
- Reference: የበቆሎ ማሳ ለማዘጋጀት መጀመሪያ አፈሩን በደንብ ያርሱ። የተሻለ ውጤት ለማግኘት ዘሩን በትክክለኛ ርቀት ይዝሩ፣ እንክርዳድን በጊዜው ያስወግዱ፣ እና የዝናብ ሁኔታን ይከታተሉ።
- Raw: None
- Domain: None
- Final: 

