# TTS Performance Report

- Base URL: `http://127.0.0.1:8009`
- Cases: `4`
- Passed: `4`
- Failed: `0`
- Mean latency: `2275.1` ms
- p95 latency: `3538.2` ms
- Mean RTF: `0.459`

Automated metrics: HTTP success, synthesis latency, real-time factor, WAV sample rate/channels/bit depth, loudness proxy, clipping ratio, and silence ratio.

Not measured automatically: MOS, CMOS, MUSHRA, PESQ, STOI, speaker similarity, and human naturalness. Those require listening panels or reference recordings.

| id | ok | latency_ms | duration_sec | rtf | rms_dbfs | peak_dbfs | clipping | silence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_greeting` | `True` | `1394.4` | `1.521` | `0.917` | `-17.2` | `-3.9` | `0.0` | `0.2857` |
| `farmer_advice_medium` | `True` | `1578.7` | `6.18` | `0.255` | `-16.1` | `-1.3` | `0.0` | `0.1129` |
| `safety_message` | `True` | `2385.5` | `7.334` | `0.325` | `-15.9` | `-1.5` | `0.0` | `0.2044` |
| `long_voice_answer` | `True` | `3741.6` | `11.108` | `0.337` | `-16.0` | `-2.0` | `0.0` | `0.1673` |
