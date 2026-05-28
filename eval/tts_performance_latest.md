# TTS Performance Report

- Base URL: `http://127.0.0.1:8009`
- Cases: `4`
- Passed: `4`
- Failed: `0`
- Mean latency: `2155.6` ms
- p95 latency: `3680.9` ms
- Mean RTF: `0.397`

Automated metrics: HTTP success, synthesis latency, real-time factor, WAV sample rate/channels/bit depth, loudness proxy, clipping ratio, and silence ratio.

Not measured automatically: MOS, CMOS, MUSHRA, PESQ, STOI, speaker similarity, and human naturalness. Those require listening panels or reference recordings.

| id | ok | latency_ms | duration_sec | rtf | rms_dbfs | peak_dbfs | clipping | silence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_greeting` | `True` | `1061.3` | `1.522` | `0.697` | `-17.3` | `-3.9` | `0.0` | `0.2857` |
| `farmer_advice_medium` | `True` | `1752.3` | `6.18` | `0.284` | `-16.1` | `-1.3` | `0.0` | `0.1129` |
| `safety_message` | `True` | `1794.9` | `7.314` | `0.245` | `-15.9` | `-1.6` | `0.0` | `0.1995` |
| `long_voice_answer` | `True` | `4013.7` | `11.108` | `0.361` | `-16.0` | `-2.0` | `0.0` | `0.1673` |
