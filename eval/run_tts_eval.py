#!/usr/bin/env python3
"""Evaluate the TTS service with operational audio metrics.

This is intentionally dependency-light so it can run in CI or on the host.
For reference-free TTS, the reliable automated metrics are latency, real-time
factor, WAV format, loudness, clipping, and silence. MOS/PESQ/STOI need either
human listeners or reference recordings and are reported as not measured here.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "eval" / "tts_cases.json"
OUT_JSON = ROOT / "eval" / "tts_performance_latest.json"
OUT_MD = ROOT / "eval" / "tts_performance_latest.md"


def dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value / 32768.0)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return round(ordered[int(k)], 1)
    return round(ordered[floor] * (ceil - k) + ordered[ceil] * (k - floor), 1)


def read_wav_metrics(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    duration = frame_count / sample_rate if sample_rate else 0.0
    samples = array.array("h")
    if sample_width == 2:
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
    else:
        samples = array.array("h")

    if samples:
        abs_samples = [abs(s) for s in samples]
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        peak = max(abs_samples)
        clipped = sum(1 for s in abs_samples if s >= 32760) / len(abs_samples)
    else:
        rms = 0.0
        peak = 0
        clipped = 0.0

    silence_ratio = 0.0
    if samples and sample_rate:
        window = max(1, int(sample_rate * channels * 0.02))
        silent = 0
        total = 0
        for i in range(0, len(samples), window):
            chunk = samples[i : i + window]
            if not chunk:
                continue
            chunk_rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            if dbfs(chunk_rms) < -50.0:
                silent += 1
            total += 1
        silence_ratio = silent / total if total else 0.0

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_sec": round(duration, 3),
        "rms_dbfs": round(dbfs(rms), 1),
        "peak_dbfs": round(dbfs(float(peak)), 1),
        "clipping_ratio": round(clipped, 6),
        "silence_ratio": round(silence_ratio, 4),
    }


def synthesize(base_url: str, text: str, timeout: float) -> tuple[int, float, bytes, str]:
    url = base_url.rstrip("/") + "/synthesize"
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            status = resp.status
            content_type = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        data = exc.read()
        status = exc.code
        content_type = exc.headers.get("content-type", "")
    latency_ms = (time.perf_counter() - start) * 1000.0
    return status, latency_ms, data, content_type


def evaluate_case(case: dict[str, Any], base_url: str, timeout: float, out_dir: Path) -> dict[str, Any]:
    status, latency_ms, audio, content_type = synthesize(base_url, case["text"], timeout)
    audio_path = out_dir / f"{case['id']}.wav"
    audio_path.write_bytes(audio)

    row: dict[str, Any] = {
        "id": case["id"],
        "chars": len(case["text"]),
        "status": status,
        "content_type": content_type,
        "bytes": len(audio),
        "latency_ms": round(latency_ms, 1),
        "audio_path": str(audio_path),
    }
    errors: list[str] = []
    if status != 200:
        errors.append(f"HTTP {status}")
    if "audio" not in content_type.lower() and status == 200:
        errors.append(f"unexpected content-type {content_type!r}")

    if status == 200:
        try:
            metrics = read_wav_metrics(audio_path)
            row.update(metrics)
            duration = float(metrics.get("duration_sec") or 0.0)
            row["rtf"] = round((latency_ms / 1000.0) / duration, 3) if duration > 0 else None
        except Exception as exc:
            errors.append(f"invalid wav: {exc}")

    rubric = case.get("rubric") or {}
    if row.get("sample_rate") != 16000:
        errors.append(f"sample_rate {row.get('sample_rate')} != 16000")
    if row.get("channels") != 1:
        errors.append(f"channels {row.get('channels')} != 1")
    if row.get("sample_width_bytes") != 2:
        errors.append(f"sample_width {row.get('sample_width_bytes')} != 2")
    if "max_latency_ms" in rubric and latency_ms > float(rubric["max_latency_ms"]):
        errors.append(f"latency {latency_ms:.0f}ms > {rubric['max_latency_ms']}ms")
    if "max_rtf" in rubric and row.get("rtf") is not None and float(row["rtf"]) > float(rubric["max_rtf"]):
        errors.append(f"rtf {row['rtf']} > {rubric['max_rtf']}")
    if "min_duration_sec" in rubric and float(row.get("duration_sec") or 0.0) < float(rubric["min_duration_sec"]):
        errors.append(f"duration {row.get('duration_sec')}s < {rubric['min_duration_sec']}s")
    if float(row.get("clipping_ratio") or 0.0) > 0.01:
        errors.append(f"clipping_ratio {row.get('clipping_ratio')} > 0.01")
    if float(row.get("silence_ratio") or 0.0) > 0.6:
        errors.append(f"silence_ratio {row.get('silence_ratio')} > 0.6")
    if float(row.get("rms_dbfs") or -120.0) < -45.0:
        errors.append(f"rms_dbfs {row.get('rms_dbfs')} < -45")

    row["ok"] = not errors
    row["errors"] = errors
    return row


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# TTS Performance Report",
        "",
        f"- Base URL: `{report['base_url']}`",
        f"- Cases: `{summary['cases']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Failed: `{summary['failed']}`",
        f"- Mean latency: `{summary['latency_ms_mean']}` ms",
        f"- p95 latency: `{summary['latency_ms_p95']}` ms",
        f"- Mean RTF: `{summary['rtf_mean']}`",
        "",
        "Automated metrics: HTTP success, synthesis latency, real-time factor, WAV sample rate/channels/bit depth, loudness proxy, clipping ratio, and silence ratio.",
        "",
        "Not measured automatically: MOS, CMOS, MUSHRA, PESQ, STOI, speaker similarity, and human naturalness. Those require listening panels or reference recordings.",
        "",
        "| id | ok | latency_ms | duration_sec | rtf | rms_dbfs | peak_dbfs | clipping | silence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["cases"]:
        lines.append(
            "| `{id}` | `{ok}` | `{latency}` | `{duration}` | `{rtf}` | `{rms}` | `{peak}` | `{clip}` | `{silence}` |".format(
                id=row.get("id"),
                ok=row.get("ok"),
                latency=row.get("latency_ms"),
                duration=row.get("duration_sec"),
                rtf=row.get("rtf"),
                rms=row.get("rms_dbfs"),
                peak=row.get("peak_dbfs"),
                clip=row.get("clipping_ratio"),
                silence=row.get("silence_ratio"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "eval" / "tts_outputs")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_case(case, args.base_url, args.timeout, args.audio_dir) for case in cases]
    latencies = [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    rtfs = [float(r["rtf"]) for r in rows if isinstance(r.get("rtf"), (int, float))]
    summary = {
        "cases": len(rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "failed": sum(1 for r in rows if not r["ok"]),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_max": round(max(latencies), 1) if latencies else None,
        "rtf_mean": round(statistics.mean(rtfs), 3) if rtfs else None,
        "rtf_p95": percentile(rtfs, 95),
        "rtf_max": round(max(rtfs), 3) if rtfs else None,
    }
    report = {
        "base_url": args.base_url,
        "summary": summary,
        "cases": rows,
        "notes": {
            "measured": [
                "latency_ms",
                "real_time_factor",
                "wav_format",
                "duration_sec",
                "rms_dbfs",
                "peak_dbfs",
                "clipping_ratio",
                "silence_ratio",
            ],
            "not_measured_without_references_or_humans": ["MOS", "CMOS", "MUSHRA", "PESQ", "STOI"],
        },
    }
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
