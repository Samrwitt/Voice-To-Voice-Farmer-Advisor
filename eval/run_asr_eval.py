#!/usr/bin/env python3
"""Evaluate ASR service latency and transcript accuracy.

Uses reference transcripts when available and reports WER/CER. The included
cases are TTS loopback clips, which are useful for repeatable smoke tests but
should not replace real farmer-call recordings for final ASR claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "eval" / "asr_cases.json"
OUT_JSON = ROOT / "eval" / "asr_performance_latest.json"
OUT_MD = ROOT / "eval" / "asr_performance_latest.md"


def normalize_amharic_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[።፣፤፥፦!?.,;:\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float | None:
    ref_words = normalize_amharic_text(reference).split()
    hyp_words = normalize_amharic_text(hypothesis).split()
    if not ref_words:
        return None
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float | None:
    ref = normalize_amharic_text(reference).replace(" ", "")
    hyp = normalize_amharic_text(hypothesis).replace(" ", "")
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def audio_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


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


def multipart_body(field_name: str, path: Path, boundary: str) -> tuple[bytes, str]:
    data = path.read_bytes()
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + data + footer, f"multipart/form-data; boundary={boundary}"


def transcribe(base_url: str, audio_path: Path, timeout: float) -> tuple[int, float, dict[str, Any], str]:
    url = base_url.rstrip("/") + "/transcribe"
    boundary = "----codex-asr-eval-boundary"
    body, content_type = multipart_body("file", audio_path, boundary)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": content_type},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
            content_type = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
        content_type = exc.headers.get("content-type", "")
    latency_ms = (time.perf_counter() - start) * 1000.0
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"raw_response": raw}
    return status, latency_ms, payload, content_type


def evaluate_case(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    audio_path = (ROOT / case["audio_path"]).resolve()
    duration = audio_duration(audio_path)
    row: dict[str, Any] = {
        "id": case["id"],
        "audio_path": str(audio_path),
        "reference": case.get("reference", ""),
        "audio_duration_sec": round(duration, 3) if duration else None,
    }
    errors: list[str] = []
    if not audio_path.exists():
        row.update({"status": None, "ok": False, "errors": [f"missing audio file: {audio_path}"]})
        return row

    status, latency_ms, payload, content_type = transcribe(base_url, audio_path, timeout)
    hyp = payload.get("final_transcript") or payload.get("transcript") or payload.get("text") or ""
    row.update(
        {
            "status": status,
            "content_type": content_type,
            "latency_ms": round(latency_ms, 1),
            "engine_latency_seconds": payload.get("latency_seconds"),
            "rtf": round((latency_ms / 1000.0) / duration, 3) if duration else None,
            "language": payload.get("language"),
            "language_probability": payload.get("language_probability"),
            "confidence": payload.get("confidence"),
            "acoustic_confidence": payload.get("acoustic_confidence"),
            "needs_confirmation": payload.get("needs_confirmation"),
            "transcript_fix_backend": payload.get("transcript_fix_backend"),
            "raw_transcript": payload.get("raw_transcript"),
            "final_transcript": hyp,
            "segments": len(payload.get("segments") or []),
        }
    )
    if status != 200:
        errors.append(f"HTTP {status}: {payload}")

    if case.get("reference") and status == 200:
        w = wer(case["reference"], hyp)
        c = cer(case["reference"], hyp)
        row["wer"] = round(w, 4) if w is not None else None
        row["cer"] = round(c, 4) if c is not None else None

    rubric = case.get("rubric") or {}
    if "max_latency_ms" in rubric and latency_ms > float(rubric["max_latency_ms"]):
        errors.append(f"latency {latency_ms:.0f}ms > {rubric['max_latency_ms']}ms")
    if "max_rtf" in rubric and row.get("rtf") is not None and float(row["rtf"]) > float(rubric["max_rtf"]):
        errors.append(f"rtf {row['rtf']} > {rubric['max_rtf']}")
    if "max_wer" in rubric and row.get("wer") is not None and float(row["wer"]) > float(rubric["max_wer"]):
        errors.append(f"wer {row['wer']} > {rubric['max_wer']}")
    if "max_cer" in rubric and row.get("cer") is not None and float(row["cer"]) > float(rubric["max_cer"]):
        errors.append(f"cer {row['cer']} > {rubric['max_cer']}")
    if row.get("language") not in (None, "am"):
        errors.append(f"language {row.get('language')} != am")

    row["ok"] = not errors
    row["errors"] = errors
    return row


def write_markdown(report: dict[str, Any], path: Path) -> None:
    s = report["summary"]
    lines = [
        "# ASR Performance Report",
        "",
        f"- Base URL: `{report['base_url']}`",
        f"- Cases: `{s['cases']}`",
        f"- Passed: `{s['passed']}`",
        f"- Failed: `{s['failed']}`",
        f"- Mean latency: `{s['latency_ms_mean']}` ms",
        f"- p95 latency: `{s['latency_ms_p95']}` ms",
        f"- Mean RTF: `{s['rtf_mean']}`",
        f"- Mean WER: `{s['wer_mean']}`",
        f"- Mean CER: `{s['cer_mean']}`",
        "",
        "These cases are TTS loopback clips. They are repeatable smoke tests, but real farmer-call recordings are needed for final accuracy claims.",
        "",
        "| id | ok | latency_ms | audio_sec | rtf | wer | cer | confidence | transcript |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["cases"]:
        transcript = (row.get("final_transcript") or "").replace("|", " ")[:80]
        lines.append(
            "| `{id}` | `{ok}` | `{lat}` | `{dur}` | `{rtf}` | `{wer}` | `{cer}` | `{conf}` | {txt} |".format(
                id=row.get("id"),
                ok=row.get("ok"),
                lat=row.get("latency_ms"),
                dur=row.get("audio_duration_sec"),
                rtf=row.get("rtf"),
                wer=row.get("wer"),
                cer=row.get("cer"),
                conf=row.get("confidence"),
                txt=transcript,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rows = [evaluate_case(case, args.base_url, args.timeout) for case in cases]
    latencies = [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    rtfs = [float(r["rtf"]) for r in rows if isinstance(r.get("rtf"), (int, float))]
    wers = [float(r["wer"]) for r in rows if isinstance(r.get("wer"), (int, float))]
    cers = [float(r["cer"]) for r in rows if isinstance(r.get("cer"), (int, float))]
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
        "wer_mean": round(statistics.mean(wers), 4) if wers else None,
        "cer_mean": round(statistics.mean(cers), 4) if cers else None,
    }
    report = {
        "base_url": args.base_url,
        "summary": summary,
        "cases": rows,
        "notes": {
            "measured": ["latency_ms", "real_time_factor", "WER", "CER", "confidence", "language_probability"],
            "loopback_caveat": "TTS-generated audio is stable for smoke tests but does not represent noisy real farmer calls.",
        },
    }
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
