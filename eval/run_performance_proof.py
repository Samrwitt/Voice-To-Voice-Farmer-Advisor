#!/usr/bin/env python3
"""Run the voice RAG golden eval and save a shareable performance proof report."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "eval" / "performance_proof_latest.json"
OUT_MD = ROOT / "eval" / "performance_proof_latest.md"


def _run_eval() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_rag_eval.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    start = proc.stdout.find("{")
    end = proc.stdout.rfind("}")
    if start < 0 or end < start:
        raise SystemExit(f"Could not parse eval JSON.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    report = json.loads(proc.stdout[start : end + 1])
    report["command"] = "python3 eval/run_rag_eval.py"
    report["exit_code"] = proc.returncode
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def _write_markdown(report: dict) -> None:
    metrics = report.get("metrics") or {}
    failures = report.get("failures") or []
    cases = report.get("cases") or []
    lines = [
        "# RAG Performance Proof",
        "",
        f"- Generated UTC: `{report.get('generated_at')}`",
        f"- Base URL: `{report.get('base_url')}`",
        f"- Command: `{report.get('command')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "cases_passed",
        "cases_total",
        "case_pass_rate_pct",
        "scenario_accuracy_pct",
        "expected_grounding_accuracy_pct",
        "forbidden_grounding_pass_rate_pct",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_max",
        "avg_references",
    ):
        lines.append(f"| `{key}` | `{metrics.get(key)}` |")
    lines.extend(["", "## Case Latency", "", "| Case | OK | Grounding | Scenario | Latency ms | References |", "|---|---:|---|---|---:|---:|"])
    for case in cases:
        lines.append(
            "| `{id}` | `{ok}` | `{grounding}` | `{scenario}` | `{latency_ms}` | `{references}` |".format(
                id=case.get("id"),
                ok=case.get("ok"),
                grounding=case.get("grounding"),
                scenario=case.get("scenario"),
                latency_ms=case.get("latency_ms"),
                references=case.get("references"),
            )
        )
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("None.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = _run_eval()
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "metrics": report.get("metrics")}, indent=2))
    return int(report.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
