#!/usr/bin/env python3
"""
Regression-style checks against ``POST /rag/answer`` (voice RAG path).

Usage (from repo root):
  export RAG_BASE_URL=http://127.0.0.1:8004
  python3 eval/run_rag_eval.py

Exit code 0 = all cases passed, 1 = any failure (for CI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean, median
import urllib.error
import urllib.request
from pathlib import Path


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("golden_questions.json must be a JSON array")
    return data


def _post_rag_answer(base: str, text: str, phone: str, session: str) -> tuple[dict, float]:
    url = base.rstrip("/") + "/rag/answer"
    body = json.dumps(
        {"text": text, "phone_number": phone, "session_id": session},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode("utf-8")
    ms = (time.perf_counter() - t0) * 1000
    return json.loads(raw), ms


def _check_rubric(case_id: str, resp: dict, latency_ms: float, rubric: dict) -> list[str]:
    errs: list[str] = []
    text = (resp.get("response") or "").strip()
    trust = resp.get("trust") or {}
    meta = resp.get("meta") or {}
    scenario = meta.get("scenario") or {}
    if isinstance(scenario, dict):
        scenario_name = scenario.get("scenario")
    else:
        scenario_name = scenario

    if rubric.get("min_response_len"):
        if len(text) < int(rubric["min_response_len"]):
            errs.append(f"{case_id}: response too short ({len(text)} < {rubric['min_response_len']})")

    mx = rubric.get("max_latency_ms")
    if mx is not None and latency_ms > float(mx):
        errs.append(f"{case_id}: latency {latency_ms:.0f}ms > {mx}ms")

    any_expected = [s for s in (rubric.get("expect_substrings_any") or []) if s]
    if any_expected and not any(s in text for s in any_expected):
        errs.append(f"{case_id}: expected one substring from {any_expected!r}")

    for s in rubric.get("expect_substrings_all") or []:
        if s and s not in text:
            errs.append(f"{case_id}: expected substring missing (all): {s!r}")

    for s in rubric.get("forbid_substrings") or []:
        if s and s in text:
            errs.append(f"{case_id}: forbidden substring present: {s!r}")

    ng = rubric.get("expect_trust_grounding_not")
    if ng:
        g = trust.get("grounding")
        if g in ng:
            errs.append(f"{case_id}: trust.grounding={g!r} in forbidden {ng}")

    eg = rubric.get("expect_trust_grounding")
    if eg:
        g = trust.get("grounding")
        if g != eg:
            errs.append(f"{case_id}: trust.grounding expected {eg!r} got {g!r}")

    for key, val in (rubric.get("expect_trust_contains") or {}).items():
        if trust.get(key) != val:
            errs.append(f"{case_id}: trust.{key} expected {val!r} got {trust.get(key)!r}")

    min_refs = rubric.get("min_references")
    if min_refs is not None:
        n = len(resp.get("references") or [])
        if n < int(min_refs):
            errs.append(f"{case_id}: references {n} < {min_refs}")

    exp_scenario = rubric.get("expect_scenario")
    if exp_scenario and scenario_name != exp_scenario:
        errs.append(f"{case_id}: scenario expected {exp_scenario!r} got {scenario_name!r}")

    exp_meta_reason = rubric.get("expect_meta_reason")
    if exp_meta_reason and meta.get("reason") != exp_meta_reason:
        errs.append(f"{case_id}: meta.reason expected {exp_meta_reason!r} got {meta.get('reason')!r}")

    return errs


def _pct(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round((num / den) * 100.0, 1)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((pct / 100.0) * len(ordered) + 0.999999) - 1))
    return round(ordered[idx], 1)


def _aggregate_metrics(cases: list[dict], report: list[dict]) -> dict:
    by_id = {row.get("id"): row for row in report}

    scenario_total = 0
    scenario_ok = 0
    grounding_total = 0
    grounding_ok = 0
    grounding_not_total = 0
    grounding_not_ok = 0

    for case in cases:
        rubric = case.get("rubric") or {}
        row = by_id.get(case.get("id")) or {}

        expected_scenario = rubric.get("expect_scenario")
        if expected_scenario:
            scenario_total += 1
            scenario_ok += int(row.get("scenario") == expected_scenario)

        expected_grounding = rubric.get("expect_trust_grounding")
        if expected_grounding:
            grounding_total += 1
            grounding_ok += int(row.get("grounding") == expected_grounding)

        forbidden_grounding = rubric.get("expect_trust_grounding_not") or []
        if forbidden_grounding:
            grounding_not_total += 1
            grounding_not_ok += int(row.get("grounding") not in forbidden_grounding)

    latencies = [float(row["latency_ms"]) for row in report if isinstance(row.get("latency_ms"), (int, float))]
    passed = sum(1 for row in report if row.get("ok"))
    references = [int(row.get("references") or 0) for row in report if "references" in row]

    return {
        "cases_total": len(cases),
        "cases_passed": passed,
        "case_pass_rate_pct": _pct(passed, len(cases)),
        "scenario_accuracy_pct": _pct(scenario_ok, scenario_total),
        "scenario_cases": scenario_total,
        "expected_grounding_accuracy_pct": _pct(grounding_ok, grounding_total),
        "expected_grounding_cases": grounding_total,
        "forbidden_grounding_pass_rate_pct": _pct(grounding_not_ok, grounding_not_total),
        "forbidden_grounding_cases": grounding_not_total,
        "latency_ms_mean": round(mean(latencies), 1) if latencies else None,
        "latency_ms_p50": round(median(latencies), 1) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 95),
        "latency_ms_max": round(max(latencies), 1) if latencies else None,
        "avg_references": round(mean(references), 1) if references else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_questions.json",
    )
    ap.add_argument("--phone", default=os.getenv("EVAL_PHONE") or f"eval_bot_{int(time.time())}")
    ap.add_argument(
        "--same-phone",
        action="store_true",
        help="Reuse one phone number across all cases to test personalization carry-over.",
    )
    ap.add_argument("--session-prefix", default="eval_session")
    args = ap.parse_args()

    base = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8004").strip()
    cases = _load_cases(args.golden)
    all_errs: list[str] = []
    report: list[dict] = []

    for i, case in enumerate(cases):
        cid = case.get("id", f"case_{i}")
        q = (case.get("question") or "").strip()
        rubric = case.get("rubric") or {}
        if not q:
            all_errs.append(f"{cid}: empty question")
            continue
        session = f"{args.session_prefix}_{i}_{int(time.time())}"
        phone = args.phone if args.same_phone else f"{args.phone}_{i}"
        try:
            resp, wall_ms = _post_rag_answer(base, q, phone, session)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if e.fp else ""
            all_errs.append(f"{cid}: HTTP {e.code} {body[:300]}")
            report.append({"id": cid, "ok": False, "http_error": e.code})
            continue
        except Exception as e:
            all_errs.append(f"{cid}: {e}")
            report.append({"id": cid, "ok": False, "error": str(e)})
            continue

        errs = _check_rubric(cid, resp, wall_ms, rubric)
        all_errs.extend(errs)
        trust = resp.get("trust") or {}
        report.append(
            {
                "id": cid,
                "ok": not bool(errs),
                "latency_ms": round(wall_ms, 1),
                "trust_latency_ms": trust.get("latency_ms"),
                "grounding": trust.get("grounding"),
                "scenario": ((resp.get("meta") or {}).get("scenario") or {}).get("scenario")
                if isinstance((resp.get("meta") or {}).get("scenario"), dict)
                else (resp.get("meta") or {}).get("scenario"),
                "references": len(resp.get("references") or []),
                "response_preview": (resp.get("response") or "")[:160],
            }
        )

    metrics = _aggregate_metrics(cases, report)
    print(
        json.dumps(
            {"base_url": base, "metrics": metrics, "cases": report, "failures": all_errs},
            indent=2,
            ensure_ascii=False,
        )
    )
    if all_errs:
        print("\nFAILED:", len(all_errs), "check(s)", file=sys.stderr)
        return 1
    print("\nOK: all", len(cases), "cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
