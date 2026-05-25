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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_questions.json",
    )
    ap.add_argument("--phone", default=os.getenv("EVAL_PHONE") or f"eval_bot_{int(time.time())}")
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
        try:
            resp, wall_ms = _post_rag_answer(base, q, args.phone, session)
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

    print(json.dumps({"base_url": base, "cases": report, "failures": all_errs}, indent=2, ensure_ascii=False))
    if all_errs:
        print("\nFAILED:", len(all_errs), "check(s)", file=sys.stderr)
        return 1
    print("\nOK: all", len(cases), "cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
