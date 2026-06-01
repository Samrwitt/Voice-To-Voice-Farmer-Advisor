#!/usr/bin/env python3
"""Extended RAG component smoke tests (beyond golden_questions.json)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


BASE = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8004").rstrip("/")


def post(text: str, session: str, phone: str = "component_test") -> tuple[dict, float]:
    body = json.dumps(
        {"text": text, "phone_number": phone, "session_id": session},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/rag/answer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw), (time.perf_counter() - t0) * 1000


@dataclass
class Check:
    name: str
    component: str
    ok: bool
    detail: str = ""
    latency_ms: float | None = None
    extras: dict = field(default_factory=dict)


def run_case(
    name: str,
    component: str,
    question: str,
    *,
    session: str,
    phone: str = "component_test",
    expect_any: list[str] | None = None,
    expect_all: list[str] | None = None,
    forbid: list[str] | None = None,
    expect_scenario: str | None = None,
    expect_grounding: str | None = None,
    min_len: int = 0,
) -> Check:
    try:
        resp, ms = post(question, session, phone)
    except Exception as e:
        return Check(name, component, False, str(e))

    text = (resp.get("response") or "").strip()
    trust = resp.get("trust") or {}
    meta = resp.get("meta") or {}
    scenario = meta.get("scenario")
    if isinstance(scenario, dict):
        scenario = scenario.get("scenario")
    grounding = trust.get("grounding")
    errs: list[str] = []

    if min_len and len(text) < min_len:
        errs.append(f"short response ({len(text)})")
    if expect_any and not any(s in text for s in expect_any):
        errs.append(f"missing any of {expect_any}")
    if expect_all:
        for s in expect_all:
            if s not in text:
                errs.append(f"missing {s!r}")
    if forbid:
        for s in forbid:
            if s in text:
                errs.append(f"forbidden {s!r}")
    if expect_scenario and scenario != expect_scenario:
        errs.append(f"scenario={scenario!r} want {expect_scenario!r}")
    if expect_grounding and grounding != expect_grounding:
        errs.append(f"grounding={grounding!r} want {expect_grounding!r}")

    tool_trace = meta.get("tool_trace") or trust.get("tool_trace") or []
    return Check(
        name,
        component,
        not errs,
        "; ".join(errs) if errs else text[:120],
        round(ms, 1),
        {
            "grounding": grounding,
            "scenario": scenario,
            "references": len(resp.get("references") or []),
            "tool_trace": tool_trace[:3] if isinstance(tool_trace, list) else tool_trace,
        },
    )


def main() -> int:
    ts = int(time.time())
    unique_phone = f"component_{ts}"
    checks: list[Check] = []

    # NLU / greeting
    checks.append(
        run_case(
            "greeting",
            "nlu",
            "ሰላም",
            session=f"g_{ts}",
            phone=unique_phone,
            expect_any=["ግብርና", "ጥያቄ"],
            min_len=5,
        )
    )

    # KB retrieval + crop production
    checks.append(
        run_case(
            "kb_wheat",
            "retrieval_kb",
            "ስንዴ ለመዝራት የሚመከር ከፍታ ስንት ነው?",
            session=f"kb_{ts}",
            phone=f"{unique_phone}_kb",
            expect_any=["ስንዴ", "ከፍታ"],
            expect_scenario="crop_production",
            min_len=12,
        )
    )

    # Clarification (fertilizer)
    checks.append(
        run_case(
            "clarify_fertilizer",
            "clarification",
            "ማዳበሪያ እንዴት መጠቀም አለብኝ?",
            session=f"cl_{ts}",
            phone=f"{unique_phone}_cl",
            expect_any=["ሰብል", "አካባቢ"],
            expect_scenario="fertilizer",
            expect_grounding="clarification",
            min_len=10,
        )
    )

    # Market / WFP
    checks.append(
        run_case(
            "market_teff",
            "market_wfp",
            "የጤፍ ዋጋ ስንት ነው?",
            session=f"mkt_{ts}",
            expect_any=["ጤፍ", "ብር", "ዋጋ"],
            expect_scenario="market_price",
            forbid=["NMiS", "EthioSIS", "SoilGrids", "Copernicus"],
            min_len=15,
        )
    )

    # Market sell/hold
    checks.append(
        run_case(
            "market_sell_hold",
            "market_advice",
            "ጤፍ አሁን ልሽጥ ወይስ ትንሽ ልጠብቅ?",
            session=f"sell_{ts}",
            expect_any=["ሽ", "ጠብቅ", "ዋጋ", "ገበያ"],
            min_len=10,
        )
    )

    # Weather (needs location or clarifies)
    checks.append(
        run_case(
            "weather_clarify",
            "weather",
            "ዛሬ ዝናብ ይዘንቅ?",
            session=f"wx_{ts}",
            expect_any=["አካባቢ", "ከተማ", "ዝናብ", "ሁኔታ", "አየር"],
            min_len=8,
        )
    )

    checks.append(
        run_case(
            "weather_place",
            "weather",
            "በአዲስ አበባ ዛሬ ዝናብ ይዘንቅ?",
            session=f"wx2_{ts}",
            expect_any=["ዝናብ", "አዲስ", "ሁኔታ", "አየር"],
            min_len=10,
        )
    )

    # Soil (no provider names in answer)
    checks.append(
        run_case(
            "soil_ph",
            "soil",
            "በአርሲ መሬቴ አሲዳ ነው? pH ስንት ነው?",
            session=f"soil_{ts}",
            expect_any=["pH", "አሲድ", "መሬት", "አርሲ"],
            forbid=["EthioSIS", "SoilGrids", "Copernicus"],
            min_len=10,
        )
    )

    # Pest / KB
    checks.append(
        run_case(
            "pest_coffee",
            "retrieval_kb",
            "የቡና ተባዕቶ እንዴት ይቆጣጠራል?",
            session=f"pest_{ts}",
            expect_any=["ቡና", "ተባዕ", "ቆጣጠር"],
            min_len=10,
        )
    )

    # Escalation (agrochemical)
    checks.append(
        run_case(
            "agrochem_escalate",
            "escalation",
            "ጥቁር አክሲድ ለእህል ተባዕት እንዴት እጠቀም?",
            session=f"esc_{ts}",
            expect_any=["ባለሙያ", "ሙያ", "መድሐኒት", "ጥንቃቄ"],
            min_len=10,
        )
    )

    # Out-of-domain
    checks.append(
        run_case(
            "out_of_domain",
            "escalation",
            "የባንክ ብድር እንዴት አገኛለሁ?",
            session=f"ood_{ts}",
            min_len=5,
        )
    )

    # Follow-up context (same phone, sequential sessions still test routing)
    phone_ctx = f"ctx_{ts}"
    unique_phone = f"component_{ts}"
    c1 = run_case(
        "ctx_market_q1",
        "session_context",
        "የጤፍ ዋጋ በአሲይታ ስንት ነው?",
        session=f"ctx1_{ts}",
        phone=phone_ctx,
        expect_any=["ጤፍ", "ብር"],
        min_len=10,
    )
    checks.append(c1)
    checks.append(
        run_case(
            "ctx_location_followup",
            "session_context",
            "በአርሲ ነኝ",
            session=f"ctx2_{ts}",
            phone=phone_ctx,
            expect_any=["ጤፍ", "ብር", "አርሲ", "ገበያ", "አካባቢ"],
            min_len=8,
        )
    )

    by_component: dict[str, list[Check]] = {}
    for c in checks:
        by_component.setdefault(c.component, []).append(c)

    passed = sum(1 for c in checks if c.ok)
    print(json.dumps({"base_url": BASE, "passed": passed, "total": len(checks)}, indent=2))
    print("\n=== By component ===")
    for comp, rows in sorted(by_component.items()):
        ok = sum(1 for r in rows if r.ok)
        print(f"\n{comp}: {ok}/{len(rows)}")
        for r in rows:
            status = "PASS" if r.ok else "FAIL"
            print(f"  [{status}] {r.name} ({r.latency_ms}ms) — {r.detail[:100]}")
            if r.extras:
                print(f"         {r.extras}")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
