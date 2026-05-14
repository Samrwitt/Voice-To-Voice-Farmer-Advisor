"""Postgres-backed quality snapshot (interaction mix + escalation backlog)."""

from __future__ import annotations

import os
from typing import Any

from database import POSTGRES_URL


def _pg_ok() -> bool:
    return bool((POSTGRES_URL or "").strip())


def quality_snapshot(*, window_hours: int = 24) -> dict[str, Any]:
    """
    Aggregate ``interaction_records`` and ``escalations`` for dashboards / eval gates.

    Env:
      ESCALATION_SLA_HOURS — target hours for pending expert reply (default 48).
    """
    sla_h = int(os.getenv("ESCALATION_SLA_HOURS", "48") or "48")
    retention = int(os.getenv("CALL_RECORDING_RETENTION_DAYS", "30") or "30")

    out: dict[str, Any] = {
        "ok": True,
        "window_hours": window_hours,
        "ops_alerts": [],
        "policy": {
            "call_recording_retention_days": retention,
            "escalation_sla_target_hours": sla_h,
            "note": "Tune via CALL_RECORDING_RETENTION_DAYS and ESCALATION_SLA_HOURS.",
        },
    }

    if not _pg_ok():
        out["ok"] = False
        out["error"] = "POSTGRES_URL not set"
        return out

    try:
        import psycopg
    except ImportError:
        out["ok"] = False
        out["error"] = "psycopg not installed"
        return out

    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(response_type, 'unknown'), COUNT(*)::int
                    FROM interaction_records
                    WHERE created_at >= NOW() - make_interval(hours => %s)
                    GROUP BY 1
                    ORDER BY 2 DESC;
                    """,
                    (max(1, window_hours),),
                )
                rows = cur.fetchall() or []
                out["interaction_records"] = {k: v for k, v in rows}

                cur.execute(
                    """
                    SELECT COALESCE(status, 'unknown'), COUNT(*)::int
                    FROM escalations
                    WHERE created_at >= NOW() - make_interval(hours => %s)
                    GROUP BY 1
                    ORDER BY 2 DESC;
                    """,
                    (max(1, window_hours),),
                )
                esc_rows = cur.fetchall() or []
                out["escalations"] = {k: v for k, v in esc_rows}

                cur.execute(
                    """
                    SELECT COUNT(*)::int
                    FROM escalations
                    WHERE status = 'pending'
                      AND created_at < NOW() - make_interval(hours => %s);
                    """,
                    (max(1, sla_h),),
                )
                overdue = cur.fetchone()
                out["escalations_pending_over_sla"] = int(overdue[0]) if overdue else 0

                cur.execute("SELECT COUNT(*)::int FROM escalations WHERE status = 'pending';")
                pend = cur.fetchone()
                out["escalations_pending_total"] = int(pend[0]) if pend else 0

                alerts: list[dict[str, Any]] = []
                breach = int(out.get("escalations_pending_over_sla") or 0)
                if breach > 0:
                    alerts.append(
                        {
                            "code": "escalation_sla_breach",
                            "severity": "warning",
                            "count": breach,
                            "message": (
                                f"{breach} pending escalation(s) older than "
                                f"{sla_h}h SLA target — assign or answer in the helpdesk."
                            ),
                        }
                    )
                pend_tot = int(out.get("escalations_pending_total") or 0)
                if pend_tot > 0 and breach == 0:
                    alerts.append(
                        {
                            "code": "escalation_backlog",
                            "severity": "info",
                            "count": pend_tot,
                            "message": f"{pend_tot} pending escalation(s) within SLA window.",
                        }
                    )
                out["ops_alerts"] = alerts
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
        out.setdefault("ops_alerts", [])
    return out
