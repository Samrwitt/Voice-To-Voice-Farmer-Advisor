"""
One-shot migrator: copy rows from a legacy SQLite advisor.db into Postgres.

Idempotent. Skips silently when:
  - no advisor.db file is found, OR
  - the corresponding Postgres tables already have rows.

Run via main.py startup; safe to run repeatedly.
"""
import os
import sqlite3
from datetime import datetime

from sqlalchemy import select

from db import SessionLocal
from models import (
    Alert,
    CallRecord,
    ConversationMessage,
    Escalation,
    FarmerKB,
    MarketPrice,
    SessionState,
)


DATA_DIR = os.environ.get("DATA_DIR", "/data")
LEGACY_DB_PATH = os.path.join(DATA_DIR, "advisor.db")


def _parse_dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _table_exists(sqlite_conn, name: str) -> bool:
    cur = sqlite_conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _pg_table_empty(db, model) -> bool:
    return db.query(model).first() is None


def migrate_sqlite_to_postgres():
    if not os.path.exists(LEGACY_DB_PATH):
        return

    print(f"[MIGRATE] Found legacy SQLite DB at {LEGACY_DB_PATH}")
    sconn = sqlite3.connect(LEGACY_DB_PATH)
    sconn.row_factory = sqlite3.Row

    db = SessionLocal()
    moved = {}
    try:
        # ── farmers ────────────────────────────────────────────────────────────
        if _table_exists(sconn, "farmers") and _pg_table_empty(db, FarmerKB):
            rows = sconn.execute("SELECT * FROM farmers").fetchall()
            for r in rows:
                db.add(
                    FarmerKB(
                        phone_number=r["phone_number"],
                        name=r["name"],
                        location=r["location"],
                        preferred_language=r["preferred_language"] or "am",
                        registered_at=_parse_dt(r["registered_at"]) or datetime.utcnow(),
                    )
                )
            moved["farmers_kb"] = len(rows)

        # ── call_records ───────────────────────────────────────────────────────
        if _table_exists(sconn, "call_records") and _pg_table_empty(db, CallRecord):
            rows = sconn.execute("SELECT * FROM call_records").fetchall()
            for r in rows:
                db.add(
                    CallRecord(
                        session_id=r["session_id"],
                        phone_number=r["phone_number"],
                        recording_path=r["recording_path"],
                        duration=r["duration"] or 0,
                        timestamp=_parse_dt(r["timestamp"]) or datetime.utcnow(),
                    )
                )
            moved["call_records"] = len(rows)

        # ── conversation_history ───────────────────────────────────────────────
        if _table_exists(sconn, "conversation_history") and _pg_table_empty(
            db, ConversationMessage
        ):
            rows = sconn.execute("SELECT * FROM conversation_history").fetchall()
            for r in rows:
                db.add(
                    ConversationMessage(
                        phone_number=r["phone_number"],
                        session_id=r["session_id"],
                        role=r["role"],
                        message=r["message"],
                        timestamp=_parse_dt(r["timestamp"]) or datetime.utcnow(),
                    )
                )
            moved["conversation_history"] = len(rows)

        # ── escalated_queries -> escalations ───────────────────────────────────
        if _table_exists(sconn, "escalated_queries") and _pg_table_empty(db, Escalation):
            rows = sconn.execute("SELECT * FROM escalated_queries").fetchall()
            for r in rows:
                db.add(
                    Escalation(
                        query=r["query"],
                        context=r["context"],
                        status=r["status"] or "pending",
                        created_at=_parse_dt(r["timestamp"]) or datetime.utcnow(),
                    )
                )
            moved["escalations"] = len(rows)

        # ── alerts ─────────────────────────────────────────────────────────────
        if _table_exists(sconn, "alerts") and _pg_table_empty(db, Alert):
            rows = sconn.execute("SELECT * FROM alerts").fetchall()
            for r in rows:
                db.add(
                    Alert(
                        target_region=r["target_region"],
                        alert_message=r["alert_message"],
                        severity=r["severity"] or "warning",
                        created_at=_parse_dt(r["created_at"]) or datetime.utcnow(),
                    )
                )
            moved["alerts"] = len(rows)

        # ── market_prices ──────────────────────────────────────────────────────
        if _table_exists(sconn, "market_prices") and _pg_table_empty(db, MarketPrice):
            rows = sconn.execute("SELECT * FROM market_prices").fetchall()
            for r in rows:
                db.add(
                    MarketPrice(
                        crop_name=r["crop_name"],
                        region=r["region"],
                        price=r["price"],
                        unit=r["unit"],
                        updated_at=_parse_dt(r["updated_at"]) or datetime.utcnow(),
                    )
                )
            moved["market_prices"] = len(rows)

        # ── session_states ─────────────────────────────────────────────────────
        if _table_exists(sconn, "session_states") and _pg_table_empty(db, SessionState):
            rows = sconn.execute("SELECT * FROM session_states").fetchall()
            for r in rows:
                db.add(
                    SessionState(
                        session_id=r["session_id"],
                        current_state=r["current_state"],
                        pending_action=r["pending_action"],
                        updated_at=_parse_dt(r["updated_at"]) or datetime.utcnow(),
                    )
                )
            moved["session_states"] = len(rows)

        if moved:
            db.commit()
            print(f"[MIGRATE] Copied rows from SQLite -> Postgres: {moved}")
        else:
            print("[MIGRATE] Nothing to migrate (Postgres tables already populated).")
    except Exception as exc:
        db.rollback()
        print(f"[MIGRATE] Failed: {exc}")
    finally:
        db.close()
        sconn.close()
