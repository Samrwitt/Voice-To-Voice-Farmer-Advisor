"""
Dynamic data layer for RAG (fast, DB-backed).

Static knowledge: vector KB chunks (rag_kb_*)
Dynamic knowledge: latest alerts + market prices from Postgres tables

This is intentionally lightweight and speed-first: small queries + short context.
"""
from __future__ import annotations

import os
from typing import Optional

import psycopg


POSTGRES_URL = os.environ.get("POSTGRES_URL", "").strip()


def _conn():
    if not POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set")
    return psycopg.connect(POSTGRES_URL)


def get_farmer_region_for_phone(phone_number: str) -> Optional[str]:
    """
    Best-effort: callers.phone_number -> caller_id -> farmer_profiles.location
    """
    if not phone_number:
        return None
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fp.location
                FROM callers c
                LEFT JOIN farmer_profiles fp ON fp.caller_id = c.caller_id
                WHERE c.phone_number = %s
                LIMIT 1;
                """,
                (phone_number,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None


def get_latest_alerts(region: Optional[str], limit: int = 2) -> list[str]:
    if not POSTGRES_URL:
        return []
    with _conn() as conn:
        with conn.cursor() as cur:
            if region:
                cur.execute(
                    """
                    SELECT alert_message
                    FROM alerts
                    WHERE target_region = %s OR target_region = 'all'
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (region, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT alert_message
                    FROM alerts
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
            return [r[0] for r in cur.fetchall() if r and r[0]]


def get_latest_market_price(crop_name: Optional[str], region: Optional[str]) -> Optional[str]:
    if not crop_name or not POSTGRES_URL:
        return None
    with _conn() as conn:
        with conn.cursor() as cur:
            if region:
                cur.execute(
                    """
                    SELECT price, unit, updated_at
                    FROM market_prices
                    WHERE crop_name = %s AND region = %s
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (crop_name, region),
                )
                row = cur.fetchone()
                if row:
                    return f"የ{crop_name} ዋጋ {row[0]} ብር በ {row[1]} ነው። (ቀን: {row[2]})"

            cur.execute(
                """
                SELECT price, unit, updated_at
                FROM market_prices
                WHERE crop_name = %s
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (crop_name,),
            )
            row = cur.fetchone()
            if row:
                return f"የ{crop_name} ዋጋ {row[0]} ብር በ {row[1]} ነው። (ቀን: {row[2]})"
    return None


def build_dynamic_context(
    phone_number: str,
    crop_name: Optional[str] = None,
    max_chars: int = 1200,
) -> str:
    """
    Returns a short, high-signal dynamic context block.
    """
    region = get_farmer_region_for_phone(phone_number)
    alerts = get_latest_alerts(region, limit=2)
    price = get_latest_market_price(crop_name, region)

    parts: list[str] = []
    if region:
        parts.append(f"አካባቢ: {region}")
    if alerts:
        parts.append("ማሳሰቢያዎች: " + " | ".join(a.strip() for a in alerts if a.strip()))
    if price:
        parts.append("ገበያ: " + price)

    text = "\n".join(parts).strip()
    return text[:max_chars]

