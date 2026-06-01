"""Tool-routed agricultural advisory pipeline.

This module keeps routing, data retrieval, prediction, and final generation
separate. Tool-only routes such as weather and market return directly without
calling Gemini or another final LLM.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .llm_providers import effective_llm_backend, gemini_api_keys
from .query_llm import prepare_rag_llm_messages, run_sync_llm
from .rag_tools import web_search
from .source_catalog import supplemental_context_block


WEATHER_TTL_SEC = int(os.getenv("RAG_WEATHER_CACHE_TTL_SEC", "7200") or "7200")
SOIL_TTL_SEC = int(os.getenv("RAG_SOIL_CACHE_TTL_SEC", str(180 * 24 * 3600)) or str(180 * 24 * 3600))
SOIL_WATER_TTL_SEC = int(os.getenv("RAG_SOIL_WATER_CACHE_TTL_SEC", "21600") or "21600")
_weather_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_soil_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_soil_water_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_wfp_hdx_market_cache: tuple[float, list[dict[str, Any]], str] | None = None
_wfp_hdx_market_error_cache: tuple[float, str] | None = None
_wfp_hdx_clean_cache: tuple[int, int, list[dict[str, Any]], str] | None = None
_wfp_hdx_records_cache: tuple[str, str, int, int, list[dict[str, Any]]] | None = None


INTENT_MAP = {
    "market_price": "market_price",
    "soil_fertility": "fertilizer_advice",
    "pest_disease": "disease_diagnosis",
    "crop_production": "crop_recommendation",
    "soil_water_conservation": "soil_advice",
    "land_characterization": "soil_advice",
    "post_harvest": "general_agriculture",
    "weather_advice": "weather_advice",
}

INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("emergency_pest_or_disease", ("severe", "emergency", "በፍጥነት", "ተስፋፋ", "ሞተ", "አደጋ")),
    ("weather_advice", ("weather", "forecast", "rain", "ዝናብ", "የአየር", "አየር")),
    ("irrigation_advice", ("irrigat", "moisture", "soil water", "drought", "መስኖ", "ውኃ", "ውሃ", "እርጥበት", "ድርቅ")),
    ("market_price", ("price", "market", "ዋጋ", "ገበያ", "ሽያጭ")),
    ("soil_advice", ("soil", "ph", "አፈር", "መሬት")),
    ("fertilizer_advice", ("fertilizer", "urea", "dap", "compost", "ማዳበሪያ", "ዩሪያ", "ኮምፖስት")),
    ("disease_diagnosis", ("disease", "pest", "rust", "spot", "በሽታ", "ተባይ", "ሩብ", "ቅጠል")),
    ("crop_recommendation", ("what crop", "plant", "መዝራት", "ምን ልዝራ", "ሰብል")),
    ("yield_prediction", ("yield", "harvest", "ምርት", "መከር")),
)

MOCK_MARKET = {
    "teff": {"price": 7800, "previous_price": 7400, "unit": "ETB/quintal", "market": "demo", "updated_at": "demo"},
    "wheat": {"price": 4200, "previous_price": 4100, "unit": "ETB/quintal", "market": "demo", "updated_at": "demo"},
    "maize": {"price": 3200, "previous_price": 3350, "unit": "ETB/quintal", "market": "demo", "updated_at": "demo"},
    "barley": {"price": 3900, "previous_price": 3800, "unit": "ETB/quintal", "market": "demo", "updated_at": "demo"},
    "sesame": {"price": 10500, "previous_price": 9800, "unit": "ETB/quintal", "market": "demo", "updated_at": "demo"},
}

ETHIOPIA_LOCATION_COORDS: dict[str, tuple[float, float, str]] = {
    "addis ababa": (9.03, 38.74, "Addis Ababa, ET"),
    "አዲስ አበባ": (9.03, 38.74, "Addis Ababa, ET"),
    "arsi": (7.95, 39.14, "Asella/Arsi, Oromia, ET"),
    "አርሲ": (7.95, 39.14, "Asella/Arsi, Oromia, ET"),
    "oromia": (8.54, 39.27, "Adama, Oromia, ET"),
    "ኦሮሚያ": (8.54, 39.27, "Adama, Oromia, ET"),
    "amhara": (11.6, 37.38, "Bahir Dar, Amhara, ET"),
    "አማራ": (11.6, 37.38, "Bahir Dar, Amhara, ET"),
    "hawassa": (7.06, 38.48, "Hawassa, Sidama, ET"),
    "ሀዋሳ": (7.06, 38.48, "Hawassa, Sidama, ET"),
    "sidama": (7.06, 38.48, "Hawassa, Sidama, ET"),
    "ሲዳማ": (7.06, 38.48, "Hawassa, Sidama, ET"),
    "bale": (7.12, 40.0, "Bale, Oromia, ET"),
    "ባሌ": (7.12, 40.0, "Bale, Oromia, ET"),
    "jimma": (7.67, 36.83, "Jimma, Oromia, ET"),
    "ጅማ": (7.67, 36.83, "Jimma, Oromia, ET"),
    "dire dawa": (9.6, 41.86, "Dire Dawa, ET"),
    "ድሬዳዋ": (9.6, 41.86, "Dire Dawa, ET"),
    "mekelle": (13.5, 39.47, "Mekelle, Tigray, ET"),
    "መቀሌ": (13.5, 39.47, "Mekelle, Tigray, ET"),
    "gondar": (12.6, 37.47, "Gondar, Amhara, ET"),
    "ጎንደር": (12.6, 37.47, "Gondar, Amhara, ET"),
    "debre birhan": (9.68, 39.53, "Debre Birhan, Amhara, ET"),
    "ደብረ ብርሃን": (9.68, 39.53, "Debre Birhan, Amhara, ET"),
}


@dataclass
class SmartResult:
    answer: str
    context: dict[str, Any]
    references: list[dict[str, Any]]
    used_llm: bool
    tool_trace: list[dict[str, Any]]


def _norm_key(value: str | None) -> str:
    return (value or "").strip().lower()


def _known_location_coords(location: str | None) -> tuple[float, float, str] | None:
    loc = (location or "").strip()
    if not loc:
        return None
    lower = loc.lower()
    for key, coords in ETHIOPIA_LOCATION_COORDS.items():
        if key in lower or key in loc:
            return coords
    return None


def _extract_location_name_from_question(question: str) -> str | None:
    q = re.sub(r"\s+", " ", (question or "").strip())
    if not q:
        return None

    for key, (_lat, _lon, label) in ETHIOPIA_LOCATION_COORDS.items():
        if key in q.lower() or key in q:
            return label.split(",")[0]

    patterns = (
        r"(?:weather|forecast|rain|climate)\s+(?:in|for|near|at)\s+([A-Za-z][A-Za-z .'-]{1,80})",
        r"(?:in|for|near|at)\s+([A-Za-z][A-Za-z .'-]{1,80})\s+(?:weather|forecast|rain|climate)",
        r"(?:^|\s)(?:ለ|በ)\s*([\u1200-\u137f ]{2,40}?)(?:\s+የ|\s+ላይ|\s+አካባቢ|\s+ዝናብ|\s+አየር|$)",
    )
    stops = re.compile(
        r"\b(?:and|should|do|for|weather|forecast|rain|climate|irrigate|maize|wheat|teff)\b.*$",
        re.I,
    )
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        loc = stops.sub("", match.group(1)).strip(" ,.?።")
        if any(term in loc for term in ("ገበያ", "ዋጋ", "ስንት", "ነው")):
            continue
        if 2 <= len(loc) <= 80:
            return loc
    return None


def _tool_offline_fallback_enabled() -> bool:
    return os.getenv("RAG_TOOL_OFFLINE_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}


def _copernicus_offline_fallback_enabled() -> bool:
    return os.getenv("RAG_COPERNICUS_SWI_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}


def _cache_get(cache: dict[str, tuple[float, dict[str, Any]]], key: str, ttl: int) -> dict[str, Any] | None:
    row = cache.get(key)
    if not row:
        return None
    ts, payload = row
    if time.time() - ts > ttl:
        cache.pop(key, None)
        return None
    return dict(payload)


def _cache_set(cache: dict[str, tuple[float, dict[str, Any]]], key: str, payload: dict[str, Any]) -> None:
    cache[key] = (time.time(), dict(payload))


def classify_intent_and_entities(question: str, nlu: Any | None = None, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rule-based multilingual NLU with an AfroXLM-R plug-in boundary."""
    q = question or ""
    lower = q.lower()
    detected = ""
    for intent, kws in INTENT_KEYWORDS:
        if any(k in lower or k in q for k in kws):
            detected = intent
            break
    if not detected and nlu is not None:
        detected = INTENT_MAP.get(getattr(nlu, "primary_intent", ""), "general_agriculture")
    if not detected:
        detected = "general_agriculture"

    entities = dict(getattr(nlu, "entities", {}) or {})
    if entities.get("location_en") and not entities.get("location"):
        entities["location"] = entities["location_en"]
    if profile and profile.get("location") and not entities.get("location"):
        entities["location"] = profile.get("location")
    extracted_location = _extract_location_name_from_question(q)
    if extracted_location and not entities.get("location"):
        entities["location"] = extracted_location

    farm_size_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:ha|hectare|ሄክታር)", q, re.I)
    if farm_size_match:
        entities["farm_size_ha"] = float(farm_size_match.group(1))
    for key, pattern in (
        ("latitude", r"(?:lat(?:itude)?|ኬክሮስ)\s*[:=]?\s*(-?[0-9.]+)"),
        ("longitude", r"(?:lon(?:gitude)?|lng|ርዝመት)\s*[:=]?\s*(-?[0-9.]+)"),
    ):
        m = re.search(pattern, lower)
        if m:
            entities[key] = float(m.group(1))
    if "latitude" not in entities or "longitude" not in entities:
        pair = re.search(r"\b(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\b", lower)
        if pair:
            entities.setdefault("latitude", float(pair.group(1)))
            entities.setdefault("longitude", float(pair.group(2)))

    return {
        "intent": detected,
        "confidence": float(getattr(nlu, "confidence", 0.55) or 0.55),
        "entities": entities,
        "nlu_model": "rules",
        "afroxlmr_ready": True,
    }


def search_knowledge_base(question: str, hits: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return compact KB snippets with source metadata from already-ranked hits."""
    del question
    out: list[dict[str, Any]] = []
    for h in hits[: max(1, top_k)]:
        text = (h.get("content") or "").strip()
        if not text:
            continue
        out.append(
            {
                "chunk_id": h.get("chunk_id"),
                "document_id": h.get("document_id"),
                "title": h.get("title"),
                "source_org": h.get("source_org"),
                "source_url": h.get("source_url"),
                "distance": h.get("distance"),
                "snippet": text[:900],
            }
        )
    return out


def get_farmer_history(farmer_id: str, profile: dict[str, Any] | None, history_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    recent = [{"role": r, "message": m} for r, m in history_pairs[-6:] if m]
    return {
        "farmer_id": farmer_id,
        "profile": profile or {},
        "crop_type": (profile or {}).get("crop_type") or (profile or {}).get("crops"),
        "location": (profile or {}).get("location"),
        "farm_size": (profile or {}).get("farm_size"),
        "soil_type": (profile or {}).get("soil_type"),
        "planting_date": (profile or {}).get("planting_date"),
        "previous_diseases": (profile or {}).get("previous_diseases"),
        "previous_fertilizer_use": (profile or {}).get("previous_fertilizer_use"),
        "irrigation_history": (profile or {}).get("irrigation_history"),
        "yield_history": (profile or {}).get("yield_history"),
        "recent_advisory_records": recent,
    }


def _geocode(location: str) -> tuple[float, float, str] | None:
    loc = (location or "").strip()
    if not loc:
        return None
    known = _known_location_coords(loc)
    if known:
        return known
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": loc[:160], "count": 1, "language": "en", "format": "json"}
    with httpx.Client(timeout=float(os.getenv("RAG_TOOL_HTTP_TIMEOUT", "20"))) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        rows = (resp.json().get("results") or [])
    if not rows:
        return None
    row = rows[0]
    label = ", ".join(x for x in (row.get("name"), row.get("admin1"), row.get("country_code")) if x)
    return float(row["latitude"]), float(row["longitude"]), label or loc


def get_weather(location: str | None) -> dict[str, Any]:
    loc = (location or "").strip()
    if not loc:
        return {"available": False, "reason": "location_missing"}
    key = _norm_key(loc)
    cached = _cache_get(_weather_cache, key, WEATHER_TTL_SEC)
    if cached:
        cached["cache"] = "hit"
        return cached
    try:
        geo = _geocode(loc)
    except Exception as exc:
        known = _known_location_coords(loc)
        if known and _tool_offline_fallback_enabled():
            geo = known
        else:
            return {"available": False, "reason": f"geocode_error: {exc}", "location": loc, "source": "Open-Meteo"}
    if not geo:
        return {"available": False, "reason": "location_not_found", "location": loc, "source": "Open-Meteo"}
    lat, lon, label = geo
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": 7,
            "timezone": "Africa/Addis_Ababa",
        }
        with httpx.Client(timeout=float(os.getenv("RAG_TOOL_HTTP_TIMEOUT", "20"))) as client:
            resp = client.get("https://api.open-meteo.com/v1/forecast", params=params)
            resp.raise_for_status()
            data = resp.json()
        daily = data.get("daily") or {}
        rain_sum = sum(float(x or 0) for x in daily.get("precipitation_sum", [])[:7])
        payload = {
            "available": True,
            "location": label,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": (data.get("current") or {}).get("temperature_2m"),
            "humidity_pct": (data.get("current") or {}).get("relative_humidity_2m"),
            "rainfall_now_mm": (data.get("current") or {}).get("precipitation"),
            "rain_now_mm": (data.get("current") or {}).get("rain"),
            "wind_kph": (data.get("current") or {}).get("wind_speed_10m"),
            "rainfall_7d_mm": round(rain_sum, 1),
            "summary": "rain_expected" if rain_sum >= 10 else "mostly_dry",
            "source": "Open-Meteo",
            "cache": "miss",
        }
        _cache_set(_weather_cache, key, payload)
        return payload
    except Exception as exc:
        if not _tool_offline_fallback_enabled():
            return {"available": False, "reason": f"weather_error: {exc}", "source": "Open-Meteo"}
        rain_sum = 14.0 if "arsi" in label.lower() or "oromia" in label.lower() else 8.0
        payload = {
            "available": True,
            "location": label,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": 20.0,
            "humidity_pct": 65,
            "rainfall_now_mm": 0.0,
            "rain_now_mm": 0.0,
            "wind_kph": None,
            "rainfall_7d_mm": rain_sum,
            "summary": "rain_expected" if rain_sum >= 10 else "mostly_dry",
            "source": "offline_location_estimate",
            "provider_unavailable": "Open-Meteo",
            "reason": f"weather_error: {exc}",
            "cache": "fallback",
        }
        _cache_set(_weather_cache, key, payload)
        return payload


def get_soil_data(latitude: float | None, longitude: float | None) -> dict[str, Any]:
    if latitude is None or longitude is None:
        return {"available": False, "reason": "coordinates_missing"}
    key = f"{round(float(latitude), 4)},{round(float(longitude), 4)}"
    cached = _cache_get(_soil_cache, key, SOIL_TTL_SEC)
    if cached:
        cached["cache"] = "hit"
        return cached
    try:
        props = ["phh2o", "soc", "nitrogen", "clay", "sand", "silt"]
        params = {
            "lat": latitude,
            "lon": longitude,
            "property": props,
            "depth": "0-5cm",
            "value": "mean",
        }
        with httpx.Client(timeout=float(os.getenv("RAG_TOOL_HTTP_TIMEOUT", "25"))) as client:
            resp = client.get("https://rest.isric.org/soilgrids/v2.0/properties/query", params=params)
            resp.raise_for_status()
            data = resp.json()
        values: dict[str, Any] = {}
        for layer in data.get("properties", {}).get("layers", []):
            name = layer.get("name")
            depths = layer.get("depths") or []
            if not name or not depths:
                continue
            values[name] = ((depths[0].get("values") or {}).get("mean"))
        clay = values.get("clay")
        sand = values.get("sand")
        texture = "unknown"
        if sand is not None and float(sand) > 650:
            texture = "sandy"
        elif clay is not None and float(clay) > 350:
            texture = "clayey"
        elif clay is not None and sand is not None:
            texture = "loam_or_mixed"
        payload = {
            "available": True,
            "latitude": latitude,
            "longitude": longitude,
            "ph_h2o": values.get("phh2o"),
            "organic_carbon": values.get("soc"),
            "nitrogen": values.get("nitrogen"),
            "clay": values.get("clay"),
            "sand": values.get("sand"),
            "silt": values.get("silt"),
            "soil_texture": texture,
            "suitability_indicators": {
                "acidic_risk": bool(values.get("phh2o") is not None and float(values["phh2o"]) < 55),
                "low_organic_matter_risk": bool(values.get("soc") is not None and float(values["soc"]) < 120),
            },
            "source": "SoilGrids",
            "cache": "miss",
        }
        _cache_set(_soil_cache, key, payload)
        return payload
    except Exception as exc:
        if not _tool_offline_fallback_enabled():
            return {"available": False, "reason": f"soil_error: {exc}", "source": "SoilGrids"}
        lat_f = float(latitude)
        lon_f = float(longitude)
        highlandish = lat_f >= 7.0 and 35.0 <= lon_f <= 40.5
        ph_scaled = 58 if highlandish else 65
        clay = 320 if highlandish else 260
        sand = 360 if highlandish else 460
        payload = {
            "available": True,
            "latitude": latitude,
            "longitude": longitude,
            "ph_h2o": ph_scaled,
            "organic_carbon": 135 if highlandish else 95,
            "nitrogen": None,
            "clay": clay,
            "sand": sand,
            "silt": None,
            "soil_texture": "loam_or_mixed",
            "suitability_indicators": {
                "acidic_risk": ph_scaled < 55,
                "low_organic_matter_risk": not highlandish,
            },
            "source": "offline_coordinate_estimate",
            "provider_unavailable": "ISRIC SoilGrids",
            "reason": f"soil_error: {exc}",
            "cache": "fallback",
        }
        _cache_set(_soil_cache, key, payload)
        return payload


def _amharic_crop_name(crop: str | None) -> str:
    return {
        "teff": "ጤፍ",
        "wheat": "ስንዴ",
        "maize": "በቆሎ",
        "barley": "ገብስ",
        "sorghum": "ማሽላ",
        "coffee": "ቡና",
        "sesame": "ሰሊጥ",
    }.get(str(crop or "").strip().lower(), crop or "ሰብል")


def get_ethiosis_baseline(location: str | None, crop: str | None, soil: dict[str, Any]) -> dict[str, Any]:
    """EthioSIS is used as Ethiopia-specific baseline context, not a live API."""
    ph = _scaled_soil_ph(soil.get("ph_h2o")) if soil.get("available") else None
    acidic = bool((soil.get("suitability_indicators") or {}).get("acidic_risk")) if soil.get("available") else False
    low_om = bool((soil.get("suitability_indicators") or {}).get("low_organic_matter_risk")) if soil.get("available") else False
    recommendations: list[str] = []
    if acidic or (ph is not None and ph < 5.5):
        recommendations.append("አፈሩ አሲዳማ ሊሆን ስለሚችል ኖራ/liming ከአካባቢ የአፈር ምርመራ ጋር ያረጋግጡ።")
    if low_om:
        recommendations.append("ኦርጋኒክ ንጥረ ነገር ለማሻሻል ኮምፖስት፣ ፍግ ወይም የሰብል ቅሪት ይጨምሩ።")
    if crop:
        recommendations.append(f"ለ{_amharic_crop_name(crop)} የNPS/UREA መጠን ከወረዳ/ቀበሌ ምክር እና የአፈር ምርመራ ጋር ይወሰን።")
    if not recommendations:
        recommendations.append("የማዳበሪያ ውሳኔን በአፈር ምርመራ፣ በሰብል አይነት እና በአካባቢ የአፈር ካርታ ምክር ላይ ያድርጉ።")
    return {
        "available": True,
        "source": "EthioSIS baseline",
        "is_live_api": False,
        "location": location,
        "crop": crop,
        "notes": (
            "EthioSIS collected and analyzed Ethiopian kebele soil samples for fertility maps "
            "and fertilizer recommendations; use it as local baseline, not daily moisture."
        ),
        "recommendations": recommendations,
    }


def _sentinelhub_token() -> str | None:
    client_id = os.getenv("COPERNICUS_CLIENT_ID") or os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET") or os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    token_url = os.getenv(
        "COPERNICUS_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )
    with httpx.Client(timeout=float(os.getenv("RAG_TOOL_HTTP_TIMEOUT", "25"))) as client:
        resp = client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        resp.raise_for_status()
        return resp.json().get("access_token")


def get_copernicus_soil_water_index(latitude: float | None, longitude: float | None, weather: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch or estimate updated soil moisture signal from Copernicus SWI context."""
    if latitude is None or longitude is None:
        return {"available": False, "reason": "coordinates_missing", "source": "Copernicus Soil Water Index"}
    key = f"{round(float(latitude), 4)},{round(float(longitude), 4)}"
    cached = _cache_get(_soil_water_cache, key, SOIL_WATER_TTL_SEC)
    if cached:
        cached["cache"] = "hit"
        return cached

    collection_id = os.getenv("COPERNICUS_SWI_COLLECTION_ID", "f2278442-eb7f-4926-93e9-7a382f567fb4")
    token = None
    try:
        token = _sentinelhub_token()
    except Exception as exc:
        token_error = str(exc)
    else:
        token_error = None

    if token:
        # The live SWI connector is intentionally conservative. The CDSE SWI data
        # is a Sentinel Hub BYOC collection; deployments with credentials can add
        # a Statistical API evalscript without changing the caller contract below.
        return {
            "available": False,
            "reason": "copernicus_swi_statistical_adapter_not_enabled",
            "source": "Copernicus Soil Water Index",
            "collection_id": collection_id,
            "access": "CDSE Sentinel Hub BYOC",
            "cache": "miss",
        }

    if not _copernicus_offline_fallback_enabled():
        return {
            "available": False,
            "reason": token_error or "copernicus_credentials_missing",
            "source": "Copernicus Soil Water Index",
            "collection_id": collection_id,
        }

    rain7 = float((weather or {}).get("rainfall_7d_mm") or 0)
    if rain7 >= 20:
        swi_estimate, status, irrigation = 72, "wet", "delay_irrigation"
    elif rain7 >= 8:
        swi_estimate, status, irrigation = 48, "moderate", "monitor"
    else:
        swi_estimate, status, irrigation = 28, "dry", "irrigate_if_crop_stressed"
    payload = {
        "available": True,
        "source": "Copernicus SWI fallback estimate",
        "provider_unavailable": "Copernicus Soil Water Index",
        "reason": token_error or "copernicus_credentials_missing",
        "collection_id": collection_id,
        "latitude": latitude,
        "longitude": longitude,
        "swi_percent": swi_estimate,
        "status": status,
        "irrigation_signal": irrigation,
        "rainfall_7d_mm_used": rain7,
        "cache": "fallback",
    }
    _cache_set(_soil_water_cache, key, payload)
    return payload


def get_market_price(
    crop: str | None,
    location_or_market: str | None,
    local_price_func: Any | None = None,
) -> dict[str, Any]:
    crop_key = _norm_key(crop)
    if not crop_key:
        return {"available": False, "reason": "crop_missing"}
    wfp_hdx = _fetch_wfp_hdx_market_price(crop, location_or_market)
    if wfp_hdx.get("available"):
        return wfp_hdx
    nmis = _fetch_configured_nmis_market_price(crop, location_or_market)
    if nmis.get("available"):
        return nmis
    row = None
    location_matched = False
    if local_price_func is not None:
        try:
            if location_or_market:
                row = local_price_func(crop, location_or_market)
                location_matched = bool(row)
            if not row:
                row = local_price_func(crop)
        except Exception:
            row = None
    if row:
        price, unit, updated_at = row
        return {
            "available": True,
            "crop": crop,
            "market": location_or_market if location_matched else "general",
            "requested_market": location_or_market,
            "price": price,
            "previous_price": None,
            "unit": unit,
            "trend": "unknown",
            "updated_at": str(updated_at),
            "source": "local_database",
            "personalized": bool(location_or_market and location_matched),
            "needs_location_for_personal_price": not bool(location_or_market),
            "location_price_unavailable": bool(location_or_market and not location_matched),
            "selling_recommendation": "Use local extension/market confirmation before selling large volume.",
        }
    demo = MOCK_MARKET.get(crop_key) or MOCK_MARKET.get(crop_key.replace(" ", "_"))
    if not demo:
        return {"available": False, "reason": "not_in_local_or_demo_market_data", "crop": crop}
    trend = "up" if demo["price"] > demo["previous_price"] else "down" if demo["price"] < demo["previous_price"] else "flat"
    return {
        "available": True,
        "crop": crop,
        **demo,
        "market": location_or_market or demo.get("market") or "demo",
        "requested_market": location_or_market,
        "trend": trend,
        "source": "mock_demo_data",
        "personalized": False,
        "needs_location_for_personal_price": not bool(location_or_market),
        "location_price_unavailable": bool(location_or_market),
        "selling_recommendation": "sell_gradually" if trend == "up" else "avoid_rushed_sale",
    }


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
        low = name.lower()
        if low in lower_map and lower_map[low] not in (None, ""):
            return lower_map[low]
    return None


_WFP_HDX_PACKAGE_ID = "2e4f1922-e446-4b57-a98a-d0e2d5e34afa"
_WFP_HDX_DEFAULT_LOCAL_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "wfp_food_prices_eth.csv")
)
_WFP_CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "teff": ("teff", "teff white", "teff mixed", "teff red", "ጤፍ"),
    "wheat": ("wheat", "wheat grain", "ስንዴ"),
    "maize": ("maize", "maize white", "maize grain", "corn", "በቆሎ"),
    "barley": ("barley", "ገብስ"),
    "sorghum": ("sorghum", "ማሽላ"),
    "sesame": ("sesame", "ሰሊጥ"),
    "coffee": ("coffee", "ቡና"),
}


def _market_source_enabled() -> bool:
    return os.getenv("WFP_HDX_MARKET_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\u1200-\u137f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _row_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    value = _first_present(row, names)
    return str(value or "").strip()


def _parse_market_date(value: Any) -> tuple[int, int, int]:
    text = str(value or "").strip()
    if not text:
        return (0, 0, 0)
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(text[:10] if "%d" in fmt else text[:7] if fmt == "%Y-%m" else text, fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    nums = [int(x) for x in re.findall(r"\d+", text)[:3]]
    if len(nums) >= 3 and nums[0] > 1900:
        return (nums[0], nums[1], nums[2])
    if len(nums) >= 2 and nums[0] > 1900:
        return (nums[0], nums[1], 1)
    return (0, 0, 0)


def _parse_market_datetime(value: Any) -> datetime | None:
    y, m, d = _parse_market_date(value)
    if y <= 0 or m <= 0:
        return None
    try:
        return datetime(y, m, max(1, d or 1), tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_price_value(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _discover_wfp_hdx_csv_url() -> str:
    direct = os.getenv("WFP_HDX_MARKET_CSV_URL", "").strip()
    if direct:
        return direct
    package_id = os.getenv("WFP_HDX_PACKAGE_ID", _WFP_HDX_PACKAGE_ID).strip() or _WFP_HDX_PACKAGE_ID
    api_url = os.getenv(
        "WFP_HDX_PACKAGE_API_URL",
        f"https://data.humdata.org/api/3/action/package_show?id={package_id}",
    )
    with httpx.Client(timeout=float(os.getenv("WFP_HDX_TIMEOUT_SEC", "20") or "20")) as client:
        resp = client.get(api_url)
        resp.raise_for_status()
        payload = resp.json()
    resources = ((payload or {}).get("result") or {}).get("resources") or []
    for res in resources:
        name = str(res.get("name") or "").lower()
        fmt = str(res.get("format") or "").lower()
        url = str(res.get("url") or "").strip()
        if url and fmt == "csv" and "food prices" in name:
            return url
    for res in resources:
        url = str(res.get("url") or "").strip()
        fmt = str(res.get("format") or "").lower()
        if url and fmt == "csv":
            return url
    raise RuntimeError("wfp_hdx_csv_resource_not_found")


def _local_wfp_hdx_csv_path() -> str:
    return os.getenv("WFP_HDX_MARKET_CSV_PATH", _WFP_HDX_DEFAULT_LOCAL_CSV).strip()


def _load_wfp_hdx_market_rows() -> tuple[list[dict[str, Any]], str]:
    global _wfp_hdx_market_cache, _wfp_hdx_market_error_cache
    ttl = int(os.getenv("WFP_HDX_MARKET_CACHE_TTL_SEC", "21600") or "21600")
    if _wfp_hdx_market_cache and time.time() - _wfp_hdx_market_cache[0] <= ttl:
        return _wfp_hdx_market_cache[1], _wfp_hdx_market_cache[2]

    local_path = _local_wfp_hdx_csv_path()
    if local_path and os.path.exists(local_path):
        csv_url = local_path
        with open(local_path, "r", encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
    else:
        csv_url = _discover_wfp_hdx_csv_url()
        with httpx.Client(timeout=float(os.getenv("WFP_HDX_TIMEOUT_SEC", "40") or "40"), follow_redirects=True) as client:
            resp = client.get(csv_url)
            resp.raise_for_status()
            text = resp.text
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        # HDX/WFP CSVs may include a second HXL tag row beginning with "#".
        if any(str(v or "").startswith("#") for v in row.values()):
            continue
        rows.append(row)
    _wfp_hdx_market_cache = (time.time(), rows, csv_url)
    _wfp_hdx_market_error_cache = None
    return rows, csv_url


def _clean_wfp_hdx_market_rows(rows: list[dict[str, Any]], csv_source: str) -> list[dict[str, Any]]:
    global _wfp_hdx_clean_cache
    cache_key = (id(rows), len(rows))
    if (
        _wfp_hdx_clean_cache
        and _wfp_hdx_clean_cache[0] == cache_key[0]
        and _wfp_hdx_clean_cache[1] == cache_key[1]
        and _wfp_hdx_clean_cache[3] == csv_source
    ):
        return list(_wfp_hdx_clean_cache[2])

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if any(str(v or "").startswith("#") for v in row.values()):
            continue
        date_text = _row_text(row, ("date", "mp_year_month", "month", "updated_at"))
        dt = _parse_market_datetime(date_text)
        price = _parse_price_value(_first_present(row, ("price", "mp_price", "value", "amount")))
        if not dt or price is None:
            continue
        market = _row_text(row, ("market", "mkt_name", "market_name"))
        commodity = _row_text(row, ("commodity", "cm_name", "product", "item", "crop", "crop_name"))
        if not market or not commodity:
            continue
        admin1 = _row_text(row, ("admin1", "adm1_name", "region"))
        admin2 = _row_text(row, ("admin2", "adm2_name", "zone", "district"))
        cleaned.append(
            {
                "date": dt.date().isoformat(),
                "date_ts": dt.timestamp(),
                "market": market,
                "admin1": admin1,
                "admin2": admin2,
                "market_context": " ".join(p for p in (market, admin2, admin1) if p),
                "commodity": commodity,
                "unit": _row_text(row, ("unit", "um_name", "price_unit", "uom")) or "unit",
                "price_type": _row_text(row, ("pricetype", "price_type", "pt_name")) or "unknown",
                "price_flag": _row_text(row, ("priceflag", "price_flag")) or "unknown",
                "currency": _row_text(row, ("currency", "cur_name")) or "ETB",
                "price": price,
                "usd_price": _parse_price_value(_first_present(row, ("usdprice", "usd_price"))),
                "latitude": _parse_price_value(_first_present(row, ("latitude", "lat"))),
                "longitude": _parse_price_value(_first_present(row, ("longitude", "lon", "lng"))),
            }
        )
    cleaned.sort(key=lambda r: float(r.get("date_ts") or 0), reverse=True)
    _wfp_hdx_clean_cache = (cache_key[0], cache_key[1], cleaned, csv_source)
    return list(cleaned)


def _crop_matches_wfp(row_commodity: str, crop: str) -> bool:
    commodity = _normalize_match_text(row_commodity)
    crop_key = _norm_key(crop)
    aliases = _WFP_CROP_ALIASES.get(crop_key, (crop_key,))
    return any(
        alias and (_normalize_match_text(alias) in commodity or commodity in _normalize_match_text(alias))
        for alias in aliases
    )


def _location_matches_wfp(row: dict[str, Any], location_or_market: str | None) -> bool:
    if not location_or_market:
        return True
    needle = str(location_or_market).strip().lower()
    if not needle:
        return True
    fields = (
        "market",
        "market_name",
        "admin1",
        "admin2",
        "adm1_name",
        "adm2_name",
        "region",
        "location",
    )
    haystack = " ".join(_row_text(row, (f,)) for f in fields).lower()
    location_aliases = {
        "arsi": ("arsi", "asella"),
        "አርሲ": ("arsi", "asella", "አርሲ"),
        "oromia": ("oromia", "ኦሮሚያ"),
        "sidama": ("sidama", "hawassa", "ሲዳማ", "ሀዋሳ"),
        "amhara": ("amhara", "bahir dar", "አማራ"),
    }.get(needle, (needle,))
    return any(alias in haystack for alias in location_aliases)


def _similarity(needle: str, haystack: str) -> float:
    n = _normalize_match_text(needle)
    h = _normalize_match_text(haystack)
    if not n or not h:
        return 0.0
    if n == h:
        return 1.0
    if n in h or h in n:
        return 0.92
    n_tokens = set(n.split())
    h_tokens = set(h.split())
    token_score = (len(n_tokens & h_tokens) / max(len(n_tokens), 1)) if n_tokens else 0.0
    seq_score = difflib.SequenceMatcher(None, n, h).ratio()
    return max(token_score, seq_score)


def _commodity_match_score(record: dict[str, Any], crop: str) -> float:
    commodity = str(record.get("commodity") or "")
    crop_key = _norm_key(crop)
    aliases = _WFP_CROP_ALIASES.get(crop_key, (crop_key,))
    return max((_similarity(alias, commodity) for alias in aliases if alias), default=0.0)


def _location_match_score(record: dict[str, Any], location_or_market: str | None) -> float:
    if not location_or_market:
        return 1.0
    location = str(location_or_market)
    candidates = (
        str(record.get("market") or ""),
        str(record.get("admin2") or ""),
        str(record.get("admin1") or ""),
        str(record.get("market_context") or ""),
    )
    aliases = {
        "arsi": ("arsi", "asella"),
        "አርሲ": ("arsi", "asella", "አርሲ"),
        "oromia": ("oromia", "ኦሮሚያ"),
        "sidama": ("sidama", "hawassa", "ሲዳማ", "ሀዋሳ"),
        "amhara": ("amhara", "bahir dar", "አማራ"),
    }.get(_normalize_match_text(location), (location,))
    return max((_similarity(alias, candidate) for alias in aliases for candidate in candidates), default=0.0)


def _find_wfp_hdx_market_match(
    records: list[dict[str, Any]],
    crop: str,
    location_or_market: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return None, [], {"quality": "none", "approximate": False, "reason": "wfp_hdx_empty"}

    scored: list[tuple[float, float, float, dict[str, Any]]] = []
    for record in records:
        commodity_score = _commodity_match_score(record, crop)
        if commodity_score < 0.35:
            continue
        location_score = _location_match_score(record, location_or_market)
        if location_or_market and location_score < 0.25:
            # Keep weak location matches only if the commodity is very strong; this
            # allows an approximate national fallback when a local market is absent.
            location_score = 0.0
        recency = float(record.get("date_ts") or 0.0)
        total = (commodity_score * 0.64) + (location_score * 0.31) + (min(recency / time.time(), 1.0) * 0.05)
        scored.append((total, commodity_score, location_score, record))
    if not scored:
        return None, [], {"quality": "none", "approximate": False, "reason": "wfp_hdx_no_close_commodity"}

    scored.sort(key=lambda item: (item[0], item[1], item[2], float(item[3].get("date_ts") or 0.0)), reverse=True)
    _total, commodity_score, location_score, best = scored[0]
    exact_commodity = commodity_score >= 0.88
    exact_location = bool(not location_or_market or location_score >= 0.88)
    same_pair_history = [
        r
        for _score, cscore, lscore, r in scored
        if cscore >= max(0.78, commodity_score - 0.05)
        and (not location_or_market or lscore >= max(0.55, location_score - 0.05))
        and _normalize_match_text(r.get("commodity")) == _normalize_match_text(best.get("commodity"))
        and _normalize_match_text(r.get("market")) == _normalize_match_text(best.get("market"))
        and str(r.get("price_type") or "").lower() == str(best.get("price_type") or "").lower()
    ]
    same_pair_history.sort(key=lambda r: float(r.get("date_ts") or 0.0), reverse=True)
    quality = "exact" if exact_commodity and exact_location else "approximate"
    if location_or_market and location_score == 0.0 and exact_commodity:
        quality = "commodity_only"
    return best, same_pair_history, {
        "quality": quality,
        "approximate": quality != "exact",
        "commodity_score": round(commodity_score, 3),
        "location_score": round(location_score, 3),
    }


def _trend_from_wfp_history(history: list[dict[str, Any]]) -> tuple[str, float | None, str | None]:
    if len(history) < 2:
        return "unknown", None, None
    latest = _parse_price_value(history[0].get("price"))
    latest_date = str(history[0].get("date") or "")
    previous_row = next(
        (
            r
            for r in history[1:]
            if _parse_price_value(r.get("price")) is not None
            and str(r.get("date") or "") != latest_date
        ),
        None,
    )
    if latest is None or not previous_row:
        return "unknown", None, None
    previous = _parse_price_value(previous_row.get("price"))
    if previous is None or previous == 0:
        return "unknown", previous, previous_row.get("date")
    pct = ((latest - previous) / previous) * 100.0
    if pct > 2.0:
        trend = "up"
    elif pct < -2.0:
        trend = "down"
    else:
        trend = "flat"
    return trend, previous, previous_row.get("date")


def _market_recency(date_text: str | None) -> dict[str, Any]:
    dt = _parse_market_datetime(date_text)
    if not dt:
        return {"age_days": None, "is_stale": True, "recency_status": "unknown_date"}
    now = datetime.now(timezone.utc)
    age_days = max(0, int((now - dt).total_seconds() // 86400))
    max_age = int(os.getenv("WFP_HDX_MARKET_MAX_AGE_DAYS", "90") or "90")
    return {
        "age_days": age_days,
        "is_stale": age_days > max_age,
        "recency_status": "stale" if age_days > max_age else "recent_enough",
        "max_age_days": max_age,
    }


def _fetch_wfp_hdx_market_price(crop: str | None, location_or_market: str | None) -> dict[str, Any]:
    global _wfp_hdx_market_error_cache
    if not _market_source_enabled() or not crop:
        return {"available": False, "reason": "wfp_hdx_disabled_or_crop_missing"}
    error_ttl = int(os.getenv("WFP_HDX_MARKET_ERROR_TTL_SEC", "600") or "600")
    if _wfp_hdx_market_error_cache and time.time() - _wfp_hdx_market_error_cache[0] <= error_ttl:
        return {"available": False, "reason": _wfp_hdx_market_error_cache[1]}
    try:
        rows, csv_url = _load_wfp_hdx_market_rows()
    except Exception as exc:
        reason = f"wfp_hdx_error: {exc}"
        _wfp_hdx_market_error_cache = (time.time(), reason)
        return {"available": False, "reason": reason}

    records = _wfp_hdx_market_records(rows, csv_url)
    row, history, match = _find_wfp_hdx_market_match(records, crop, location_or_market)
    if not row:
        return {"available": False, "reason": match.get("reason") or "wfp_hdx_no_match", "crop": crop}
    trend, previous_price, previous_date = _trend_from_wfp_history(history)
    price = row.get("price")
    currency = str(row.get("currency") or "ETB")
    unit = str(row.get("unit") or "unit")
    market = str(row.get("market") or location_or_market or "")
    date = str(row.get("date") or "unknown")
    approximate = bool(match.get("approximate"))
    recency = _market_recency(date)
    return {
        "available": True,
        "crop": crop,
        "matched_commodity": row.get("commodity"),
        "market": market,
        "matched_admin1": row.get("admin1"),
        "matched_admin2": row.get("admin2"),
        "requested_market": location_or_market,
        "price": price,
        "previous_price": previous_price,
        "previous_date": previous_date,
        "unit": f"{currency}/{unit}" if currency and str(currency).upper() not in str(unit).upper() else unit,
        "price_type": row.get("price_type"),
        "price_flag": row.get("price_flag"),
        "trend": trend,
        "updated_at": str(date),
        "source": "wfp_hdx",
        "source_url": csv_url,
        **recency,
        "match_quality": match.get("quality"),
        "approximate": approximate,
        "commodity_match_score": match.get("commodity_score"),
        "location_match_score": match.get("location_score"),
        "personalized": bool(location_or_market and not approximate),
        "needs_location_for_personal_price": not bool(location_or_market),
        "location_price_unavailable": bool(location_or_market and approximate),
        "selling_recommendation": "Use local extension/market confirmation before selling large volume.",
    }


def _wfp_hdx_market_year_filter() -> str:
    return os.getenv("WFP_HDX_MARKET_YEAR_FILTER", "2026").strip()


def _filter_wfp_hdx_records_by_year(records: list[dict[str, Any]], year: str) -> list[dict[str, Any]]:
    if not year or year.lower() in {"all", "0", "none", "off"}:
        return records
    prefix = f"{year}-"
    filtered = [r for r in records if str(r.get("date") or "").startswith(prefix)]
    return filtered or records


def _wfp_hdx_market_records(rows: list[dict[str, Any]], csv_url: str) -> list[dict[str, Any]]:
    global _wfp_hdx_records_cache
    year = _wfp_hdx_market_year_filter()
    cache_key = (csv_url, year, len(rows), id(rows))
    if (
        _wfp_hdx_records_cache
        and _wfp_hdx_records_cache[0] == cache_key[0]
        and _wfp_hdx_records_cache[1] == cache_key[1]
        and _wfp_hdx_records_cache[2] == cache_key[2]
        and _wfp_hdx_records_cache[3] == cache_key[3]
    ):
        return _wfp_hdx_records_cache[4]
    records = _filter_wfp_hdx_records_by_year(_clean_wfp_hdx_market_rows(rows, csv_url), year)
    _wfp_hdx_records_cache = (cache_key[0], cache_key[1], cache_key[2], cache_key[3], records)
    return records


def warm_wfp_hdx_market_cache() -> dict[str, Any]:
    """Load, clean, and year-filter WFP/HDX market rows once at service startup."""
    if not _market_source_enabled():
        return {"enabled": False}
    rows, csv_url = _load_wfp_hdx_market_rows()
    records = _wfp_hdx_market_records(rows, csv_url)
    year = _wfp_hdx_market_year_filter()
    return {
        "enabled": True,
        "csv_source": csv_url,
        "raw_rows": len(rows),
        "records": len(records),
        "year_filter": year or "all",
    }


def _fetch_configured_nmis_market_price(crop: str | None, location_or_market: str | None) -> dict[str, Any]:
    """Optional live market adapter. It is inactive unless NMIS_MARKET_API_URL is configured."""
    api_url = os.getenv("NMIS_MARKET_API_URL", "").strip()
    if not api_url or not crop:
        return {"available": False, "reason": "nmis_api_not_configured"}
    try:
        url = api_url.format(crop=crop, market=location_or_market or "", location=location_or_market or "")
        headers = {}
        if os.getenv("NMIS_MARKET_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('NMIS_MARKET_API_KEY')}"
        with httpx.Client(timeout=float(os.getenv("NMIS_MARKET_TIMEOUT_SEC", "12") or "12")) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return {"available": False, "reason": f"nmis_error: {exc}"}

    rows = payload
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("results") or payload.get("items") or [payload]
    if not isinstance(rows, list):
        return {"available": False, "reason": "nmis_response_not_list"}

    crop_l = str(crop).strip().lower()
    loc_l = str(location_or_market or "").strip().lower()
    best: dict[str, Any] | None = None
    for item in rows:
        if not isinstance(item, dict):
            continue
        row_crop = str(_first_present(item, ("crop", "crop_name", "commodity", "product", "item")) or "").lower()
        row_market = str(_first_present(item, ("market", "market_name", "location", "region", "place")) or "")
        if row_crop and crop_l not in row_crop and row_crop not in crop_l:
            continue
        if loc_l and row_market and loc_l not in row_market.lower() and row_market.lower() not in loc_l:
            continue
        best = item
        break
    if not best:
        return {"available": False, "reason": "nmis_no_matching_price", "crop": crop, "market": location_or_market}

    price = _first_present(best, ("price", "price_etb", "value", "amount", "avg_price", "average_price"))
    if price in (None, ""):
        return {"available": False, "reason": "nmis_price_missing", "crop": crop, "market": location_or_market}
    unit = _first_present(best, ("unit", "price_unit", "uom")) or "ETB"
    updated_at = _first_present(best, ("updated_at", "date", "market_day", "created_at")) or "unknown"
    market = _first_present(best, ("market", "market_name", "location", "region", "place")) or location_or_market
    return {
        "available": True,
        "crop": crop,
        "market": market,
        "requested_market": location_or_market,
        "price": price,
        "previous_price": _first_present(best, ("previous_price", "prev_price")),
        "unit": unit,
        "trend": str(_first_present(best, ("trend",)) or "unknown").lower(),
        "updated_at": str(updated_at),
        "source": "nmis_api",
        "personalized": bool(location_or_market and market),
        "needs_location_for_personal_price": not bool(location_or_market),
        "selling_recommendation": "Use local extension/market confirmation before selling large volume.",
    }


def predict_agricultural_advice(
    farmer_profile: dict[str, Any],
    weather: dict[str, Any],
    soil: dict[str, Any],
    soil_water: dict[str, Any],
    market: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    q = (question or "").lower()
    rain7 = float(weather.get("rainfall_7d_mm") or 0) if weather.get("available") else 0.0
    humidity = float(weather.get("humidity_pct") or 0) if weather.get("available") and weather.get("humidity_pct") is not None else 0.0
    temp = float(weather.get("temperature_c") or 0) if weather.get("available") and weather.get("temperature_c") is not None else 0.0
    irrigation = "needed_soon" if rain7 < 8 else "delay_irrigation"
    if soil_water.get("available"):
        sw_status = soil_water.get("status")
        if sw_status == "dry":
            irrigation = "needed_soon"
        elif sw_status == "wet":
            irrigation = "delay_irrigation"
    disease = "high" if humidity >= 75 and 15 <= temp <= 28 else "medium" if humidity >= 60 else "low"
    acidic = bool((soil.get("suitability_indicators") or {}).get("acidic_risk")) if soil.get("available") else False
    low_om = bool((soil.get("suitability_indicators") or {}).get("low_organic_matter_risk")) if soil.get("available") else False
    fertilizer = "soil_test_first"
    if low_om:
        fertilizer = "add_compost_or_manure_and_confirm_with_soil_test"
    if acidic:
        fertilizer = "check_liming_need_before_high_fertilizer_rates"
    market_rec = market.get("selling_recommendation") if market.get("available") else "market_data_missing"
    return {
        "irrigation_need": irrigation,
        "disease_risk": disease,
        "fertilizer_recommendation": fertilizer,
        "yield_risk": "higher" if irrigation == "needed_soon" or disease == "high" else "normal",
        "market_selling_recommendation": market_rec,
        "crop_suitability": "needs_local_validation" if "what crop" in q or "ምን" in q else "not_assessed",
        "method": "rules_v1",
        "ml_ready": "RandomForest/XGBoost can replace rules behind this function.",
        "profile_used": bool(farmer_profile),
    }


def web_search_fallback(query: str, kb_results: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    if os.getenv("RAG_WEB_ALLOW", "1").strip().lower() in ("0", "false", "no", "off"):
        return []
    recent_terms = ("today", "latest", "current", "የዛሬ", "አዲስ", "አሁን")
    sparse = len(kb_results) < int(os.getenv("RAG_WEB_MIN_KB_HITS", "2") or "2")
    recent = any(t in (query or "").lower() or t in query for t in recent_terms)
    if not sparse and not recent and intent not in {"market_price", "weather_advice"}:
        return []
    snippets = web_search.fetch_web_snippets(query, max_results=int(os.getenv("RAG_WEB_MAX_RESULTS", "4") or "4"))
    return snippets[:4]


def _compact(obj: Any, max_chars: int = 9000) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n...truncated for Gemini cost control..."


def _final_system_prompt(language_hint: str = "farmer language") -> str:
    del language_hint
    return (
        "You are an agricultural extension advisor for Ethiopian farmers. "
        "Answer in the farmer's language when possible. Be practical, concise, and step-by-step for voice. "
        "Use local KB and farmer history first, then weather, soil, market, prediction, and web context. "
        "Do not invent missing data. Do not add generic missing-data caveats for topics the farmer did not ask about. "
        "Mention missing data only when it is essential to safely answer the exact question. "
        "Do not repeat the user's question. Do not mention source names in the spoken answer unless the user asks. "
        "Do not end with generic 'consult an expert' advice unless the route is an escalation or the context says the issue is unsafe. "
        "Treat supplemental_sources as lower-priority signposts for relevant datasets, not as direct facts unless retrieved context includes details. "
        "Avoid Markdown formatting such as **bold**, tables, and long numbered outlines. "
        "Sound like a calm, capable assistant: direct, respectful, and not robotic."
    )


_PROVIDER_NAME_PATTERNS = (
    r"EthioSIS",
    r"SoilGrids",
    r"Copernicus",
    r"Open-Meteo",
    r"ISRIC",
    r"NMiS",
    r"HDX",
    r"wfp_hdx",
)


def strip_provider_names_from_voice(text: str) -> str:
    """Remove data-provider labels from spoken farmer answers."""
    out = (text or "").strip()
    if not out:
        return ""
    for pat in _PROVIDER_NAME_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(
        r"የአየር መረጃው ከ\s*ነው።",
        "የአየር ሁኔታው እንደሚከተለው ነው።",
        out,
    )
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+።", "።", out)
    return out.strip()


def sanitize_voice_advice(answer: str, question: str) -> str:
    """Remove LLM boilerplate that is unhelpful for voice answers."""
    text = strip_provider_names_from_voice(answer)
    if not text:
        return ""

    # TTS should not read markdown markers aloud.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)

    q = (question or "").lower()
    asked_soil_or_fertilizer = any(
        k in q or k in question
        for k in ("soil", "fertilizer", "ph", "አፈር", "መሬት", "ማዳበሪያ", "ፒኤች")
    )
    if not asked_soil_or_fertilizer:
        noisy_patterns = (
            r"\s*ስለ\s+አፈር\s+አይነትና\s+ማዳበሪያ\s+አጠቃቀም\s+መረጃ\s+የለኝም።\s*መረጃው\s+ካለኝ\s+የተሻለ\s+ምክር\s+ልሰጥ\s+እችላለሁ።\s*$",
            r"\s*ስለ\s+አፈር\s+አይነት.*?የተሻለ\s+ምክር\s+ልሰጥ\s+እችላለሁ።\s*$",
        )
        for pat in noisy_patterns:
            text = re.sub(pat, "", text, flags=re.DOTALL)

    return text.strip()


def _simple_market_answer(market: dict[str, Any], language: str) -> str:
    if not market.get("available"):
        return ""
    crop = market.get("crop") or "crop"
    crop_am = _amharic_crop_name(crop)
    trend_am = {
        "up": "እየጨመረ",
        "down": "እየቀነሰ",
        "flat": "ተመሳሳይ",
        "unknown": "ያልታወቀ",
    }.get(str(market.get("trend") or "").lower(), "ያልታወቀ")
    recommendation_am = {
        "up": "ዋጋው እየጨመረ ስለሆነ ካልቸኮሉ በከፊል መሸጥ እና ቀሪውን መጠበቅ ይችላሉ።",
        "down": "ዋጋው እየቀነሰ ከሆነ ብዙ ጊዜ ከመጠበቅ በፊት የአካባቢ ገበያን ያረጋግጡ።",
        "flat": "ዋጋው ተመሳሳይ ስለሆነ የገንዘብ ፍላጎትዎን እና የማከማቻ አቅምዎን ተመልከቱ።",
        "unknown": "ከመሸጥዎ በፊት የአካባቢ ገበያ ዋጋን ያረጋግጡ።",
    }.get(str(market.get("trend") or "").lower(), "ከመሸጥዎ በፊት የአካባቢ ገበያ ዋጋን ያረጋግጡ።")
    recommendation_en = {
        "up": "Because the price is rising, consider selling gradually if you are not under pressure.",
        "down": "Because the price is falling, check the local market before waiting too long.",
        "flat": "Because the price is stable, decide based on cash need and storage capacity.",
        "unknown": "Confirm the local market price before selling a large amount.",
    }.get(str(market.get("trend") or "").lower(), "Confirm the local market price before selling a large amount.")
    certainty_am = ""
    certainty_en = ""
    stale_am = ""
    stale_en = ""
    if market.get("source") == "mock_demo_data":
        certainty_am = " ቀጥታ የገበያ ዋጋ አልተገኘም፤ ይህ በስርዓቱ ያለ አጠቃላይ/የሙከራ ዋጋ ነው።"
        certainty_en = " Live market price was not available; this is a general/demo value from the system."
    elif market.get("source") == "local_database" and not market.get("personalized"):
        certainty_am = " ይህ የተመዘገበ አጠቃላይ ዋጋ ነው፤ የአካባቢዎ ገበያ ሊለይ ይችላል።"
        certainty_en = " This is a stored general price; your local market may differ."
    elif market.get("source") == "wfp_hdx" and market.get("approximate"):
        matched = " / ".join(
            str(x)
            for x in (market.get("matched_commodity"), market.get("market"), market.get("matched_admin1"))
            if x
        )
        certainty_am = f" ትክክለኛ ግጥሚያ ስላልተገኘ ይህ የቀረበው ቅርብ ግምታዊ ግጥሚያ ነው: {matched}።"
        certainty_en = f" No exact match was found; this is the closest approximate match: {matched}."
    elif market.get("source") == "wfp_hdx":
        certainty_am = " ይህ በመዝገቡ ያለ የመጨረሻ ዋጋ ነው፤ የአካባቢ ገበያ ሊቀየር ይችላል።"
        certainty_en = " This is the latest available record; the local market may have changed."
    if market.get("source") == "wfp_hdx" and market.get("is_stale"):
        stale_am = (
            f" ማሳሰቢያ: ይህ ዋጋ {market.get('age_days')} ቀን ያህል የቆየ የመጨረሻ የተገኘ መዝገብ ነው፤ "
            "እንደ የዛሬ ዋጋ አይጠቀሙት። ከመሸጥዎ በፊት የአካባቢ ገበያን ያረጋግጡ።"
        )
        stale_en = (
            f" Warning: this is the latest available record but it is about {market.get('age_days')} days old; "
            "do not treat it as today's price. Confirm the local market before selling."
        )
    date_note_am = f" የመረጃው ቀን: {market.get('updated_at')}።" if market.get("source") == "wfp_hdx" and market.get("updated_at") else ""
    date_note_en = f" Data date: {market.get('updated_at')}." if market.get("source") == "wfp_hdx" and market.get("updated_at") else ""
    price_type_am = (
        f" የዋጋ አይነት: {market.get('price_type')}።"
        if market.get("source") == "wfp_hdx" and market.get("price_type")
        else ""
    )
    price_type_en = (
        f" Price type: {market.get('price_type')}."
        if market.get("source") == "wfp_hdx" and market.get("price_type")
        else ""
    )
    location_followup_am = (
        " ከተማዎን ወይም የሚጠቀሙበትን ገበያ ከነገሩኝ፣ "
        "በዳታቤዙ ካለ የዚያን ቦታ የተለየ ዋጋ እፈትሻለሁ።"
    )
    location_followup_en = (
        " Tell me your town or market and I will check a more specific local price if it is available."
    )
    place_note = "" if not market.get("needs_location_for_personal_price") else (
        location_followup_am if language == "am" else location_followup_en
    )
    requested_market = market.get("requested_market") or market.get("market")
    local_missing_am = (
        f" ለ{requested_market} የተለየ ዋጋ በአካባቢ ዳታቤዙ አልተገኘም፤ አጠቃላይ ዋጋን እያሳየሁ ነው።"
        if market.get("location_price_unavailable") and language == "am"
        else ""
    )
    local_missing_en = (
        f" A specific price for {requested_market} was not found, so I am showing the general price."
        if market.get("location_price_unavailable") and language != "am"
        else ""
    )
    return (
        f"የ{crop_am} ዋጋ በ{market.get('market') or 'ገበያ'} {market.get('price')} {market.get('unit')} ነው። "
        f"አዝማሚያው {trend_am} ነው። {recommendation_am}{date_note_am}{price_type_am}{certainty_am}"
        f"{stale_am}{local_missing_am}{place_note}"
        if language == "am"
        else (
            f"{crop} price in {market.get('market') or 'the matched market'} is {market.get('price')} {market.get('unit')}; "
            f"trend: {market.get('trend')}. {recommendation_en}{date_note_en}{price_type_en}{certainty_en}{stale_en}{local_missing_en}{place_note}"
        )
    )


def _simple_weather_answer(weather: dict[str, Any], language: str) -> str:
    if weather.get("available"):
        if language == "am":
            summary = "ዝናብ ይጠበቃል" if weather.get("summary") == "rain_expected" else "በአብዛኛው ደረቅ ሊሆን ይችላል"
            rain_now = weather.get("rain_now_mm")
            if rain_now is None:
                rain_now = weather.get("rainfall_now_mm")
            return (
                f"ቦታ: {weather.get('location')}። "
                f"ሙቀት {weather.get('temperature_c')}°C፣ እርጥበት {weather.get('humidity_pct')}%፣ "
                f"አሁን ዝናብ {rain_now} ሚሜ፣ "
                f"የ7 ቀን ዝናብ {weather.get('rainfall_7d_mm')} ሚሜ ነው። {summary}።"
            )
        return (
            f"Weather for {weather.get('location')}: temperature {weather.get('temperature_c')} C, "
            f"humidity {weather.get('humidity_pct')}%, current rain {weather.get('rain_now_mm')}, "
            f"7-day rainfall {weather.get('rainfall_7d_mm')} mm."
        )
    reason = weather.get("reason") or "unknown"
    if reason == "location_missing":
        return "የአየር መረጃ ለመፈተሽ ከተማዎን ወይም አካባቢዎን ይንገሩኝ።" if language == "am" else "Tell me your town or area to check weather."
    return (
        "የአየር መረጃ አልተገኘም። ትንሽ ቆይተው ይሞክሩ ወይም አካባቢዎን ይግለጹ።"
        if language == "am"
        else f"Weather was not available: {reason}"
    )


def _scaled_soil_ph(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    return round(raw / 10.0, 2) if raw > 14 else round(raw, 2)


def _soil_texture_am(texture: str | None) -> str:
    return {
        "clayey": "ሸክላማ",
        "loamy": "ሎሚ/መካከለኛ",
        "loam": "ሎሚ/መካከለኛ",
        "loam_or_mixed": "መካከለኛ ድብልቅ",
        "mixed": "ድብልቅ",
        "clay_loam": "ሸክላማ ሎሚ",
        "sandy": "አሸዋማ",
        "sandy_loam": "አሸዋማ ሎሚ",
        "silty": "ደቃቅ አፈር",
        "silt_loam": "ደቃቅ ሎሚ",
        "unknown": "ያልታወቀ",
    }.get(str(texture or "unknown").strip().lower(), str(texture or "ያልታወቀ"))


def _soil_water_status_am(status: str | None) -> str:
    return {
        "dry": "ደረቅ",
        "moderate": "መካከለኛ",
        "wet": "እርጥብ",
    }.get(str(status or "").strip().lower(), "ያልታወቀ")


def _irrigation_signal_am(signal: str | None) -> str:
    return {
        "irrigate_if_crop_stressed": "ተክሉ የውሃ ጭንቀት ካሳየ ያጠጡ",
        "monitor": "እርጥበቱን በቅርብ ይከታተሉ",
        "delay_irrigation": "አሁን መስኖን ትንሽ ያዘግዩ",
        "needed_soon": "በቅርብ መስኖ ያስፈልጋል",
    }.get(str(signal or "").strip().lower(), "የመስኖ ፍላጎትን በማሳ ላይ ያረጋግጡ")


def _simple_soil_answer(soil: dict[str, Any], language: str) -> str:
    if not soil.get("available"):
        reason = soil.get("reason") or "unknown"
        if reason == "coordinates_missing":
            return (
                "የአፈር መረጃ ለመፈተሽ ከተማዎን ወይም ወረዳ/አካባቢዎን ይንገሩኝ።"
                if language == "am"
                else "Tell me your town or district to check soil information."
            )
        return (
            "የአፈር መረጃ አልተገኘም። አካባቢዎን ይግለጹ ወይም የአፈር ምርመራ ያድርጉ።"
            if language == "am"
            else f"Soil data was not available: {reason}"
        )

    ph = _scaled_soil_ph(soil.get("ph_h2o"))
    texture = soil.get("soil_texture") or "unknown"
    acidic = bool((soil.get("suitability_indicators") or {}).get("acidic_risk"))
    low_om = bool((soil.get("suitability_indicators") or {}).get("low_organic_matter_risk"))
    if language == "am":
        if ph is None:
            acid_text = "pH አልተገኘም፤ አሲዳማነትን በአፈር ምርመራ ያረጋግጡ"
        elif acidic or ph < 5.5:
            acid_text = f"pH {ph} ነው፤ አፈሩ አሲዳማ ሊሆን ይችላል"
        elif ph <= 7.5:
            acid_text = f"pH {ph} ነው፤ አፈሩ ከፍተኛ አሲዳማ አይደለም"
        else:
            acid_text = f"pH {ph} ነው፤ አፈሩ አልካላይን ሊሆን ይችላል"
        om_text = "ኦርጋኒክ ንጥረ ነገር ዝቅ ሊሆን ስለሚችል ኮምፖስት/ፍግ ይጨምሩ" if low_om else "የኦርጋኒክ ንጥረ ነገር ትልቅ አደጋ አልታየም"
        return (
            f"{acid_text}። "
            f"የአፈር አይነቱ {_soil_texture_am(texture)} ነው። {om_text}። "
            "ትክክለኛ የማዳበሪያ/ኖራ መጠን ከመወሰን በፊት የአካባቢ የአፈር ምርመራ ያረጋግጡ።"
        )
    return f"Soil pH is {ph}, texture is {texture}, acidic risk is {acidic}, low organic matter risk is {low_om}."


def _simple_soil_fertilizer_answer(
    *,
    soil: dict[str, Any],
    ethiosis: dict[str, Any],
    soil_water: dict[str, Any],
    prediction: dict[str, Any],
    language: str,
) -> str:
    if language != "am":
        parts = ["Use the soil result and local fertilizer recommendation together."]
        if soil.get("available"):
            parts.append(_simple_soil_answer(soil, language))
        if soil_water.get("available"):
            parts.append(
                f"Soil moisture signal: {soil_water.get('status')} "
                f"({soil_water.get('swi_percent')}%), irrigation: {soil_water.get('irrigation_signal')}."
            )
        return " ".join(parts)

    parts: list[str] = []
    if soil.get("available"):
        parts.append(_simple_soil_answer(soil, "am"))
    else:
        parts.append("የቦታ መረጃ ካልተገኘ ትክክለኛ የአፈር ሁኔታ ለመመርመር latitude/longitude ወይም ከተማ/ወረዳ ይስጡ።")
    if soil_water.get("available"):
        parts.append(
            f"የአፈር እርጥበት {soil_water.get('swi_percent')}% "
            f"({_soil_water_status_am(soil_water.get('status'))}) ነው። "
            f"{_irrigation_signal_am(soil_water.get('irrigation_signal'))}።"
        )
    else:
        parts.append(
            "የአፈር እርጥበት ቁጥር አልተገኘም፤ በማሳ ላይ አፈሩን በእጅ በመፈተሽ የውሃ ጭንቀትን ያረጋግጡ።"
        )
    if ethiosis.get("available"):
        parts.extend((ethiosis.get("recommendations") or [])[:1])
    fert = prediction.get("fertilizer_recommendation")
    if fert == "check_liming_need_before_high_fertilizer_rates":
        parts.append("ከፍተኛ የማዳበሪያ መጠን ከመጠቀም በፊት የኖራ ፍላጎትን ያረጋግጡ።")
    elif fert == "add_compost_or_manure_and_confirm_with_soil_test":
        parts.append("ኮምፖስት/ፍግ ይጨምሩ፣ ከዚያ የማዳበሪያ መጠንን በአፈር ምርመራ ያረጋግጡ።")
    elif not ethiosis.get("available"):
        parts.append("ትክክለኛ መጠን ለመስጠት ሰብል፣ ወረዳ/ቀበሌ እና የአፈር ምርመራ ያስፈልጋሉ።")
    return strip_provider_names_from_voice(" ".join(p for p in parts if p).strip())


def _voice_kb_excerpt_from_hits(hits: list[dict[str, Any]], *, max_chars: int = 520) -> str:
    """Fast grounded voice answer from ranked KB hits (no final LLM)."""
    if not hits:
        return ""
    try:
        max_dist = float(
            os.getenv(
                "RAG_VOICE_KB_MAX_L2",
                os.getenv("RAG_PG_MAX_L2_DISTANCE", "1.35"),
            )
            or "1.35"
        )
    except ValueError:
        max_dist = 1.35
    ranked = sorted(hits, key=lambda h: float(h.get("distance", 999)))
    top = ranked[0]
    if float(top.get("distance", 999)) > max_dist:
        return ""
    cap = max(120, max_chars)
    parts: list[str] = []
    per = max(80, cap // min(2, len(ranked[:2])))
    for h in ranked[:2]:
        body = (h.get("content") or "").strip()
        if body:
            parts.append(body[:per])
    if not parts:
        return ""
    text = " ".join(parts)
    if len(text) > cap:
        text = text[: max(0, cap - 3)].rstrip() + "..."
    return text


def _is_compost_general_info(question: str) -> bool:
    q = (question or "").lower()
    return ("ኮምፖስት" in question or "compost" in q) and any(
        term in q or term in question
        for term in ("ጥቅም", "ምንድን", "ምንድነው", "benefit", "what is", "why")
    )


def _simple_compost_answer(language: str) -> str:
    if language == "am":
        return (
            "ኮምፖስት የአፈርን ኦርጋኒክ ንጥረ ነገር ይጨምራል፣ "
            "የውሃ መያዝ አቅምን ያሻሽላል፣ አፈርን ለሰብል ሥር ይለሰልሳል፣ "
            "እና ኬሚካል ማዳበሪያን ብቻ መመካትን ይቀንሳል።"
        )
    return (
        "Compost improves soil organic matter, helps the soil hold water, "
        "supports root growth, and reduces reliance on chemical fertilizer alone."
    )


def _dynamic_source_intro(context: dict[str, Any], language: str) -> str:
    """Build a short voice-friendly source phrase for live tool data."""
    sources: list[str] = []
    weather = context.get("weather") or {}
    soil = context.get("soil") or {}
    market = context.get("market") or {}
    web_results = context.get("web_results") or []

    if weather.get("available") and weather.get("source"):
        sources.append(str(weather["source"]))
    if soil.get("available") and soil.get("source"):
        sources.append(str(soil["source"]))
    if market.get("available"):
        source = str(market.get("source") or "").strip()
        if source == "local_database":
            sources.append("የአካባቢ ገበያ ዳታቤዝ")
        elif source and source != "mock_demo_data":
            sources.append(source)
    if web_results:
        title = str((web_results[0] or {}).get("title") or "").strip()
        href = str((web_results[0] or {}).get("href") or "").strip()
        if title:
            sources.append(title)
        elif href:
            sources.append(href)

    deduped = [s for i, s in enumerate(sources) if s and s not in sources[:i]][:2]
    if not deduped:
        return ""
    joined = " እና ".join(deduped)
    if language == "am":
        return f"እንደ {joined} የተገኘው መረጃ፣ "
    return f"According to {joined}, "


def _ensure_spoken_source_reference(answer: str, context: dict[str, Any], language: str) -> str:
    if os.getenv("RAG_SPOKEN_DYNAMIC_SOURCES", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return (answer or "").strip()

    text = (answer or "").strip()
    if not text:
        return ""
    intro = _dynamic_source_intro(context, language)
    if not intro:
        return text
    if any(src in text for src in ("Open-Meteo", "SoilGrids", "According to", "እንደ", "ምንጭ")):
        return text
    return intro + text[0].lower() + text[1:] if text and language != "am" else intro + text


def build_structured_context(
    *,
    question: str,
    nlu_result: dict[str, Any],
    farmer_history: dict[str, Any],
    kb_results: list[dict[str, Any]],
    weather: dict[str, Any],
    soil: dict[str, Any],
    ethiosis: dict[str, Any],
    soil_water: dict[str, Any],
    market: dict[str, Any],
    prediction: dict[str, Any],
    web_results: list[dict[str, Any]],
    supplemental_sources: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "user_question": question,
        "detected_intent": nlu_result.get("intent"),
        "entities": nlu_result.get("entities", {}),
        "farmer_profile": farmer_history,
        "kb_results": kb_results,
        "weather": weather,
        "soil": soil,
        "ethiosis": ethiosis,
        "soil_water": soil_water,
        "market": market,
        "prediction": prediction,
        "web_results": web_results,
        "supplemental_sources": supplemental_sources,
    }


def build_smart_context_only(
    *,
    question: str,
    phone_number: str,
    nlu: Any,
    profile: dict[str, Any] | None,
    history_pairs: list[tuple[str, str]],
    hits: list[dict[str, Any]],
    local_market_price_func: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build routed context and trace without calling a final LLM."""
    routed = classify_intent_and_entities(question, nlu=nlu, profile=profile)
    entities = routed["entities"]
    crop = entities.get("crop_en") or entities.get("crop") or (profile or {}).get("crop_type")
    location = (
        entities.get("location")
        or entities.get("location_en")
        or entities.get("region_keyword")
        or (profile or {}).get("location")
    )
    kb = search_knowledge_base(question, hits, top_k=int(os.getenv("RAG_SMART_TOP_K", "4") or "4"))
    supplemental_sources = supplemental_context_block(question)
    farmer_history = get_farmer_history(phone_number, profile, history_pairs)
    question_l = (question or "").lower()
    crop_needs_live_context = any(
        term in question_l or term in question
        for term in (
            "weather",
            "rain",
            "forecast",
            "irrigat",
            "drought",
            "soil",
            "market",
            "price",
            "ዝናብ",
            "አየር",
            "መስኖ",
            "ድርቅ",
            "አፈር",
            "ገበያ",
            "ዋጋ",
        )
    )

    tool_trace: list[dict[str, Any]] = [{"tool": "knowledge_base", "results": len(kb)}]
    if supplemental_sources:
        tool_trace.append({
            "tool": "supplemental_source_catalog",
            "results": len(supplemental_sources),
            "priority": "after_kb_and_live_tools",
        })
    weather = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"weather_advice", "irrigation_advice", "disease_diagnosis", "yield_prediction", "emergency_pest_or_disease"} or (
        routed["intent"] == "crop_recommendation" and crop_needs_live_context
    ):
        weather = get_weather(location)
        tool_trace.append({
            "tool": "weather",
            "provider": "Open-Meteo",
            "available": weather.get("available"),
            "location": weather.get("location") or location,
            "reason": weather.get("reason"),
            "cache": weather.get("cache"),
        })

    lat = entities.get("latitude") or (profile or {}).get("latitude")
    lon = entities.get("longitude") or (profile or {}).get("longitude")
    soil_location_source = None
    if (lat is None or lon is None) and location:
        try:
            coords = _geocode(str(location))
        except Exception:
            coords = _known_location_coords(str(location))
        if coords:
            lat, lon, soil_location_source = coords
    soil = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"soil_advice", "fertilizer_advice", "irrigation_advice", "yield_prediction"} or (
        routed["intent"] == "crop_recommendation" and crop_needs_live_context
    ):
        soil = get_soil_data(lat, lon)
        tool_trace.append({
            "tool": "soil",
            "provider": "ISRIC SoilGrids",
            "available": soil.get("available"),
            "location": soil_location_source,
            "latitude": lat,
            "longitude": lon,
            "reason": soil.get("reason"),
            "cache": soil.get("cache"),
        })

    ethiosis = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"soil_advice", "fertilizer_advice", "yield_prediction", "irrigation_advice"} or (
        routed["intent"] == "crop_recommendation" and crop_needs_live_context
    ):
        ethiosis = get_ethiosis_baseline(location, crop, soil)
        tool_trace.append({
            "tool": "ethiosis",
            "provider": "EthioSIS baseline",
            "available": ethiosis.get("available"),
            "is_live_api": ethiosis.get("is_live_api"),
        })

    soil_water = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"soil_advice", "fertilizer_advice", "irrigation_advice", "yield_prediction"} or (
        routed["intent"] == "crop_recommendation" and crop_needs_live_context
    ):
        soil_water = get_copernicus_soil_water_index(lat, lon, weather=weather)
        tool_trace.append({
            "tool": "soil_water",
            "provider": "Copernicus Soil Water Index",
            "available": soil_water.get("available"),
            "status": soil_water.get("status"),
            "swi_percent": soil_water.get("swi_percent"),
            "reason": soil_water.get("reason"),
            "cache": soil_water.get("cache"),
        })

    market = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"market_price", "yield_prediction"} or (
        routed["intent"] == "crop_recommendation" and crop_needs_live_context
    ):
        market = get_market_price(crop, location, local_price_func=local_market_price_func)
        tool_trace.append({
            "tool": "market",
            "available": market.get("available"),
            "source": market.get("source"),
            "market": market.get("market"),
            "matched_commodity": market.get("matched_commodity"),
            "match_quality": market.get("match_quality"),
            "approximate": market.get("approximate"),
            "price_type": market.get("price_type"),
            "trend": market.get("trend"),
            "recency_status": market.get("recency_status"),
            "age_days": market.get("age_days"),
            "is_stale": market.get("is_stale"),
            "personalized": market.get("personalized"),
            "reason": market.get("reason"),
        })

    prediction = predict_agricultural_advice(farmer_history, weather, soil, soil_water, market, question)
    if routed["intent"] == "market_price" and market.get("available"):
        web = []
    elif routed["intent"] in {"weather_advice", "soil_advice", "fertilizer_advice", "irrigation_advice"}:
        web = []
    else:
        web = web_search_fallback(question, kb, routed["intent"])
    if web:
        tool_trace.append({"tool": "web_search", "results": len(web)})

    context = build_structured_context(
        question=question,
        nlu_result=routed,
        farmer_history=farmer_history,
        kb_results=kb,
        weather=weather,
        soil=soil,
        ethiosis=ethiosis,
        soil_water=soil_water,
        market=market,
        prediction=prediction,
        web_results=web,
        supplemental_sources=supplemental_sources,
    )
    return context, tool_trace, kb


def run_smart_advisory(
    *,
    question: str,
    phone_number: str,
    nlu: Any,
    profile: dict[str, Any] | None,
    history_pairs: list[tuple[str, str]],
    hits: list[dict[str, Any]],
    local_market_price_func: Any | None,
) -> SmartResult:
    context, tool_trace, kb = build_smart_context_only(
        question=question,
        phone_number=phone_number,
        nlu=nlu,
        profile=profile,
        history_pairs=history_pairs,
        hits=hits,
        local_market_price_func=local_market_price_func,
    )
    routed_intent = context.get("detected_intent")
    market = context.get("market") or {}
    weather = context.get("weather") or {}
    soil = context.get("soil") or {}
    ethiosis = context.get("ethiosis") or {}
    soil_water = context.get("soil_water") or {}
    prediction = context.get("prediction") or {}

    # Cheap deterministic paths keep live tool answers working even without a final LLM.
    lang = (profile or {}).get("preferred_language") or (profile or {}).get("primary_language") or "am"
    def _finish_voice(answer: str, *, used_llm: bool) -> SmartResult:
        cleaned = strip_provider_names_from_voice(sanitize_voice_advice(answer, question))
        if not used_llm:
            cleaned = _ensure_spoken_source_reference(cleaned, context, lang)
        return SmartResult(cleaned, context, kb, used_llm, tool_trace)

    if routed_intent == "market_price" and market.get("available"):
        return _finish_voice(_simple_market_answer(market, lang), used_llm=False)
    if routed_intent == "weather_advice":
        return _finish_voice(_simple_weather_answer(weather, lang), used_llm=False)
    if routed_intent == "fertilizer_advice" and _is_compost_general_info(question):
        return _finish_voice(_simple_compost_answer(lang), used_llm=False)
    if routed_intent in {"soil_advice", "fertilizer_advice", "irrigation_advice"}:
        return _finish_voice(
            _simple_soil_fertilizer_answer(
                soil=soil,
                ethiosis=ethiosis,
                soil_water=soil_water,
                prediction=prediction,
                language=lang,
            ),
            used_llm=False,
        )

    skip_final_llm = os.getenv("RAG_SMART_SKIP_LLM_ON_KB", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if skip_final_llm and hits:
        try:
            cap = int(os.getenv("RAG_VOICE_KB_EXCERPT_CHARS", "520") or "520")
        except ValueError:
            cap = 520
        excerpt = _voice_kb_excerpt_from_hits(hits, max_chars=cap)
        if excerpt:
            tool_trace.append({"tool": "kb_excerpt", "chars": len(excerpt)})
            return _finish_voice(excerpt, used_llm=False)

    backend = os.getenv("RAG_SMART_FINAL_BACKEND", "gemini").strip().lower()
    if backend not in {"gemini", "groq", "openai"}:
        backend = "gemini"
    if backend == "gemini" and not gemini_api_keys():
        backend = effective_llm_backend()
    system = _final_system_prompt(str(lang))
    user_block = (
        "Final compact context follows. Answer only from this context. "
        "Only mention missing data if it is essential to the user's exact question; otherwise give the useful advice and stop.\n\n"
        + _compact(context, max_chars=int(os.getenv("RAG_SMART_CONTEXT_CHARS", "4500") or "4500"))
    )
    messages = prepare_rag_llm_messages(system, [], user_block, backend)
    answer, used_backend = run_sync_llm(backend, messages, fast=True)
    tool_trace.append({"tool": "final_llm", "backend": used_backend})
    return _finish_voice(answer, used_llm=True)
