"""Tool-routed agricultural advisory pipeline.

This module keeps routing, data retrieval, prediction, and final generation
separate so Gemini is used only after local tools have built compact context.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .llm_providers import effective_llm_backend, gemini_api_keys
from .query_llm import prepare_rag_llm_messages, run_sync_llm
from .rag_tools import web_search


WEATHER_TTL_SEC = int(os.getenv("RAG_WEATHER_CACHE_TTL_SEC", "7200") or "7200")
SOIL_TTL_SEC = int(os.getenv("RAG_SOIL_CACHE_TTL_SEC", str(180 * 24 * 3600)) or str(180 * 24 * 3600))
_weather_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_soil_cache: dict[str, tuple[float, dict[str, Any]]] = {}


INTENT_MAP = {
    "market_price": "market_price",
    "soil_fertility": "fertilizer_advice",
    "pest_disease": "disease_diagnosis",
    "crop_production": "crop_recommendation",
    "soil_water_conservation": "soil_advice",
    "land_characterization": "soil_advice",
    "post_harvest": "general_agriculture",
}

INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("emergency_pest_or_disease", ("severe", "emergency", "በፍጥነት", "ተስፋፋ", "ሞተ", "አደጋ")),
    ("weather_advice", ("weather", "forecast", "rain", "ዝናብ", "የአየር", "አየር")),
    ("irrigation_advice", ("irrigat", "መስኖ", "ውኃ", "water")),
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


@dataclass
class SmartResult:
    answer: str
    context: dict[str, Any]
    references: list[dict[str, Any]]
    used_llm: bool
    tool_trace: list[dict[str, Any]]


def _norm_key(value: str | None) -> str:
    return (value or "").strip().lower()


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
    if profile and profile.get("location") and not entities.get("location"):
        entities["location"] = profile.get("location")

    farm_size_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:ha|hectare|ሄክታር)", q, re.I)
    if farm_size_match:
        entities["farm_size_ha"] = float(farm_size_match.group(1))
    for key, pattern in (("latitude", r"lat(?:itude)?\s*[:=]?\s*(-?[0-9.]+)"), ("longitude", r"lon(?:gitude)?\s*[:=]?\s*(-?[0-9.]+)")):
        m = re.search(pattern, lower)
        if m:
            entities[key] = float(m.group(1))

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
        if not geo:
            return {"available": False, "reason": "location_not_found", "location": loc}
        lat, lon, label = geo
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
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
            "wind_kph": (data.get("current") or {}).get("wind_speed_10m"),
            "rainfall_7d_mm": round(rain_sum, 1),
            "summary": "rain_expected" if rain_sum >= 10 else "mostly_dry",
            "source": "Open-Meteo",
            "cache": "miss",
        }
        _cache_set(_weather_cache, key, payload)
        return payload
    except Exception as exc:
        return {"available": False, "reason": f"weather_error: {exc}"}


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
        return {"available": False, "reason": f"soil_error: {exc}"}


def get_market_price(
    crop: str | None,
    location_or_market: str | None,
    local_price_func: Any | None = None,
) -> dict[str, Any]:
    crop_key = _norm_key(crop)
    if not crop_key:
        return {"available": False, "reason": "crop_missing"}
    row = None
    if local_price_func is not None:
        try:
            row = local_price_func(crop, location_or_market) or local_price_func(crop)
        except Exception:
            row = None
    if row:
        price, unit, updated_at = row
        return {
            "available": True,
            "crop": crop,
            "market": location_or_market,
            "price": price,
            "previous_price": None,
            "unit": unit,
            "trend": "unknown",
            "updated_at": str(updated_at),
            "source": "local_database",
            "selling_recommendation": "Use local extension/market confirmation before selling large volume.",
        }
    demo = MOCK_MARKET.get(crop_key) or MOCK_MARKET.get(crop_key.replace(" ", "_"))
    if not demo:
        return {"available": False, "reason": "not_in_local_or_demo_market_data", "crop": crop}
    trend = "up" if demo["price"] > demo["previous_price"] else "down" if demo["price"] < demo["previous_price"] else "flat"
    return {
        "available": True,
        "crop": crop,
        "market": location_or_market or "demo",
        **demo,
        "trend": trend,
        "source": "mock_demo_data",
        "selling_recommendation": "sell_gradually" if trend == "up" else "avoid_rushed_sale",
    }


def predict_agricultural_advice(
    farmer_profile: dict[str, Any],
    weather: dict[str, Any],
    soil: dict[str, Any],
    market: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    q = (question or "").lower()
    rain7 = float(weather.get("rainfall_7d_mm") or 0) if weather.get("available") else 0.0
    humidity = float(weather.get("humidity_pct") or 0) if weather.get("available") and weather.get("humidity_pct") is not None else 0.0
    temp = float(weather.get("temperature_c") or 0) if weather.get("available") and weather.get("temperature_c") is not None else 0.0
    irrigation = "needed_soon" if rain7 < 8 else "delay_irrigation"
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
        "Avoid Markdown formatting such as **bold**, tables, and long numbered outlines. "
        "For pesticides, chemicals, animal/human health, or severe disease, include safety warnings and advise contacting an extension worker."
    )


def sanitize_voice_advice(answer: str, question: str) -> str:
    """Remove LLM boilerplate that is unhelpful for voice answers."""
    text = (answer or "").strip()
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
    return (
        f"የ{crop} ዋጋ {market.get('price')} {market.get('unit')} ነው። "
        f"አዝማሚያው {market.get('trend')} ነው። "
        f"ምንጭ: {market.get('source')}። "
        "ትልቅ መጠን ከመሸጥዎ በፊት የአካባቢ ገበያ ዋጋን ያረጋግጡ።"
        if language == "am"
        else f"{crop} price is {market.get('price')} {market.get('unit')}; trend: {market.get('trend')}. Source: {market.get('source')}."
    )


def build_structured_context(
    *,
    question: str,
    nlu_result: dict[str, Any],
    farmer_history: dict[str, Any],
    kb_results: list[dict[str, Any]],
    weather: dict[str, Any],
    soil: dict[str, Any],
    market: dict[str, Any],
    prediction: dict[str, Any],
    web_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "user_question": question,
        "detected_intent": nlu_result.get("intent"),
        "entities": nlu_result.get("entities", {}),
        "farmer_profile": farmer_history,
        "kb_results": kb_results,
        "weather": weather,
        "soil": soil,
        "market": market,
        "prediction": prediction,
        "web_results": web_results,
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
    location = entities.get("location") or entities.get("region_keyword") or (profile or {}).get("location")
    kb = search_knowledge_base(question, hits, top_k=int(os.getenv("RAG_SMART_TOP_K", "5") or "5"))
    farmer_history = get_farmer_history(phone_number, profile, history_pairs)

    tool_trace: list[dict[str, Any]] = [{"tool": "knowledge_base", "results": len(kb)}]
    weather = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"weather_advice", "irrigation_advice", "disease_diagnosis", "crop_recommendation", "yield_prediction", "emergency_pest_or_disease"}:
        weather = get_weather(location)
        tool_trace.append({"tool": "weather", "available": weather.get("available"), "cache": weather.get("cache")})

    lat = entities.get("latitude") or (profile or {}).get("latitude")
    lon = entities.get("longitude") or (profile or {}).get("longitude")
    soil = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"soil_advice", "fertilizer_advice", "crop_recommendation", "yield_prediction"}:
        soil = get_soil_data(lat, lon)
        tool_trace.append({"tool": "soil", "available": soil.get("available"), "cache": soil.get("cache")})

    market = {"available": False, "reason": "not_routed"}
    if routed["intent"] in {"market_price", "crop_recommendation", "yield_prediction"}:
        market = get_market_price(crop, location, local_price_func=local_market_price_func)
        tool_trace.append({"tool": "market", "available": market.get("available"), "source": market.get("source")})

    prediction = predict_agricultural_advice(farmer_history, weather, soil, market, question)
    if routed["intent"] == "market_price" and market.get("available"):
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
        market=market,
        prediction=prediction,
        web_results=web,
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

    # Cheap deterministic path for pure market questions.
    lang = (profile or {}).get("preferred_language") or (profile or {}).get("primary_language") or "am"
    if routed_intent == "market_price" and market.get("available") and not kb and not context.get("web_results"):
        return SmartResult(_simple_market_answer(market, lang), context, kb, False, tool_trace)

    backend = os.getenv("RAG_SMART_FINAL_BACKEND", "gemini").strip().lower()
    if backend not in {"gemini", "groq", "ollama", "openai"}:
        backend = "gemini"
    if backend == "gemini" and not gemini_api_keys():
        backend = effective_llm_backend()
    system = _final_system_prompt(str(lang))
    user_block = (
        "Final compact context follows. Answer only from this context. "
        "Only mention missing data if it is essential to the user's exact question; otherwise give the useful advice and stop.\n\n"
        + _compact(context, max_chars=int(os.getenv("RAG_SMART_CONTEXT_CHARS", "8500") or "8500"))
    )
    messages = prepare_rag_llm_messages(system, [], user_block, backend)
    answer, used_backend = run_sync_llm(backend, messages, fast=True)
    answer = sanitize_voice_advice(answer, question)
    tool_trace.append({"tool": "final_llm", "backend": used_backend})
    return SmartResult(answer.strip(), context, kb, True, tool_trace)
