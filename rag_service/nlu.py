"""
Amharic-oriented NLU: intent classification + light entity extraction (SRS FR04-style).
Rule-based MVP — no training on your PDFs; more documents improve retrieval coverage, not this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Crop / commodity mentions → English key used by market_prices table
CROP_KEYWORDS: dict[str, str] = {
    "ጤፍ": "Teff",
    "teff": "Teff",
    "ስንዴ": "Wheat",
    "wheat": "Wheat",
    "ቦሎቄ": "Maize",
    "corn": "Maize",
    "maize": "Maize",
    "ገብስ": "Barley",
    "barley": "Barley",
    "ቦርኬ": "Sorghum",
    "sorghum": "Sorghum",
    "ዳጉሳ": "Sorghum",
    "ሽምብራ": "Chickpea",
    "chickpea": "Chickpea",
    "ምስር": "Lentil",
    "lentil": "Lentil",
    "ቅቤ": "Butter",
    "ቡና": "Coffee",
    "coffee": "Coffee",
}

REGION_KEYWORDS: dict[str, str] = {
    "ደጋ": "highland",
    "highland": "highland",
    "ቆላ": "lowland",
    "lowland": "lowland",
    "ወይና ደጋ": "midland",
    "ወይናደጋ": "midland",
    "midland": "midland",
    "መስኖ": "irrigated",
    "irrigation": "irrigated",
}

CROP_ENTITY_WORDS = set(CROP_KEYWORDS.keys())
REGION_ENTITY_WORDS = set(REGION_KEYWORDS.keys())

# Explicit market/price terms only — bare "ስንት" matches fertilizer dose questions too.
MARKET_KEYWORDS = [
    "ዋጋ",
    "ገበያ",
    "price",
    "market",
    "cost",
    "ብር",
    "ሽያጭ",
]

# Dose / application questions often use "ስንት" + "መጠን" without asking market price.
_NON_MARKET_DOSE_SIGNALS = [
    "ማዳበሪያ",
    "fertilizer",
    "urea",
    "dap",
    "npk",
    "compost",
    "nutrient",
    "መጠን",
    "dose",
    "dosage",
    "spray",
    "መርጨት",
    "application rate",
]


def _is_market_price_intent(text: str, lower: str) -> bool:
    """True when the farmer is asking commodity/market price, not rate/dose."""
    if not any(k in lower or k in text for k in MARKET_KEYWORDS):
        return False
    has_dose = any(s in lower or s in text for s in _NON_MARKET_DOSE_SIGNALS)
    strong_market = (
        "ዋጋ",
        "ገበያ",
        "price",
        "market",
        "cost",
        "ብር",
        "ሽያጭ",
    )
    if has_dose and not any(m in lower or m in text for m in strong_market):
        return False
    return True

AGRI_INTENT_KEYWORDS = [
    "ማዳበሪያ",
    "ፀረ-ተባይ",
    "ፀረ ተባይ",
    "ምርት",
    "ዘር",
    "መዝራት",
    "fertilizer",
    "pest",
    "disease",
    "plant",
    "crop",
    "spray",
    "harvest",
    "ሰብል",
    "ለምለም",
    "ፍሬ",
    "ቅጠል",
    "መሬት",
    "ውኃ",
    "ጥበቃ",
    "ዝናብ",
    "የአየር",
    "extension",
    "ማራዘም",
]

# (intent_id, keywords) — Amharic + English tokens; first match wins by score
_TOPIC_RULES: list[tuple[str, list[str]]] = [
    (
        "weather_advice",
        [
            "weather",
            "forecast",
            "rain",
            "ዝናብ",
            "የአየር",
            "አየር ሁኔታ",
            "ትንበያ",
        ],
    ),
    (
        "post_harvest",
        [
            "post-harvest",
            "postharvest",
            "ከመከር",
            "አከማችት",
            "አከማቻ",
            "ማከማቻ",
            "ድራር",
            "ማደስ",
            "እንዴት ይቆማል",
            "storage",
            "loss",
        ],
    ),
    (
        "soil_water_conservation",
        [
            "soil and water",
            "water conservation",
            "soil conservation",
            "የመሬት",
            "ውኃ",
            "ጥበቃ",
            "erosion",
            "terrace",
            "ቋሚነት",
        ],
    ),
    (
        "soil_fertility",
        [
            "soil fertility",
            "fertility",
            "ማዳበሪያ",
            "nutrient",
            "organic matter",
            "የመሬት ሀብት",
            "isfm",
            "የመሬት ስም",
        ],
    ),
    (
        "pest_disease",
        [
            "pest",
            "vector",
            "disease",
            "ፀረ",
            "ተባይ",
            "በሽታ",
            "wheat value chain",
            "pvmp",
        ],
    ),
    (
        "land_characterization",
        [
            "landpks",
            "soil type",
            "classification",
            "የመሬት አይነት",
            "land potential",
        ],
    ),
    (
        "extension_advisory",
        [
            "extension",
            "ማራዘም",
            "ማስፋፊያ",
            "ቁሳቁስ",
            "ቁሳቁሶች",
            "መመሪያ",
            "መምሪያ",
            "ፖስተር",
            "ፍሊፕ",
            "ፍሊፕ መጽሐፍ",
            "የውይይት ቡድን",
            "የመስክ ጉብኝት",
            "material",
            "ttl",
            "da ",
            "development agent",
            "የማራዘም",
        ],
    ),
    (
        "crop_production",
        [
            "lowland",
            "crop option",
            "ዝቅተኛ",
            "ሰብል",
            "irrigation",
            "መስኖ",
            "planting",
            "መዝራት",
            "cultivar",
        ],
    ),
]

# Short Amharic hints appended only for embedding search (not shown to user)
_RETRIEVAL_HINTS: dict[str, str] = {
    "post_harvest": "የከመር ምርት አያያዝ ማከማቻ እና ኪሳራ መቀነስ",
    "soil_water_conservation": "የመሬት እና ውኃ ጥበቃ እና መሬት ማቆየት",
    "soil_fertility": "የመሬት ሀብት ማዳበሪያ እና ISFM",
    "pest_disease": "ፀረ ተባይ እና በሽታ አስተዳደር",
    "land_characterization": "የመሬት ምድብ እና LandPKS",
    "extension_advisory": "የማራዘም ቅያት እና ሰነዶች",
    "weather_advice": "የአየር ሁኔታ ዝናብ ትንበያ እና የሰብል ምክር",
    "crop_production": "የሰብል ምርት እና የመስኖ አማራጮች",
    "general_agronomy": "የግብርና ምክር እና ልምድ",
}


@dataclass
class NLUResult:
    primary_intent: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "confidence": round(self.confidence, 3),
            "entities": self.entities,
        }


def _extract_crop_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    lower = text.lower()
    for kw, crop_en in CROP_KEYWORDS.items():
        if kw in lower or kw in text:
            out["crop_en"] = crop_en
            out["crop_keyword"] = kw
            break
    return out


def _extract_region_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    lower = text.lower()
    for kw, reg_en in REGION_KEYWORDS.items():
        if kw in lower or kw in text:
            out["region_en"] = reg_en
            out["region_keyword"] = kw
            break
    return out


def analyze_intent(text: str) -> NLUResult:
    """
    Classify farmer question intent and build a retrieval query
    (original question + optional Amharic topic hint for embedding).
    """
    stripped = (text or "").strip()
    if not stripped:
        return NLUResult("unknown", 0.0, {}, "")

    entities = _extract_crop_entities(stripped)
    entities.update(_extract_region_entities(stripped))
    lower = stripped.lower()

    # Market price (separate data path in main)
    if _is_market_price_intent(stripped, lower):
        conf = 0.88 if entities.get("crop_en") else 0.72
        return NLUResult("market_price", conf, entities, stripped)

    best_intent = "general_agronomy"
    best_score = 0
    for intent_id, keywords in _TOPIC_RULES:
        score = 0
        for kw in keywords:
            if len(kw) <= 2:
                continue
            if kw in lower or kw in stripped:
                score += 1
        if score > best_score:
            best_score = score
            best_intent = intent_id

    if best_score == 0:
        # Weak agr signal → unknown vs general
        has_agri = any(
            k in lower or k in stripped
            for k in AGRI_INTENT_KEYWORDS
        )
        if has_agri:
            best_intent = "general_agronomy"
            best_score = 1
            conf = 0.42
        else:
            best_intent = "unknown"
            conf = 0.28
    else:
        conf = min(0.93, 0.38 + 0.11 * best_score)

    # For "unknown" we do NOT append topic hints; it can bias retrieval
    # away from the user's exact wording (e.g. manuals / extension materials).
    if best_intent == "unknown":
        retrieval = stripped
    else:
        hint = _RETRIEVAL_HINTS.get(best_intent, _RETRIEVAL_HINTS["general_agronomy"])
        retrieval = f"{stripped}\n{hint}"

    return NLUResult(
        primary_intent=best_intent,
        confidence=conf,
        entities=entities,
        retrieval_query=retrieval,
    )


def needs_slot_filling(text: str, session_state: Optional[dict], nlu: NLUResult) -> Optional[str]:
    """
    Ask for crop when agronomy question lacks crop (skipped for market / unknown with no agr).
    """
    if session_state and session_state.get("current_state") != "active":
        return None
    if nlu.primary_intent == "market_price":
        return None

    # Only ask for crop on intents where crop is usually required to give safe/specific advice.
    # Post-harvest and general info can often be answered generically.
    intents_requiring_crop = {
        "crop_production",
        "soil_fertility",
        "pest_disease",
        "general_agronomy",
        "weather_advice",
    }
    if nlu.primary_intent not in intents_requiring_crop:
        return None

    lower = text.lower()
    has_agri = any(k in lower or k in text for k in AGRI_INTENT_KEYWORDS)
    has_crop = any(k in lower or k in text for k in CROP_ENTITY_WORDS)
    has_region = any(k in lower or k in text for k in REGION_ENTITY_WORDS)

    # 1. Check for missing Crop
    if has_agri and not has_crop:
        return "ለምን ሰብል ነው ጥያቄዎ? (ስንዴ፣ ጤፍ፣ ቦሎቄ፣ ወዘተ.)"
    
    # 2. Check for missing Region on region-dependent questions
    # (Planting and soil production are highly region-dependent)
    region_critical_intents = {"crop_production", "soil_fertility"}
    if nlu.primary_intent in region_critical_intents and not has_region:
        return "ለየትኛው አካባቢ ነው የሚፈልጉት? (ደጋ፣ ቆላ ወይም ወይና ደጋ)"
    
    return None
