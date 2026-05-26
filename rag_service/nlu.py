"""
Amharic-oriented NLU: intent classification + light entity extraction (SRS FR04-style).
Rule-based MVP — no training on PDFs; more documents improve retrieval coverage, not this layer.

Scope:
- Crop production
- Soil fertility
- Pest and disease
- Soil and water conservation
- Post-harvest handling
- Extension advisory materials
- Weather-related crop advice
- Market price routing

Out of scope:
- Livestock advisory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Entity dictionaries
# -----------------------------------------------------------------------------
# Crop / commodity mentions → English key used by market_prices table or KB metadata.
# Keep the English values aligned with your DB table values.
CROP_KEYWORDS: dict[str, str] = {
    # Cereals
    "ጤፍ": "Teff",
    "ጣፍ": "Teff",
    "teff": "Teff",
    "ስንዴ": "Wheat",
    "wheat": "Wheat",
    "በቆሎ": "Maize",
    "በቆሎዬ": "Maize",
    "corn": "Maize",
    "maize": "Maize",
    "ገብስ": "Barley",
    "barley": "Barley",
    "ማሽላ": "Sorghum",
    "ቦርኬ": "Sorghum",
    "ዳጉሳ": "Sorghum",
    "sorghum": "Sorghum",

    # Pulses / legumes
    "ቦሎቄ": "Haricot Bean",
    "ባቄላ": "Faba Bean",
    "haricot bean": "Haricot Bean",
    "common bean": "Haricot Bean",
    "faba bean": "Faba Bean",
    "ሽምብራ": "Chickpea",
    "chickpea": "Chickpea",
    "ምስር": "Lentil",
    "lentil": "Lentil",

    # Horticulture / cash crops commonly useful for demo
    "ድንች": "Potato",
    "potato": "Potato",
    "ቲማቲም": "Tomato",
    "tomato": "Tomato",
    "ሽንኩርት": "Onion",
    "onion": "Onion",
    "ቡና": "Coffee",
    "coffee": "Coffee",

    # Commodity item sometimes used in market questions
    "ቅቤ": "Butter",
    "butter": "Butter",
}

# Agro-ecological / production condition terms.
REGION_KEYWORDS: dict[str, str] = {
    "ወይና ደጋ": "midland",
    "ወይናደጋ": "midland",
    "midland": "midland",
    "ደጋ": "highland",
    "highland": "highland",
    "ቆላ": "lowland",
    "lowland": "lowland",
    "መስኖ": "irrigated",
    "irrigation": "irrigated",
    "irrigated": "irrigated",
    "rainfed": "rainfed",
    "በዝናብ": "rainfed",
}

# Administrative / market-location words.
# These are separate from agro-ecological region because weather/market routing needs actual place names.
LOCATION_KEYWORDS: dict[str, str] = {
    "አዲስ አበባ": "Addis Ababa",
    "addis ababa": "Addis Ababa",
    "ኦሮሚያ": "Oromia",
    "oromia": "Oromia",
    "አማራ": "Amhara",
    "amhara": "Amhara",
    "ትግራይ": "Tigray",
    "tigray": "Tigray",
    "ሲዳማ": "Sidama",
    "sidama": "Sidama",
    "አፋር": "Afar",
    "afar": "Afar",
    "ሶማሌ": "Somali",
    "somali": "Somali",
    "ደቡብ": "SNNPR",
    "snnpr": "SNNPR",
    "አርሲ": "Arsi",
    "arsi": "Arsi",
    "ባሌ": "Bale",
    "bale": "Bale",
    "ሸዋ": "Shewa",
    "shewa": "Shewa",
}

CROP_ENTITY_WORDS = set(CROP_KEYWORDS.keys())
REGION_ENTITY_WORDS = set(REGION_KEYWORDS.keys())
LOCATION_ENTITY_WORDS = set(LOCATION_KEYWORDS.keys())


# -----------------------------------------------------------------------------
# Intent keywords
# -----------------------------------------------------------------------------
# Explicit market/price terms only — bare "ስንት" can also appear in fertilizer dose questions.
MARKET_KEYWORDS = [
    "ዋጋ",
    "ገበያ",
    "price",
    "market",
    "cost",
    "ብር",
    "ሽያጭ",
    "ኪሎ ስንት",
    "quintal",
    "ኩንታል",
]

# Dose / application questions often use amount/quantity words without asking market price.
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

AGRI_INTENT_KEYWORDS = [
    "ማዳበሪያ",
    "ፀረ-ተባይ",
    "ፀረ ተባይ",
    "ተባይ",
    "በሽታ",
    "አረም",
    "አፈር",
    "የአፈር",
    "አሲዳማ",
    "አሲዳማነት",
    "ፒኤች",
    "ኖራ",
    "ምርት",
    "ዘር",
    "መዝራት",
    "ማከማቻ",
    "ከመከር",
    "ኪሳራ",
    "መሸርሸር",
    "እርከን",
    "የመሬት አይነት",
    "መመሪያ",
    "ማስፋፊያ",
    "የግብርና",
    "ሰብል",
    "ፍሬ",
    "ቅጠል",
    "መሬት",
    "ውኃ",
    "ውሃ",
    "ጥበቃ",
    "ዝናብ",
    "የአየር",
    "soil",
    "acidity",
    "acidic",
    "ph",
    "lime",
    "liming",
    "fertilizer",
    "pest",
    "disease",
    "weed",
    "rust",
    "fungus",
    "plant",
    "crop",
    "spray",
    "harvest",
    "postharvest",
    "post-harvest",
    "storage",
    "erosion",
    "terrace",
    "conservation",
    "landpks",
    "land type",
    "guideline",
    "manual",
    "training",
    "agriculture",
    "agronomy",
    "extension",
]

# (intent_id, keywords) — Amharic + English tokens; highest score wins.
_TOPIC_RULES: list[tuple[str, list[str]]] = [
    (
        "weather_advice",
        [
            "weather",
            "forecast",
            "rain",
            "rainfall",
            "climate",
            "temperature",
            "humidity",
            "dry spell",
            "drought",
            "ዝናብ",
            "የዝናብ",
            "የአየር",
            "አየር ሁኔታ",
            "ትንበያ",
            "ድርቅ",
            "ሙቀት",
        ],
    ),
    (
        "post_harvest",
        [
            "post-harvest",
            "postharvest",
            "post harvest",
            "after harvest",
            "ከመከር",
            "ድህረ ምርት",
            "ድህረ-ምርት",
            "አከማቻ",
            "ማከማቻ",
            "ማከማቸት",
            "ማጠራቀም",
            "ጎተራ",
            "እንዴት ይቆማል",
            "storage",
            "store",
            "stored",
            "loss",
            "grain loss",
            "moisture",
            "drying",
            "threshing",
        ],
    ),
    (
        "soil_water_conservation",
        [
            "soil and water",
            "water conservation",
            "soil conservation",
            "soil erosion",
            "watershed",
            "water harvesting",
            "runoff",
            "bund",
            "check dam",
            "የመሬት",
            "ውኃ",
            "ውሃ",
            "ጥበቃ",
            "የአፈር ጥበቃ",
            "የውሃ ጥበቃ",
            "መሸርሸር",
            "እርከን",
            "እርከን ስራ",
            "erosion",
            "terrace",
            "terracing",
            "ቋሚነት",
        ],
    ),
    (
        "soil_fertility",
        [
            "soil fertility",
            "soil nutrient",
            "soil acid",
            "soil acidity",
            "soil ph",
            "acidic soil",
            "acidity",
            "acidic",
            "fertility",
            "ማዳበሪያ",
            "ዩሪያ",
            "ዲኤፒ",
            "ኮምፖስት",
            "ፍግ",
            "ናይትሮጂን",
            "ፎስፈረስ",
            "ንጥረ ነገር",
            "nutrient",
            "nutrients",
            "urea",
            "dap",
            "npk",
            "compost",
            "manure",
            "organic matter",
            "ph",
            "lime",
            "liming",
            "የመሬት ሀብት",
            "አፈር",
            "የአፈር",
            "አሲዳማ",
            "አሲዳማነት",
            "ፒኤች",
            "ኖራ",
            "የኖራ",
            "isfm",
        ],
    ),
    (
        "pest_disease",
        [
            "pest",
            "insect",
            "insects",
            "weed",
            "weeds",
            "vector",
            "disease",
            "fungus",
            "fungal",
            "rust",
            "leaf spot",
            "blight",
            "aphid",
            "armyworm",
            "fall armyworm",
            "locust",
            "ፀረ",
            "ተባይ",
            "በሽታ",
            "አረም",
            "ፈንገስ",
            "ፈንጋይ",
            "ቅጠል",
            "ዝገት",
            "wheat value chain",
            "pvmp",
        ],
    ),
    (
        "land_characterization",
        [
            "landpks",
            "soil type",
            "land type",
            "classification",
            "land classification",
            "land capability",
            "land suitability",
            "slope",
            "texture",
            "የመሬት አይነት",
            "የአፈር አይነት",
            "የመሬት ምድብ",
            "የመሬት ተስማሚነት",
            "ተዳፋት",
            "land potential",
        ],
    ),
    (
        "extension_advisory",
        [
            "extension",
            "ማስፋፊያ",
            "ቁሳቁስ",
            "ቁሳቁሶች",
            "መመሪያ",
            "መምሪያ",
            "ማኑዋል",
            "ስልጠና",
            "የመስክ ትምህርት",
            "ፖስተር",
            "ፍሊፕ",
            "ፍሊፕ መጽሐፍ",
            "የውይይት ቡድን",
            "የመስክ ጉብኝት",
            "material",
            "materials",
            "manual",
            "guide",
            "guideline",
            "training",
            "field visit",
            "demonstration",
            "ttl",
            "da ",
            "development agent",
        ],
    ),
    (
        "crop_production",
        [
            "crop option",
            "crop production",
            "production package",
            "ሰብል",
            "ሰብሎች",
            "planting",
            "sowing",
            "seed",
            "seeding",
            "spacing",
            "variety",
            "nursery",
            "production",
            "grow",
            "መዝራት",
            "ዘር",
            "ዝርያ",
            "ርቀት",
            "መትከል",
            "ማምረት",
            "cultivar",
            "lowland",
            "irrigation",
            "መስኖ",
        ],
    ),
]

# Short Amharic hints appended only for embedding search; these are not shown to the user.
_RETRIEVAL_HINTS: dict[str, str] = {
    "post_harvest": "ድህረ ምርት አያያዝ ማከማቻ እና ኪሳራ መቀነስ",
    "soil_water_conservation": "የአፈር እና የውሃ ጥበቃ መሸርሸር መከላከል እና እርከን",
    "soil_fertility": "የአፈር ለምነት ማዳበሪያ ኮምፖስት ኖራ እና ISFM",
    "pest_disease": "የሰብል ተባይ በሽታ አረም እና መከላከል",
    "land_characterization": "የመሬት ምድብ የአፈር አይነት ተዳፋት እና LandPKS",
    "extension_advisory": "የግብርና ማስፋፊያ መመሪያ ስልጠና እና የመስክ ጉብኝት",
    "weather_advice": "የአየር ሁኔታ ዝናብ ትንበያ ድርቅ እና የሰብል ምክር",
    "crop_production": "የሰብል ምርት ዘር ዝርያ መዝራት ርቀት እና መስኖ",
    "general_agronomy": "የግብርና ምክር የሰብል አያያዝ እና ልምድ",
}

# Which information is required before giving a useful answer.
# This controls short clarification questions.
REQUIRED_SLOTS: dict[str, list[str]] = {
    "weather_advice": ["location"],
    "market_price": ["crop", "location"],
    "crop_production": ["crop", "region"],
    "soil_fertility": ["crop", "region"],
    "pest_disease": ["crop"],
    "general_agronomy": ["crop"],
}

SLOT_QUESTIONS: dict[str, str] = {
    "crop": "ለየትኛው ሰብል ነው ጥያቄዎ? (ለምሳሌ፦ ስንዴ፣ ጤፍ፣ በቆሎ፣ ድንች)",
    "region": "ለየትኛው የምርት አካባቢ ነው? (ደጋ፣ ቆላ፣ ወይና ደጋ ወይም መስኖ)",
    "location": "ለየትኛው አካባቢ ነው? ከተማውን ወይም ዞኑን ይግለጹ።",
}


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class NLUResult:
    primary_intent: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""

    def to_dict(self, include_retrieval_query: bool = False) -> dict[str, Any]:
        data = {
            "primary_intent": self.primary_intent,
            "confidence": round(self.confidence, 3),
            "entities": self.entities,
        }
        if include_retrieval_query:
            data["retrieval_query"] = self.retrieval_query
        return data


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _contains(text: str, lower: str, keyword: str) -> bool:
    """Case-insensitive containment helper for intent keywords."""
    kw_lower = keyword.lower()
    return kw_lower in lower or keyword in text


def _looks_amharic(keyword: str) -> bool:
    return any("\u1200" <= ch <= "\u137f" for ch in keyword)


def _contains_entity(text: str, lower: str, keyword: str, *, allow_suffix: bool = True) -> bool:
    """
    Safer entity matching for Amharic words.

    It prevents false matches such as location 'ሲዳማ' inside 'አሲዳማነት',
    while still allowing common prefixes such as 'በ', 'የ', 'ከ', and 'ለ'.
    """
    if not _looks_amharic(keyword):
        return keyword.lower() in lower

    allowed_before = set(" \n\t፣።,.?!;:()[]{}<>/\\|-_") | {"በ", "የ", "ከ", "ለ"}
    boundary_after = set(" \n\t፣።,.?!;:()[]{}<>/\\|-_")

    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx == -1:
            return False

        before_ok = idx == 0 or text[idx - 1] in allowed_before
        end = idx + len(keyword)
        after_ok = end == len(text) or allow_suffix or text[end] in boundary_after

        if before_ok and after_ok:
            return True
        start = idx + 1


def _find_first_keyword(
    text: str,
    keywords: dict[str, str],
    *,
    allow_suffix: bool = True,
) -> Optional[tuple[str, str]]:
    """Find the longest matching keyword first to avoid matching 'ደጋ' before 'ወይና ደጋ'."""
    lower = text.lower()
    for kw in sorted(keywords.keys(), key=len, reverse=True):
        if _contains_entity(text, lower, kw, allow_suffix=allow_suffix):
            return kw, keywords[kw]
    return None


def _extract_crop_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    match = _find_first_keyword(text, CROP_KEYWORDS)
    if match:
        kw, crop_en = match
        out["crop_en"] = crop_en
        out["crop_keyword"] = kw
    return out


def _extract_region_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    region_match = _find_first_keyword(text, REGION_KEYWORDS)
    if region_match:
        kw, reg_en = region_match
        out["region_en"] = reg_en
        out["region_keyword"] = kw

    location_match = _find_first_keyword(text, LOCATION_KEYWORDS, allow_suffix=False)
    if location_match:
        kw, loc_en = location_match
        out["location_en"] = loc_en
        out["location_keyword"] = kw

    return out


def _is_market_price_intent(text: str, lower: str) -> bool:
    """True when the farmer is asking commodity/market price, not rate/dose."""
    has_market = any(_contains(text, lower, k) for k in MARKET_KEYWORDS)
    if not has_market:
        return False

    has_dose = any(_contains(text, lower, s) for s in _NON_MARKET_DOSE_SIGNALS)

    # If the text has both market and dose signals, require a strong price word.
    # Example: "ዩሪያ ዋጋ" = market; "ዩሪያ መጠን" = not market.
    strong_market_terms = ("ዋጋ", "ገበያ", "price", "market", "cost", "ብር", "ሽያጭ")
    if has_dose and not any(_contains(text, lower, m) for m in strong_market_terms):
        return False

    return True


def _score_topic(text: str, lower: str, keywords: list[str]) -> tuple[int, list[str]]:
    matched: list[str] = []
    for kw in keywords:
        # Ignore extremely short English tokens because they create noisy matches.
        if len(kw.strip()) <= 2:
            continue
        if _contains(text, lower, kw):
            matched.append(kw)
    return len(matched), matched


def _has_any_keyword(text: str, lower: str, keywords: set[str] | list[str]) -> bool:
    return any(_contains(text, lower, k) for k in keywords)


def _build_retrieval_query(question: str, intent: str) -> str:
    """Original question + optional Amharic retrieval hint."""
    if intent == "unknown":
        return question
    hint = _RETRIEVAL_HINTS.get(intent, _RETRIEVAL_HINTS["general_agronomy"])
    return f"{question}\n{hint}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def analyze_intent(text: str) -> NLUResult:
    """
    Classify farmer question intent and build a retrieval query.

    The confidence value is a rule-confidence estimate, not a trained probability.
    It is used for routing/fallback decisions only.
    """
    stripped = (text or "").strip()
    if not stripped:
        return NLUResult("unknown", 0.0, {}, "")

    lower = stripped.lower()
    entities = _extract_crop_entities(stripped)
    entities.update(_extract_region_entities(stripped))

    # Market price has its own data path in the main app.
    if _is_market_price_intent(stripped, lower):
        entities["matched_keywords"] = [k for k in MARKET_KEYWORDS if _contains(stripped, lower, k)]
        conf = 0.9 if entities.get("crop_en") else 0.72
        return NLUResult("market_price", conf, entities, stripped)

    best_intent = "general_agronomy"
    best_score = 0
    best_matches: list[str] = []

    for intent_id, keywords in _TOPIC_RULES:
        score, matched = _score_topic(stripped, lower, keywords)
        if score > best_score:
            best_score = score
            best_intent = intent_id
            best_matches = matched

    if best_score == 0:
        has_agri_signal = _has_any_keyword(stripped, lower, AGRI_INTENT_KEYWORDS) or bool(entities.get("crop_en"))
        if has_agri_signal:
            best_intent = "general_agronomy"
            best_score = 1
            best_matches = ["weak_agri_signal"]
            conf = 0.42
        else:
            best_intent = "unknown"
            conf = 0.28
    else:
        conf = min(0.93, 0.38 + 0.11 * best_score)

    if best_matches:
        entities["matched_keywords"] = best_matches

    retrieval = _build_retrieval_query(stripped, best_intent)

    return NLUResult(
        primary_intent=best_intent,
        confidence=conf,
        entities=entities,
        retrieval_query=retrieval,
    )


def needs_slot_filling(text: str, session_state: Optional[dict], nlu: NLUResult) -> Optional[str]:
    """
    Return a short clarification question when required information is missing.

    This function is intentionally conservative: it avoids asking unnecessary questions
    for post-harvest, conservation, land-characterization, and extension-material queries.
    """
    if session_state and session_state.get("current_state") != "active":
        return None

    if nlu.primary_intent == "unknown":
        return None

    stripped = (text or "").strip()
    lower = stripped.lower()

    required_slots = REQUIRED_SLOTS.get(nlu.primary_intent, [])
    if not required_slots:
        return None

    has_crop = bool(nlu.entities.get("crop_en")) or _has_any_keyword(stripped, lower, CROP_ENTITY_WORDS)
    has_region = bool(nlu.entities.get("region_en")) or _has_any_keyword(stripped, lower, REGION_ENTITY_WORDS)
    has_location = bool(nlu.entities.get("location_en")) or _has_any_keyword(stripped, lower, LOCATION_ENTITY_WORDS)

    slot_present = {
        "crop": has_crop,
        "region": has_region,
        "location": has_location,
    }

    # Ask only one clarification at a time, in the order defined in REQUIRED_SLOTS.
    for slot in required_slots:
        if not slot_present.get(slot, False):
            return SLOT_QUESTIONS[slot]

    return None


if __name__ == "__main__":
    # Small smoke test. Remove this block if importing only from another app.
    examples = [
        "ስንዴ ላይ ዝገት በሽታ እንዴት እከላከላለሁ?",
        "በቆሎ ዋጋ በአዲስ አበባ ስንት ነው?",
        "የድንች መዝራት ርቀት በደጋ ስንት ነው?",
        "ነገ ዝናብ አለ?",
        "የአፈር አሲዳማነት እንዴት ይታከማል?",
    ]
    for q in examples:
        result = analyze_intent(q)
        print(q)
        print(result.to_dict(include_retrieval_query=True))
        print("slot:", needs_slot_filling(q, {"current_state": "active"}, result))
        print("-" * 80)