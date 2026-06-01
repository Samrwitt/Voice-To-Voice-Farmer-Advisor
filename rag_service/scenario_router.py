"""Scenario routing helpers for farmer voice RAG.

This layer is intentionally rule-based and conservative. It decides whether a
voice turn should clarify, use a tool/KB route, or remain eligible for expert
handoff before the low-confidence KB guard can escalate normal questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


WEATHER_SIGNALS = (
    "weather",
    "forecast",
    "rain",
    "rainfall",
    "raining",
    "climate",
    "temperature",
    "humidity",
    "drought",
    "ዝናብ",
    "የዝናብ",
    "የአየር",
    "አየር",
    "ትንበያ",
    "ድርቅ",
    "ሙቀት",
)

FERTILIZER_SIGNALS = (
    "ማዳበሪያ",
    "ዩሪያ",
    "ኮምፖስት",
    "አፈር",
    "የአፈር",
    "አሲዳማ",
    "አሲዳማነት",
    "ፒኤች",
    "ኖራ",
    "fertilizer",
    "urea",
    "dap",
    "npk",
    "compost",
    "nutrient",
    "soil acidity",
    "soil acid",
    "soil ph",
    "acidic soil",
    "acidity",
    "acidic",
    "ph",
    "lime",
    "liming",
)

PEST_DISEASE_SIGNALS = (
    "ተባይ",
    "በሽታ",
    "ቅጠል",
    "አረም",
    "ፈንገስ",
    "ፈንጋይ",
    "ዝገት",
    "pest",
    "insect",
    "disease",
    "fungus",
    "rust",
    "spot",
    "blight",
    "aphid",
    "armyworm",
    "weed",
)

POST_HARVEST_SIGNALS = (
    "post-harvest",
    "postharvest",
    "post harvest",
    "after harvest",
    "storage",
    "drying",
    "threshing",
    "grain loss",
    "moisture",
    "ከመከር",
    "ድህረ ምርት",
    "ድህረ-ምርት",
    "ማከማቻ",
    "ማከማቸት",
    "ማጠራቀም",
    "ጎተራ",
    "ኪሳራ",
)

SOIL_WATER_SIGNALS = (
    "soil and water",
    "soil conservation",
    "water conservation",
    "soil erosion",
    "erosion",
    "terrace",
    "terracing",
    "watershed",
    "water harvesting",
    "runoff",
    "bund",
    "check dam",
    "የአፈር ጥበቃ",
    "የውሃ ጥበቃ",
    "የመሬት ጥበቃ",
    "መሸርሸር",
    "እርከን",
)

LAND_CHARACTERIZATION_SIGNALS = (
    "landpks",
    "soil type",
    "land type",
    "land classification",
    "land capability",
    "land suitability",
    "land potential",
    "slope",
    "texture",
    "classification",
    "የመሬት አይነት",
    "የአፈር አይነት",
    "የመሬት ምድብ",
    "የመሬት ተስማሚነት",
    "ተዳፋት",
)

EXTENSION_SIGNALS = (
    "extension",
    "development agent",
    "da ",
    "manual",
    "guide",
    "guideline",
    "training",
    "field visit",
    "demonstration",
    "ttl",
    "ማራዘም",
    "ማስፋፊያ",
    "መመሪያ",
    "መምሪያ",
    "ማኑዋል",
    "ስልጠና",
    "የመስክ ጉብኝት",
    "የመስክ ትምህርት",
    "ፖስተር",
    "ፍሊፕ",
)

FOLLOW_UP_SIGNALS = (
    "እና",
    "እሱ",
    "ይህ",
    "ያ",
    "በዚህ",
    "በላይ",
    "ከዚያ",
    "ሌላ",
    "ቀጥሎ",
    "also",
    "what about",
    "more",
    "again",
    "then",
    "that",
)

CROP_PRODUCTION_SIGNALS = (
    "እንዴት ይመረታል",
    "ይመረታል",
    "ለማምረት",
    "መዝራት",
    "ዘር",
    "plant",
    "grow",
    "produce",
    "production",
)

GENERAL_INFO_SIGNALS = (
    "ጥቅም",
    "ምንድን",
    "ምንድነው",
    "ምንድን ነው",
    "አሲዳማነት",
    "ፒኤች",
    "benefit",
    "what is",
    "meaning",
    "acidity",
    "soil acidity",
    "ph",
)


@dataclass
class ScenarioDecision:
    scenario: str
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    missing_slots: list[str] = field(default_factory=list)
    allow_low_conf_escalation: bool = True
    route_hint: str = "kb"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "needs_clarification": self.needs_clarification,
            "missing_slots": self.missing_slots,
            "allow_low_conf_escalation": self.allow_low_conf_escalation,
            "route_hint": self.route_hint,
            "details": self.details,
        }


def _has_any(text: str, lower: str, needles: tuple[str, ...]) -> bool:
    return any(n in lower or n in text for n in needles)


def _has_location(text: str, lower: str, nlu: Any, profile: dict | None, user_region: str | None) -> bool:
    entities = getattr(nlu, "entities", {}) or {}
    if user_region or entities.get("location") or entities.get("location_en") or entities.get("region_en"):
        return True
    if profile and (profile.get("location") or profile.get("latitude")):
        return True
    return _has_any(text, lower, ("አዲስ አበባ", "ኦሮሚያ", "አማራ", "ትግራይ", "ሲዳማ"))


def classify_voice_scenario(
    *,
    text: str,
    nlu: Any,
    profile: dict | None,
    user_region: str | None,
    history_pairs: list[tuple[str, str]],
    is_agrochemical: bool,
) -> ScenarioDecision:
    q = (text or "").strip()
    lower = q.lower()
    entities = getattr(nlu, "entities", {}) or {}
    primary = getattr(nlu, "primary_intent", "unknown")
    has_crop = bool(entities.get("crop_en"))

    if is_agrochemical:
        return ScenarioDecision(
            scenario="safety_agrochemical",
            allow_low_conf_escalation=True,
            route_hint="safety",
        )

    if primary == "market_price" or _has_any(q, lower, ("ዋጋ", "ገበያ", "price", "market", "ሽያጭ")):
        if not has_crop:
            return ScenarioDecision(
                scenario="market_price",
                needs_clarification=True,
                clarification_prompt="የየትኛው ሰብል ዋጋ ነው የሚፈልጉት? (ጤፍ፣ ስንዴ፣ በቆሎ፣ ወዘተ.)",
                missing_slots=["crop"],
                allow_low_conf_escalation=False,
                route_hint="market",
            )
        return ScenarioDecision("market_price", allow_low_conf_escalation=False, route_hint="market")

    if _has_any(q, lower, WEATHER_SIGNALS):
        if not _has_location(q, lower, nlu, profile, user_region):
            return ScenarioDecision(
                scenario="weather",
                needs_clarification=True,
                clarification_prompt="የአየር ሁኔታ ለየትኛው አካባቢ ነው የሚፈልጉት?",
                missing_slots=["location"],
                allow_low_conf_escalation=False,
                route_hint="weather",
            )
        return ScenarioDecision("weather", allow_low_conf_escalation=False, route_hint="weather")

    if primary == "soil_water_conservation" or _has_any(q, lower, SOIL_WATER_SIGNALS):
        return ScenarioDecision("soil_water_conservation", allow_low_conf_escalation=False, route_hint="kb")

    if primary == "land_characterization" or _has_any(q, lower, LAND_CHARACTERIZATION_SIGNALS):
        return ScenarioDecision("land_characterization", allow_low_conf_escalation=False, route_hint="kb")

    if primary == "soil_fertility" or _has_any(q, lower, FERTILIZER_SIGNALS):
        if _has_any(q, lower, GENERAL_INFO_SIGNALS):
            return ScenarioDecision("fertilizer", allow_low_conf_escalation=False, route_hint="kb_tool")
        missing: list[str] = []
        if not has_crop:
            missing.append("crop")
        if not user_region and not (profile or {}).get("location"):
            missing.append("region")
        if missing:
            fert_label = "ዩሪያ" if ("ዩሪያ" in q or "urea" in lower) else "ማዳበሪያ"
            prompt = (
                f"ለ{fert_label} ምክር ሰብል እና አካባቢውን ይንገሩኝ። "
                "ለምሳሌ፦ ስንዴ በደጋ።"
            )
            if missing == ["region"]:
                prompt = "ለየትኛው አካባቢ ነው? (ደጋ፣ ቆላ ወይም ወይና ደጋ)"
            return ScenarioDecision(
                scenario="fertilizer",
                needs_clarification=True,
                clarification_prompt=prompt,
                missing_slots=missing,
                allow_low_conf_escalation=False,
                route_hint="kb_tool",
            )
        return ScenarioDecision("fertilizer", allow_low_conf_escalation=False, route_hint="kb_tool")

    if primary == "pest_disease" or _has_any(q, lower, PEST_DISEASE_SIGNALS):
        if not has_crop:
            return ScenarioDecision(
                scenario="pest_disease",
                needs_clarification=True,
                clarification_prompt="ተባይ/በሽታው በየትኛው ሰብል ላይ ነው? (ስንዴ፣ ጤፍ፣ ቡና፣ ወዘተ.)",
                missing_slots=["crop"],
                allow_low_conf_escalation=False,
                route_hint="kb_tool",
            )
        return ScenarioDecision("pest_disease", allow_low_conf_escalation=False, route_hint="kb_tool")

    if primary == "post_harvest" or _has_any(q, lower, POST_HARVEST_SIGNALS):
        return ScenarioDecision("post_harvest", allow_low_conf_escalation=False, route_hint="kb")

    if primary == "extension_advisory" or _has_any(q, lower, EXTENSION_SIGNALS):
        return ScenarioDecision("extension_advisory", allow_low_conf_escalation=False, route_hint="kb")

    if primary in {"crop_production", "general_agronomy"} or (
        has_crop and _has_any(q, lower, CROP_PRODUCTION_SIGNALS)
    ):
        if not has_crop and not history_pairs:
            return ScenarioDecision(
                scenario="crop_production",
                needs_clarification=True,
                clarification_prompt="ጥያቄዎ ስለ የትኛው ሰብል ነው? (ስንዴ፣ ጤፍ፣ በቆሎ፣ ወዘተ.)",
                missing_slots=["crop"],
                allow_low_conf_escalation=False,
                route_hint="kb",
            )
        return ScenarioDecision("crop_production", allow_low_conf_escalation=False, route_hint="kb")

    if history_pairs and _has_any(q, lower, FOLLOW_UP_SIGNALS):
        return ScenarioDecision("follow_up", allow_low_conf_escalation=False, route_hint="kb")

    return ScenarioDecision(
        "unknown",
        needs_clarification=False,
        allow_low_conf_escalation=False,
        route_hint="clarify_or_fallback",
    )
