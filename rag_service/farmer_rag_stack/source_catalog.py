"""Supplemental Ethiopian agriculture data-source catalog.

These source hints do not replace local KB retrieval or live tools. They provide
extra context for query rewriting and final grounding when the farmer asks about
soil, fertilizer response, crop trials, production statistics, or related topics.
"""

from __future__ import annotations

import re
from typing import Any


SUPPLEMENTAL_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "moa_agri_data_hub",
        "name": "Ethiopian National Agri Data Hub / EIAR Open Research Data",
        "url": "https://data.moa.gov.et",
        "priority": "supplemental",
        "topics": ("fertilizer_response", "yield", "agronomy_trials", "crop_research", "pest_disease", "soil"),
        "keywords": (
            "fertilizer response",
            "fertilizer",
            "yield response",
            "NPK",
            "nitrogen",
            "phosphorus",
            "potassium",
            "crop trial",
            "agronomy",
            "disease",
            "pest",
            "ማዳበሪያ",
            "ምርት",
            "ሙከራ",
            "ተባይ",
            "በሽታ",
        ),
        "use_for": (
            "Crop research datasets, fertilizer-response experiments, crop yield data, "
            "agronomy trials, crop disease/pest datasets, and soil/agronomy datasets. "
            "For fertilizer-response questions, prefer datasets with region, crop, soil type, "
            "N/P/K rates, control yield, and fertilized yield response."
        ),
        "search_terms": "data.moa.gov.et EIAR fertilizer response Ethiopia crop yield N P K soil type",
    },
    {
        "id": "nsis_ethiosis_geonode",
        "name": "NSIS / EthioSIS GeoNode layers",
        "url": "https://nsis.moa.gov.et",
        "priority": "supplemental",
        "topics": ("soil", "soil_fertility", "fertilizer_recommendation", "soil_health"),
        "keywords": (
            "NSIS",
            "EthioSIS",
            "GeoNode",
            "total nitrogen",
            "soil nutrient",
            "soil fertility",
            "soil health",
            "fertilizer recommendation",
            "አፈር",
            "ናይትሮጂን",
            "ማዳበሪያ",
            "ፒኤች",
        ),
        "use_for": (
            "Soil nutrient layers, fertility status, fertilizer recommendation context, "
            "and soil-health maps. Treat modelled layers as context; do not invent a field-specific rate."
        ),
        "search_terms": "nsis.moa.gov.et EthioSIS GeoNode total nitrogen soil fertility Ethiopia",
    },
    {
        "id": "ethiopia_lsc_hub",
        "name": "Ethiopia Land Soil Crop Hub",
        "url": "https://ethiopia.lsc-hubs.org",
        "priority": "supplemental",
        "topics": ("land", "soil", "crop", "climate", "water", "yield_prediction"),
        "keywords": (
            "land soil crop",
            "LSC",
            "climate",
            "water",
            "soil water",
            "yield prediction",
            "field study",
            "map service",
            "መሬት",
            "አፈር",
            "ውሃ",
            "ዝናብ",
        ),
        "use_for": (
            "Broader land, soil, crop, climate, and water datasets including field-study data, "
            "aerial/space observations, predictive outputs, and map/data services."
        ),
        "search_terms": "ethiopia.lsc-hubs.org land soil crop climate water yield datasets Ethiopia",
    },
    {
        "id": "eiar_lsc_hub",
        "name": "EIAR Land, Soil and Crop Hub",
        "url": "https://lsc-hub.eiar.gov.et",
        "priority": "supplemental",
        "topics": ("land", "soil", "crop", "climate_smart_agriculture", "maps", "dashboards"),
        "keywords": (
            "EIAR LSC",
            "climate-smart",
            "GeoStory",
            "dashboard",
            "map",
            "land soil crop",
            "ካርታ",
            "መሬት",
            "አፈር",
        ),
        "use_for": (
            "Ethiopia-focused datasets, maps, GeoStories, and dashboards supporting land, "
            "soil, and crop information for climate-smart agriculture decisions."
        ),
        "search_terms": "lsc-hub.eiar.gov.et EIAR land soil crop climate-smart agriculture maps dashboard",
    },
    {
        "id": "lsc_coalition_catalog",
        "name": "Coalition of the Willing / LSC catalog datasets",
        "url": "https://ethiopia.lsc-hubs.org",
        "priority": "supplemental_restricted_possible",
        "topics": ("fertilizer_response", "npks_trials", "legacy_soil_profiles", "crop_response"),
        "keywords": (
            "OARI NPKS",
            "ATA balanced fertilizer",
            "SARI NPKS",
            "crop response",
            "legacy soil profile",
            "fertilizer trial",
            "NPKS",
            "ማዳበሪያ",
            "ሙከራ",
        ),
        "use_for": (
            "Detailed fertilizer-response datasets such as OARI NPKS crop response, "
            "ATA balanced fertilizer crop response, and SARI NPKS. Some datasets may be restricted."
        ),
        "search_terms": "OARI NPKS ATA balanced fertilizer SARI NPKS crop response Ethiopia LSC catalog",
    },
    {
        "id": "ess_agriculture",
        "name": "Ethiopian Statistical Service Agriculture",
        "url": "https://www.ess.gov.et/agriculture",
        "priority": "supplemental_statistics",
        "topics": ("crop_statistics", "production", "area", "survey", "commercial_farm", "household_context"),
        "keywords": (
            "statistics",
            "production",
            "area",
            "survey",
            "commercial farm",
            "agricultural household",
            "ESS",
            "ስታቲስቲክስ",
            "ምርት",
            "ስፋት",
        ),
        "use_for": (
            "Official crop production, agricultural survey, commercial farm, and socio-economic "
            "agriculture background. Use for context, not field-level fertilizer rates."
        ),
        "search_terms": "ess.gov.et agriculture crop production area survey Ethiopia statistics",
    },
)


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u1200-\u137f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def relevant_supplemental_sources(question: str, *, limit: int = 4) -> list[dict[str, Any]]:
    q = _norm(question)
    if not q:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, src in enumerate(SUPPLEMENTAL_SOURCES):
        score = 0
        for kw in src.get("keywords", ()):
            if _norm(str(kw)) in q:
                score += 2
        for topic in src.get("topics", ()):
            if _norm(str(topic)) in q:
                score += 1
        if score:
            scored.append((score, -idx, src))
    scored.sort(reverse=True)
    return [src for _score, _idx, src in scored[: max(1, limit)]]


def supplemental_retrieval_terms(question: str) -> str:
    sources = relevant_supplemental_sources(question, limit=3)
    terms: list[str] = []
    for src in sources:
        terms.append(str(src.get("search_terms") or ""))
    return " ".join(t for t in terms if t).strip()


def supplemental_context_block(question: str, *, max_sources: int = 4) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for src in relevant_supplemental_sources(question, limit=max_sources):
        out.append(
            {
                "id": str(src.get("id") or ""),
                "name": str(src.get("name") or ""),
                "url": str(src.get("url") or ""),
                "priority": str(src.get("priority") or "supplemental"),
                "use_for": str(src.get("use_for") or ""),
            }
        )
    return out
