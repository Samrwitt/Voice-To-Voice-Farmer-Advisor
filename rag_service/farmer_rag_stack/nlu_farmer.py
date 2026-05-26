"""Lightweight Amharic NLU for farmer questions: crop + aspect (no external API).

Used to tighten Chroma retrieval (embedding query) and to pick crop/topic rules
when the wording is indirect. Disable with ``RAG_NLU=0``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Longer phrases first within each crop so «ለቡና» wins over «ቡና» when both match logic needs care.
_CROP_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "coffee",
        (
            "አረቢካ",
            "አረብካ",
            "አራቢካ",
            "ሮቡስታ",
            "ለቡና",
            "ቡናን",
            "ቡና",
            "ጎማ",
            "coffee",
        ),
    ),
    ("teff", ("ጤፍ", "ጣፍ", "teff")),
    ("wheat", ("ስንዴ", "wheat")),
    ("maize", ("በቆሎዬ", "በቆሎ", "maize", "corn")),
    ("sorghum", ("ማሽላ", "ቦርኬ", "ዳጉሳ", "sorghum")),
    ("sesame", ("ሰሊጥ", "ሰሊት", "sesame")),
    ("barley", ("ገብስ", "barley")),
    ("faba", ("ቡቃያ", "faba", "broad bean")),
    ("haricot", ("ቦሎቄ", "haricot bean", "common bean")),
    ("chickpea", ("ሽምብራ", "chickpea")),
    ("lentil", ("ምስር", "lentil")),
    ("potato", ("ድንች", "potato")),
    ("tomato", ("ቲማቲም", "tomato")),
    ("onion", ("ሽንኩርት", "onion")),
)

# Aspect keywords (Amharic + a few Latin). Score = number of hits; tie-break by list order (earlier = higher priority if equal).
_ASPECT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "price",
        (
            "ዋጋ",
            "ገበያ",
            "ግዢ",
            "ሽያጭ",
            "ዋጋ በ",
            "የገበያ",
            "market",
            "price",
            "nmis",
        ),
    ),
    (
        "altitude",
        (
            "ከፍታ",
            "ሜትር",
            "ምትር",
            "ከባህር ጠለል",
            "ከ ባህር",
            "ከባህር",
            "asl",
            "elevation",
        ),
    ),
    (
        "rainfall",
        (
            "ዝናብ",
            "የዝናብ",
            "ሚሜ",
            "ዝናቡ",
            "rain",
            "rainfall",
            "weather",
            "forecast",
            "drought",
        ),
    ),
    (
        "soil",
        (
            "አፈር",
            "ፒኤች",
            "አሲዳማ",
            "አሲዳማነት",
            "ኖራ",
            "ph",
            "acidity",
            "acidic",
            "lime",
            "liming",
            "የአፈር",
            "soil",
        ),
    ),
    (
        "conservation",
        (
            "የአፈር ጥበቃ",
            "የውሃ ጥበቃ",
            "መሸርሸር",
            "እርከን",
            "soil conservation",
            "water conservation",
            "erosion",
            "terrace",
            "watershed",
        ),
    ),
    (
        "land",
        (
            "የመሬት አይነት",
            "የአፈር አይነት",
            "የመሬት ምድብ",
            "landpks",
            "land type",
            "soil type",
            "classification",
            "slope",
            "texture",
        ),
    ),
    (
        "fertilizer",
        (
            "ማዳበሪያ",
            "ኮምፖስት",
            "ዩሪያ",
            "ናይትሮጂን",
            "ፎስፈረስ",
            "fertilizer",
            "urea",
            "dap",
            "npk",
            "compost",
        ),
    ),
    (
        "disease",
        (
            "በሽታ",
            "ተባይ",
            "ፈንጋይ",
            "ሩብ",
            "disease",
            "fungus",
            "rust",
            "aphid",
            "armyworm",
            "blight",
            "weed",
        ),
    ),
    (
        "pest",
        (
            "ብልትኝ",
            "አረም",
            "ጥቃቅ",
            "pest",
            "insect",
        ),
    ),
    (
        "planting",
        (
            "መትከል",
            "ዘር",
            "መዝራት",
            "መዝሪያ",
            "planting",
        ),
    ),
    (
        "harvest",
        (
            "መከር",
            "መሰብሰቢያ",
            "አፈራ",
            "harvest",
        ),
    ),
    (
        "yield",
        (
            "ምርት",
            "የምርት",
            "ሀብታም",
            "yield",
        ),
    ),
    (
        "storage",
        (
            "ማከማቻ",
            "መያዣ",
            "ማደስ",
            "ድህረ ምርት",
            "ከመከር",
            "storage",
            "postharvest",
            "post-harvest",
            "drying",
            "moisture",
        ),
    ),
    (
        "extension",
        (
            "ማራዘም",
            "ማስፋፊያ",
            "መመሪያ",
            "መምሪያ",
            "ማኑዋል",
            "ስልጠና",
            "extension",
            "development agent",
            "manual",
            "guideline",
            "training",
        ),
    ),
)


@dataclass(frozen=True)
class FarmerNLU:
    """Structured guess from the user utterance (rules only, no LLM)."""

    crop_id: str | None
    aspect: str | None
    retrieval_boost: str  # short Amharic / mixed fragment for embedding
    problem: str | None = None
    location: str | None = None
    season: str | None = None
    goal: str | None = None
    search_queries: tuple[str, ...] = ()


def _norm(s: str) -> str:
    return (s or "").strip()


def _detect_crop(q: str) -> str | None:
    for cid, phrases in _CROP_ORDER:
        for p in phrases:
            if p and p in q:
                return cid
    return None


def _detect_aspect(q: str) -> str | None:
    best: str | None = None
    best_score = 0
    for aid, kws in _ASPECT_SPECS:
        score = sum(1 for k in kws if k in q)
        if score > best_score:
            best_score = score
            best = aid
    return best


def _boost_for(crop_id: str | None, aspect: str | None) -> str:
    parts: list[str] = []
    crop_words = {
        "coffee": "ቡና",
        "teff": "ጤፍ",
        "wheat": "ስንዴ",
        "maize": "በቆሎ",
        "sorghum": "ማሽላ",
        "sesame": "ሰሊጥ",
        "barley": "ገብስ",
        "faba": "ቡቃያ",
        "haricot": "ቦሎቄ",
        "chickpea": "ሽምብራ",
        "lentil": "ምስር",
        "potato": "ድንች",
        "tomato": "ቲማቲም",
        "onion": "ሽንኩርት",
    }
    aspect_words = {
        "price": "ዋጋ ገበያ",
        "altitude": "ከፍታ ሜትር ከባህር ጠለል",
        "rainfall": "ዝናብ ሚሜ",
        "soil": "አፈር ፒኤች አሲዳማነት",
        "conservation": "አፈር ውሃ ጥበቃ መሸርሸር እርከን",
        "land": "የመሬት አይነት ምድብ LandPKS",
        "fertilizer": "ማዳበሪያ",
        "disease": "በሽታ",
        "pest": "ብልትኝ",
        "planting": "መትከል ዘር",
        "harvest": "መከር",
        "yield": "ምርት",
        "storage": "ማከማቻ",
        "extension": "ማስፋፊያ መመሪያ ስልጠና",
    }
    if crop_id and crop_id in crop_words:
        parts.append(crop_words[crop_id])
    if aspect and aspect in aspect_words:
        parts.append(aspect_words[aspect])
    # De-duplicate while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        for w in p.split():
            if w not in seen:
                seen.add(w)
                out.append(w)
    return " ".join(out[:14]).strip()


_KNOWN_LOCATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Arsi", ("አርሲ", "arsi", "asella")),
    ("Oromia", ("ኦሮሚያ", "oromia")),
    ("Amhara", ("አማራ", "amhara")),
    ("Hawassa", ("ሀዋሳ", "hawassa")),
    ("Sidama", ("ሲዳማ", "sidama")),
    ("Bale", ("ባሌ", "bale")),
    ("Jimma", ("ጅማ", "jimma")),
    ("Dire Dawa", ("ድሬዳዋ", "dire dawa")),
    ("Mekelle", ("መቀሌ", "mekelle")),
    ("Gondar", ("ጎንደር", "gondar")),
    ("Debre Birhan", ("ደብረ ብርሃን", "debre birhan")),
    ("Addis Ababa", ("አዲስ አበባ", "addis ababa")),
)


_SEASON_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("meher", ("መኸር", "meher", "main season", "rainy season", "ዝናብ")),
    ("belg", ("በልግ", "belg", "short rain")),
    ("dry", ("በጋ", "dry season", "drought", "ድርቅ")),
)


_GOAL_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sell_or_hold", ("ልሽጥ", "መሸጥ", "ሽጥ", "sell", "selling", "hold", "መጠበቅ")),
    ("diagnose", ("ምንድን", "ምን ነው", "ምልክት", "diagnose", "identify")),
    ("prevent", ("እንዳይመጣ", "መከላከል", "prevent", "avoid")),
    ("recommend", ("ምክር", "ምን ላድርግ", "recommend", "advice")),
    ("plant", ("መዝራት", "እዘራለሁ", "ዘር", "plant", "sow")),
    ("irrigate", ("መስኖ", "ውሃ", "irrigate", "irrigation")),
)


_PROBLEM_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rust_or_disease", ("በሽታ", "ዝገት", "rust", "disease", "fungus", "blight")),
    ("pest", ("ተባይ", "ብልትኝ", "armyworm", "aphid", "pest", "insect")),
    ("low_rainfall", ("ዝናብ ከቀነሰ", "ድርቅ", "rainfall", "drought", "water stress")),
    ("soil_acidity", ("አሲዳማ", "ፒኤች", "ph", "acidity", "acidic", "lime", "ኖራ")),
    ("fertility", ("ማዳበሪያ", "ዩሪያ", "dap", "nps", "fertilizer", "compost")),
    ("market_price", ("ዋጋ", "ገበያ", "price", "market", "ሽያጭ")),
)


_PROBLEM_QUERY_WORDS: dict[str, str] = {
    "rust_or_disease": "በሽታ ዝገት disease rust prevention symptoms",
    "pest": "ተባይ pest insect control prevention",
    "low_rainfall": "ዝናብ ድርቅ drought rainfall irrigation water stress",
    "soil_acidity": "አፈር pH አሲዳማነት acidity lime liming",
    "fertility": "ማዳበሪያ NPS UREA fertilizer compost soil fertility",
    "market_price": "ዋጋ ገበያ market price selling trend",
}


_GOAL_QUERY_WORDS: dict[str, str] = {
    "sell_or_hold": "መሸጥ መጠበቅ sell hold price trend",
    "diagnose": "ምልክት diagnosis identify symptoms",
    "prevent": "መከላከል prevention avoid",
    "recommend": "ምክር recommendation advice",
    "plant": "መዝራት seed planting sowing",
    "irrigate": "መስኖ irrigation water scheduling",
}


def _detect_first(q: str, specs: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    lower = q.lower()
    for label, terms in specs:
        if any(term in q or term in lower for term in terms):
            return label
    return None


def _detect_location(q: str) -> str | None:
    lower = q.lower()
    for label, terms in _KNOWN_LOCATIONS:
        if any(term in q or term in lower for term in terms):
            return label
    match = re.search(r"(?:^|\s)(?:በ|ለ)\s*([\u1200-\u137f ]{2,30}?)(?:\s+ላይ|\s+የ|\s+አካባቢ|$)", q)
    if not match:
        return None
    loc = match.group(1).strip(" ,.?።")
    if any(term in loc for term in ("ገበያ", "ዋጋ", "ስንት", "ነው")):
        return None
    return loc if 2 <= len(loc) <= 30 else None


def _dedupe_queries(queries: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        cleaned = re.sub(r"\s+", " ", (q or "").strip())
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out[:6])


def build_search_queries_for_nlu(base_query: str, nlu: FarmerNLU) -> tuple[str, ...]:
    """Rewrite an open farmer question into precise multilingual search queries."""
    crop_am = _CROP_LABEL_AM.get(nlu.crop_id or "", "")
    crop_en = nlu.crop_id or ""
    aspect_am = _ASPECT_LABEL_AM.get(nlu.aspect or "", "")
    problem_words = _PROBLEM_QUERY_WORDS.get(nlu.problem or "", "")
    goal_words = _GOAL_QUERY_WORDS.get(nlu.goal or "", "")
    loc = nlu.location or ""
    season = nlu.season or ""
    queries = [
        base_query,
        " ".join(p for p in (crop_am, aspect_am, problem_words, goal_words, loc, season) if p),
        " ".join(p for p in (crop_en, nlu.aspect or "", problem_words, goal_words, loc, season, "Ethiopia agriculture") if p),
        " ".join(p for p in (crop_am, crop_en, nlu.retrieval_boost, problem_words) if p),
    ]
    return _dedupe_queries(queries)


def parse_farmer_nlu(question: str) -> FarmerNLU:
    if os.environ.get("RAG_NLU", "1").strip().lower() in ("0", "false", "no", "off"):
        return FarmerNLU(crop_id=None, aspect=None, retrieval_boost="")
    q = _norm(question)
    if not q:
        return FarmerNLU(crop_id=None, aspect=None, retrieval_boost="")
    # Strip common UI prefixes so triggers still fire
    q2 = re.sub(r"^ጥያቄው\s*", "", q).strip()
    crop = _detect_crop(q2) or _detect_crop(q)
    aspect = _detect_aspect(q2) or _detect_aspect(q)
    problem = _detect_first(q2, _PROBLEM_TERMS) or _detect_first(q, _PROBLEM_TERMS) or aspect
    location = _detect_location(q2) or _detect_location(q)
    season = _detect_first(q2, _SEASON_TERMS) or _detect_first(q, _SEASON_TERMS)
    goal = _detect_first(q2, _GOAL_TERMS) or _detect_first(q, _GOAL_TERMS)
    boost = _boost_for(crop, aspect)
    partial = FarmerNLU(
        crop_id=crop,
        aspect=aspect,
        retrieval_boost=boost,
        problem=problem,
        location=location,
        season=season,
        goal=goal,
    )
    return FarmerNLU(
        crop_id=crop,
        aspect=aspect,
        retrieval_boost=boost,
        problem=problem,
        location=location,
        season=season,
        goal=goal,
        search_queries=build_search_queries_for_nlu(q2 or q, partial),
    )


def augment_retrieval_query_with_nlu(base_rq: str, nlu: FarmerNLU) -> str:
    """Append a short NLU hint line for dense/hybrid embedding (same script as the question)."""
    if not nlu.retrieval_boost:
        return base_rq.strip()
    return f"{base_rq.strip()}\n{nlu.retrieval_boost}".strip()


_CROP_LABEL_AM: dict[str, str] = {
    "coffee": "ቡና",
    "teff": "ጤፍ",
    "wheat": "ስንዴ",
    "maize": "በቆሎ",
    "sorghum": "ማሽላ",
    "sesame": "ሰሊጥ",
    "barley": "ገብስ",
    "faba": "ቡቃያ",
    "haricot": "ቦሎቄ",
    "chickpea": "ሽምብራ",
    "lentil": "ምስር",
    "potato": "ድንች",
    "tomato": "ቲማቲም",
    "onion": "ሽንኩርት",
}
_ASPECT_LABEL_AM: dict[str, str] = {
    "price": "ዋጋ/ገበያ",
    "altitude": "ከፍታ",
    "rainfall": "ዝናብ",
    "soil": "አፈር",
    "conservation": "አፈር/ውሃ ጥበቃ",
    "land": "የመሬት አይነት",
    "fertilizer": "ማዳበሪያ",
    "disease": "በሽታ",
    "pest": "ተባይ",
    "planting": "መትከል/ዘር",
    "harvest": "መከር",
    "yield": "ምርት",
    "storage": "ማከማቻ",
    "extension": "ማስፋፊያ",
}


def nlu_answer_scope_hint(nlu: FarmerNLU | None) -> str:
    """One short Amharic line for the LLM system prompt (optional)."""
    if not nlu or os.environ.get("RAG_NLU_PROMPT", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""
    c = _CROP_LABEL_AM.get(nlu.crop_id or "", "")
    a = _ASPECT_LABEL_AM.get(nlu.aspect or "", "")
    if c and a:
        return (
            f" (የጥያቄ ትርጉም፦ በዋናው ስለ «{c}» እና «{a}» ነው። "
            "የማይመለከቱ ክፍሎችን በመልስ አትጨምር።)"
        )
    if c:
        return f" (የጥያቄ ትርጉም፦ በዋናው ስለ «{c}» ነው።)"
    if a:
        return f" (የጥያቄ ትርጉም፦ በዋናው ስለ «{a}» ነው።)"
    return ""
