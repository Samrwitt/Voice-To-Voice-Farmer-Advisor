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
    boost = _boost_for(crop, aspect)
    return FarmerNLU(crop_id=crop, aspect=aspect, retrieval_boost=boost)


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
