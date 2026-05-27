from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path


BASE_DOMAIN_TERMS = [
    # Crops and agricultural products — base forms
    "ስንዴ", "በቆሎ", "ጤፍ", "ገብስ", "ማሽላ", "ሰርገኛ",
    "ቲማቲም", "ድንች", "ሽንኩርት", "ቡና", "ሰሊጥ", "ምስር", "ሽምብራ",
    "ቦሎቄ", "አተር", "ምርት", "ሰብል", "እህል", "ጥራጥሬ", "ቅባት ሰብል",

    # Inflected / prefixed crop forms that Whisper commonly outputs
    # ስንዴ (wheat)
    "ለስንዴ", "ስንዴው", "ስንዴዎ", "ስንዴን", "ስንዴዎን", "የስንዴ", "ስንዴዎቻቸው",
    # በቆሎ (maize)
    "የበቆሎ", "ለበቆሎ", "በቆሎውን", "በቆሎዎ", "በቆሎ ማሳ",
    # ጤፍ (teff)
    "የጤፍ", "ለጤፍ", "ጤፉ", "ጤፍን",
    # ገብስ (barley)
    "የገብስ", "ለገብስ", "ገብሱ", "ገብሱን",
    # ማሽላ (sorghum)
    "የማሽላ", "ለማሽላ", "ማሽላዎ", "ማሽላውን",

    # Soil, fertility, and water — base forms
    "አፈር", "የአፈር", "የአፈር ለምነት", "የአፈር አሲዳማነት", "አሲዳማነት",
    "አፈር አሲዳማነት", "ፒኤች", "ኖራ", "ምልክት", "ማዳበሪያ", "ዩሪያ", "ዳፕ", "ኮምፖስት",
    "ናይትሮጅን", "ፎስፈረስ", "ፖታሽየም", "ዘር", "ችግኝ", "መስኖ",
    "ውሃ", "እርጥበት", "የውሃ ጥበቃ", "እርከን", "ተፋሰስ",

    # Inflected soil/input forms
    "አፈሩን", "አፈሩ", "ለአፈር", "የማዳበሪያ", "ማዳበሪያውን", "ማዳበሪያ ዓይነት",
    "ዩሪያ ማዳበሪያ", "ዳፕ ማዳበሪያ", "ለዩሪያ", "ኮምፖስቱ", "የዘር",
    "ዘሩን", "ዘሩ", "ዘር ቤት", "ዘር ምርጫ",
    "ውሃ ማጠጣት", "መስኖ ልማት", "ውሃ ጥበቃ",

    # Pest, disease, and plant health — base forms
    "ተባይ", "በሽታ", "ፈንገስ", "ቅጠል", "ሥር", "ስር", "አረም",
    "ፀረ ተባይ", "ጸረ ተባይ", "አረም መከላከል", "የተክል በሽታ",
    "ትል", "ቅማል", "ሻጋታ", "ዋግ", "ዝገት", "አንበጣ",

    # Inflected pest/disease forms
    "ፀረ ተባይ መድሃኒት", "ጸረ ተባይ ኬሚካል", "ፀረ ተባይ ዓይነት",
    "ተባዩን", "ተባዩ", "ከተባይ", "ለተባይ",
    "አረሙን", "አረሙ", "ከአረም", "አረምን",
    "ቅጠሉ", "ቅጠሉን", "ቅጠል ቀለም",

    # Farming operations — common verb-noun forms
    "ማሳ", "ማሳዎ", "ማሳውን", "ማሳ ዝግጅት", "ያሳዎ",
    "ማረስ", "ይህርሱ", "ያርሱ", "ዘር መዝራት", "ዘር ከመዝራት",
    "ምርት ማሰባሰብ", "ምርት ማጨድ", "ድህረ ምርት", "ምርት ቅነሳ",
    "አትክልት ስራ", "የሰብል አመራረት",

    # Market and units — base forms
    "ዋጋ", "ገበያ", "ሽያጭ", "ግዢ", "ኩንታል", "ኪሎ", "ኪሎ ግራም",
    "ግራም", "ሊትር", "ሄክታር", "ብር", "ቶን",

    # Inflected market/unit forms
    "ዋጋው", "ዋጋውን", "ዋጋ ምን", "ለኩንታል", "ኩንታሉ",
    "በሄክታር", "ለሄክታር", "ሄክታሩ",

    # Common English/transliterated agricultural terms that Whisper may emit
    "wheat", "maize", "teff", "barley", "sorghum", "coffee", "fertilizer",
    "urea", "dap", "nps", "compost", "irrigation", "pest", "disease", "market",
    "price", "soil", "soil acidity", "acidic soil", "soil ph", "seed", "harvest", "post harvest",
]


COMMON_AMHARIC_CONVERSATION_TERMS = [
    # Greetings and closings
    "ሰላም", "ሰላም ነው", "እንደምን አደሩ", "እንደምን ዋሉ", "እንዴት ነዎት",
    "አመሰግናለሁ", "እናመሰግናለን", "ደህና ሁኑ", "በሰላም", "ቻው",
    "እሺ", "አዎ", "አይ", "አይደለም", "እባክዎ", "ይቅርታ",

    # Interruptions, repairs, and call control language
    "ቆይ", "ቆይ ቆይ", "ተው", "አቁም", "ይቁም", "ድገም", "እንደገና",
    "አልሰማሁም", "አልገባኝም", "በድጋሚ", "ሌላ ጥያቄ", "ቀጥል",
    "መልስ", "ጥያቄ", "ልጠይቅ", "ልጠይቅዎት", "ባለሙያ", "ሰው",

    # Frequently used farmer phrasing
    "ምን ላድርግ", "እንዴት ላድርግ", "መቼ", "ስንት", "የት", "ለምን",
    "እችላለሁ", "አለብኝ", "ይሆናል", "አሁን", "ነገ", "ዛሬ", "በቅርቡ",
]


_TERM_SPLIT_RE = re.compile(r"[\s,;:/\\|()[\]{}<>\"'“”‘’`~!@#$%^&*_+=?፣።፤፦፧፨\-.]+")
_AMHARIC_RE = re.compile(r"[\u1200-\u137f]{2,}")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


def _configured_kb_paths() -> list[Path]:
    raw = os.getenv(
        "ASR_DOMAIN_TERMS_PATHS",
        "/app/RAG/KB,/app/kb_documents/amharic",
    )
    paths: list[Path] = []
    for part in raw.replace(";", ",").split(","):
        value = part.strip()
        if value:
            paths.append(Path(value))
    return paths


def _add_term(terms: set[str], value: str) -> None:
    term = re.sub(r"\s+", " ", (value or "").strip())
    if len(term) < 2:
        return
    if len(term) > 64:
        return
    terms.add(term)


def _terms_from_text(text: str, max_terms: int) -> set[str]:
    terms: set[str] = set()
    for token in _AMHARIC_RE.findall(text or ""):
        _add_term(terms, token)
        if len(terms) >= max_terms:
            return terms
    for token in _LATIN_RE.findall(text or ""):
        _add_term(terms, token.lower())
        if len(terms) >= max_terms:
            return terms
    return terms


def _terms_from_filename(path: Path) -> set[str]:
    stem = path.stem.replace("_", " ").replace("-", " ")
    terms = _terms_from_text(stem, max_terms=80)
    for part in _TERM_SPLIT_RE.split(stem):
        _add_term(terms, part.lower() if part.isascii() else part)
    return terms


def _terms_from_small_text_file(path: Path, max_terms: int) -> set[str]:
    try:
        if path.suffix.lower() == ".jsonl":
            terms: set[str] = set()
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(terms) >= max_terms:
                        break
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for key in ("title", "topic", "crop", "text_am", "text"):
                        value = item.get(key)
                        if isinstance(value, str):
                            terms.update(_terms_from_text(value[:2500], max_terms=max_terms - len(terms)))
            return terms
        if path.suffix.lower() in (".txt", ".md"):
            return _terms_from_text(path.read_text(encoding="utf-8", errors="ignore")[:20000], max_terms)
    except OSError:
        return set()
    return set()


def _kb_terms() -> set[str]:
    max_files = int(os.getenv("ASR_DOMAIN_TERMS_MAX_FILES", "160") or "160")
    max_terms = int(os.getenv("ASR_DOMAIN_TERMS_MAX_TERMS", "1800") or "1800")
    terms: set[str] = set()

    for folder in _configured_kb_paths():
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower())[:max_files]:
            if not path.is_file() or path.name.startswith("."):
                continue
            terms.update(_terms_from_filename(path))
            terms.update(_terms_from_small_text_file(path, max_terms=max(0, max_terms - len(terms))))
            if len(terms) >= max_terms:
                return set(list(terms)[:max_terms])
    return terms


@lru_cache(maxsize=1)
def get_domain_terms() -> list[str]:
    terms: set[str] = set()
    for term in BASE_DOMAIN_TERMS:
        _add_term(terms, term)
    terms.update(_kb_terms())
    return sorted(terms, key=lambda t: (-len(t), t))


@lru_cache(maxsize=1)
def get_asr_vocabulary() -> list[str]:
    terms = set(get_domain_terms())
    for term in COMMON_AMHARIC_CONVERSATION_TERMS:
        _add_term(terms, term)
    return sorted(terms, key=lambda t: (-len(t), t))


def refresh_domain_terms() -> None:
    get_domain_terms.cache_clear()
    get_asr_vocabulary.cache_clear()


# Backwards-compatible constant for older imports. New code should call
# get_domain_terms()/get_asr_vocabulary() so KB changes can be picked up.
DOMAIN_TERMS = get_asr_vocabulary()