import json
import re
import hashlib
import pathlib
import urllib.parse
from datetime import datetime, UTC

import requests
import urllib3
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text


# =========================
# CONFIG
# =========================

BASE_DIR = pathlib.Path("data")

RAW_DIR = BASE_DIR / "raw" / "agronomy_expansion"
TEXT_DIR = BASE_DIR / "text" / "agronomy_expansion"
CHUNKS_DIR = BASE_DIR / "chunks"
LOG_DIR = BASE_DIR / "logs"

EXISTING_CLEAN_CHUNKS = CHUNKS_DIR / "agronomy_chunks_clean.jsonl"

EXPANSION_CHUNKS = CHUNKS_DIR / "agronomy_expansion_chunks.jsonl"
COMBINED_CHUNKS = CHUNKS_DIR / "agronomy_chunks_enhanced.jsonl"
REJECTED_CHUNKS = CHUNKS_DIR / "agronomy_expansion_rejected.jsonl"
REPORT_FILE = CHUNKS_DIR / "agronomy_expansion_report.json"

MOA_MANUALS_PAGE = "https://www.moa.gov.et/manuals-and-guidelines/"

# You had SSL certificate problems with moa.gov.et before.
# For local research downloading, this avoids failure.
# For production, set VERIFY_SSL = True.
VERIFY_SSL = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MIN_CHARS = 300
MAX_CHARS = 2600
OVERLAP_CHARS = 250


# =========================
# WHAT WE WANT TO ADD
# =========================

CROP_KEYWORDS = [
    "tef", "teff", "ጤፍ",
    "wheat", "durum", "bread", "irrigated wheat", "ስንዴ",
    "maize", "corn", "በቆሎ",
    "barley", "ገብስ",
    "sorghum", "ማሽላ",
    "rice", "ሩዝ",
    "finger millet",
    "millet",
    "sesame", "ሰሊጥ",
    "oil crops", "oil crop",
    "pulse", "pulses",
    "haricot", "bean",
    "chickpea",
    "lentil",
    "vegetable", "horticulture",
    "coffee", "ቡና",
]

PEST_SOIL_KEYWORDS = [
    "pest", "disease", "vector", "weed", "ipm",
    "fertilizer", "fertiliser", "soil", "nutrient",
    "irrigation", "agronomy", "post harvest", "post-harvest",
    "storage", "seed", "production package", "production manual",
]

# Trusted HTML pages that are useful as context.
# These are not all technical manuals, but they help the RAG know source systems.
TRUSTED_CONTEXT_PAGES = [
    {
        "title": "ATI Crops",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/crops/",
        "kb": "agronomy",
        "source_type": "html_program_context",
        "priority": "medium",
        "reason": "ATI crop transformation program context",
    },
    {
        "title": "ATI Production",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/production/",
        "kb": "agronomy",
        "source_type": "html_program_context",
        "priority": "medium",
        "reason": "production systems, FPC and irrigation context",
    },
    {
        "title": "ATI Digital Agriculture",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/digital-agriculture-2/",
        "kb": "advisory_system",
        "source_type": "html_program_context",
        "priority": "high",
        "reason": "8028 hotline, digital advisory, market information context",
    },
    {
        "title": "ATI National Market Information System",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/nmis/",
        "kb": "market",
        "source_type": "html_market_context",
        "priority": "high",
        "reason": "weekly agricultural market data source context",
    },
    {
        "title": "Ethiopian National Soil Information System",
        "source_org": "Ethiopian National Soil Information System / MoA",
        "url": "https://nsis.moa.gov.et/",
        "kb": "soil",
        "source_type": "html_soil_context",
        "priority": "high",
        "reason": "national soil information source context",
    },
    {
        "title": "Ethiopian Meteorological Institute",
        "source_org": "Ethiopian Meteorological Institute",
        "url": "https://www.ethiomet.gov.et/",
        "kb": "weather",
        "source_type": "html_weather_context",
        "priority": "high",
        "reason": "official weather and agrometeorology source context",
    },
    {
        "title": "Ethiopian Commodity Exchange",
        "source_org": "Ethiopian Commodity Exchange",
        "url": "https://www.ecx.com.et/",
        "kb": "market",
        "source_type": "html_market_context",
        "priority": "high",
        "reason": "commodity exchange market information source context",
    },
]


# =========================
# HELPERS
# =========================

def ensure_dirs():
    for d in [RAW_DIR, TEXT_DIR, CHUNKS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u1200-\u137F\s\-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:120] or "document"


def normalize_url(base_url: str, href: str) -> str:
    return urllib.parse.urljoin(base_url, href)


def is_pdf_url(url: str) -> bool:
    return ".pdf" in url.lower().split("?")[0]


def title_matches_needed_data(title: str, url: str) -> bool:
    combined = f"{title} {url}".lower()

    crop_hit = any(k.lower() in combined for k in CROP_KEYWORDS)
    support_hit = any(k.lower() in combined for k in PEST_SOIL_KEYWORDS)

    # We especially want crop manuals/packages, pest/soil/fertilizer/irrigation/postharvest.
    return crop_hit or support_hit


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # Fix broken line hyphenation
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Remove isolated page numbers
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        if re.fullmatch(r"page\s+\d+", line, flags=re.I):
            continue

        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\.{5,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def infer_language(text: str) -> str:
    am = sum(1 for c in text if "\u1200" <= c <= "\u137F")
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")

    if am > latin * 1.5:
        return "am"
    if latin > am * 1.5:
        return "en"
    return "mixed"


def looks_bad_or_noise(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return True

    lower = text.lower()

    dotted_refs = len(re.findall(r"\.{3,}\s*\d+", text))
    if dotted_refs >= 4:
        return True

    if "table of contents" in lower and dotted_refs >= 2:
        return True

    if text.count("|") > 30:
        return True

    words = len(text.split())
    sentence_marks = len(re.findall(r"[.!?።፧፨]", text))

    if words > 100 and sentence_marks == 0:
        return True

    return False


def trim_broken_edges(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    # Broken lowercase start from overlap
    if text[0].islower() or text[0] in ",.;:)]}":
        match = re.search(r"(?<=[.!?።፧፨])\s+", text[:450])
        if match:
            text = text[match.end():].strip()

    # Broken ending
    if text and text[-1] not in ".!?።፧፨":
        matches = list(re.finditer(r"[.!?።፧፨]", text))
        if matches:
            last = matches[-1].end()
            if len(text) - last < 350:
                text = text[:last].strip()

    return text


def chunk_text(text: str, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = (tail + "\n\n" + p).strip()

    if current:
        chunks.append(current)

    final = []

    for ch in chunks:
        if len(ch) <= max_chars:
            final.append(ch)
            continue

        sentences = re.split(r"(?<=[.!?።፧፨])\s+", ch)
        current = ""

        for s in sentences:
            if len(current) + len(s) + 1 <= max_chars:
                current = (current + " " + s).strip()
            else:
                if current:
                    final.append(current)
                tail = current[-overlap:] if current else ""
                current = (tail + " " + s).strip()

        if current:
            final.append(current)

    return [trim_broken_edges(c) for c in final if c.strip()]


def ethiopia_relevance(source_url: str, source_org: str, title: str, text: str) -> str:
    joined = f"{source_url} {source_org} {title} {text[:1000]}".lower()

    if "moa.gov.et" in joined:
        return "ethiopia_direct"
    if "ati.gov.et" in joined:
        return "ethiopia_direct"
    if "eiar.gov.et" in joined:
        return "ethiopia_direct"
    if "ethiomet.gov.et" in joined:
        return "ethiopia_direct"
    if "ecx.com.et" in joined:
        return "ethiopia_direct"
    if "ethiopia" in joined or "ethiopian" in joined:
        return "ethiopia_related"

    return "general_methodological"


def risk_tags(text: str) -> list[str]:
    lower = text.lower()
    tags = []

    pesticide_terms = [
        "pesticide", "insecticide", "fungicide", "herbicide", "rodenticide",
        "chemical", "spray", "spraying", "dosage", "dose", "application rate",
        "ppe", "personal protective", "poison", "toxicity",
        "ተባይ", "ኬሚካል", "መድሃኒት", "መርዝ", "ፀረ"
    ]

    fertilizer_terms = [
        "fertilizer", "fertiliser", "urea", "dap", "np", "nps", "compost",
        "manure", "soil nutrient", "የአፈር", "ማዳበሪያ"
    ]

    weather_terms = [
        "rainfall", "weather", "forecast", "drought", "moisture", "temperature",
        "ዝናብ", "አየር", "ድርቅ"
    ]

    market_terms = [
        "market", "price", "commodity", "exchange", "value chain",
        "ገበያ", "ዋጋ"
    ]

    if any(t in lower for t in pesticide_terms):
        tags.append("high_risk_pesticide_or_chemical")
    if any(t in lower for t in fertilizer_terms):
        tags.append("fertilizer_or_soil")
    if any(t in lower for t in weather_terms):
        tags.append("weather_or_climate")
    if any(t in lower for t in market_terms):
        tags.append("market")

    return tags


# =========================
# DISCOVERY
# =========================

def fetch_html(url: str) -> str:
    r = requests.get(
        url,
        timeout=90,
        verify=VERIFY_SSL,
        headers={"User-Agent": "ethiopia-farmer-advisory-rag/1.0"}
    )
    r.raise_for_status()
    return r.text


def discover_moa_manual_pdfs():
    print(f"Discovering MoA manuals from: {MOA_MANUALS_PAGE}")

    html = fetch_html(MOA_MANUALS_PAGE)
    soup = BeautifulSoup(html, "html.parser")

    discovered = {}

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(" ", strip=True)

        url = normalize_url(MOA_MANUALS_PAGE, href)

        if not is_pdf_url(url):
            continue

        title = text or pathlib.Path(urllib.parse.urlparse(url).path).stem
        title = re.sub(r"\s+", " ", title).strip()

        if not title_matches_needed_data(title, url):
            continue

        discovered[url] = {
            "title": title,
            "source_org": "FDRE Ministry of Agriculture",
            "url": url,
            "kb": "agronomy",
            "source_type": "pdf_manual_or_package",
            "priority": "high",
            "reason": "crop-specific or agronomy-support manual discovered from MoA manuals page",
        }

    # Add a few known MoA PDFs that may be found by search/index but not always exposed clearly.
    known_extra_moa_pdfs = [
        {
            "title": "Guideline on Irrigation Agronomy",
            "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/201109_Guideline-on-Irrigation-Agronomy-compressed.pdf",
        },
        {
            "title": "Maize Crop Guide",
            "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202112_Maize-compressed-1.pdf",
        },
        {
            "title": "Pest and Vector Management Plan for Ethiopia Wheat Value Chain Development Project",
            "url": "https://www.moa.gov.et/wp-content/uploads/2024/07/pvmp-ecs-wvcdp.pdf",
        },
    ]

    for item in known_extra_moa_pdfs:
        discovered[item["url"]] = {
            "title": item["title"],
            "source_org": "FDRE Ministry of Agriculture",
            "url": item["url"],
            "kb": "agronomy",
            "source_type": "pdf_manual_or_package",
            "priority": "high",
            "reason": "known relevant MoA PDF",
        }

    print(f"Discovered relevant MoA PDFs: {len(discovered)}")
    return list(discovered.values())


def build_source_list():
    moa_sources = discover_moa_manual_pdfs()
    all_sources = moa_sources + TRUSTED_CONTEXT_PAGES

    # Deduplicate by URL
    seen = set()
    unique = []

    for s in all_sources:
        if s["url"] in seen:
            continue
        seen.add(s["url"])
        unique.append(s)

    return unique


# =========================
# DOWNLOAD + EXTRACT
# =========================

def download_source(source, index):
    url = source["url"]
    title = source["title"]

    ext = ".pdf" if is_pdf_url(url) else ".html"
    filename = f"{index:03d}_{slugify(title)}{ext}"
    path = RAW_DIR / filename

    print(f"\n[{index}] Downloading: {title}")

    try:
        if path.exists() and path.stat().st_size > 0:
            print(f"  exists: {path}")
            return {**source, "local_path": str(path), "download_status": "exists"}

        r = requests.get(
            url,
            timeout=120,
            verify=VERIFY_SSL,
            headers={"User-Agent": "ethiopia-farmer-advisory-rag/1.0"},
            allow_redirects=True,
        )
        r.raise_for_status()

        path.write_bytes(r.content)
        print(f"  saved: {path}")

        return {**source, "local_path": str(path), "download_status": "ok"}

    except Exception as e:
        print(f"  failed: {e}")
        return {**source, "local_path": None, "download_status": "failed", "error": str(e)}


def extract_pdf(path: pathlib.Path) -> str:
    try:
        text = pdf_extract_text(str(path)) or ""
        return clean_text(text)
    except Exception as e:
        print(f"  PDF extraction failed: {e}")
        return ""


def extract_html(path: pathlib.Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    return clean_text(text)


def extract_source(downloaded):
    path_value = downloaded.get("local_path")
    if not path_value:
        return None

    path = pathlib.Path(path_value)

    print(f"Extracting: {path.name}")

    if path.suffix.lower() == ".pdf":
        text = extract_pdf(path)
    else:
        text = extract_html(path)

    text_path = TEXT_DIR / f"{path.stem}.txt"
    text_path.write_text(text, encoding="utf-8")

    downloaded["text_path"] = str(text_path)
    downloaded["char_count"] = len(text)
    downloaded["text_hash"] = sha256_text(text)
    downloaded["needs_review"] = len(text) < 600

    if downloaded["needs_review"]:
        print("  WARNING: short extracted text, may need OCR or manual review")

    print(f"  chars: {len(text)}")

    return {**downloaded, "text": text}


# =========================
# CHUNKING
# =========================

def make_chunks(extracted_sources):
    chunks = []
    rejected = []
    seen_hashes = set()

    for doc_i, src in enumerate(extracted_sources, start=1):
        if not src or not src.get("text"):
            continue

        text = src["text"]
        pieces = chunk_text(text)

        for chunk_i, piece in enumerate(pieces):
            piece = clean_text(piece)
            piece = trim_broken_edges(piece)

            if looks_bad_or_noise(piece):
                rejected.append({
                    "source_url": src.get("url"),
                    "title": src.get("title"),
                    "reason": "too_short_or_noisy",
                    "preview": piece[:500],
                })
                continue

            duplicate_key = re.sub(r"[^\w\u1200-\u137F]+", " ", piece.lower())
            duplicate_key = re.sub(r"\s+", " ", duplicate_key).strip()
            h = sha256_text(duplicate_key)

            if h in seen_hashes:
                rejected.append({
                    "source_url": src.get("url"),
                    "title": src.get("title"),
                    "reason": "duplicate",
                    "preview": piece[:500],
                })
                continue

            seen_hashes.add(h)

            chunk_id = f"agronomy_expansion_{doc_i:03d}_{chunk_i:04d}"

            record = {
                "id": chunk_id,
                "kb": src.get("kb", "agronomy"),
                "title": src.get("title"),
                "source_org": src.get("source_org"),
                "source_url": src.get("url"),
                "source_type": src.get("source_type"),
                "priority": src.get("priority"),
                "reason_added": src.get("reason"),
                "language_segment": infer_language(piece),
                "ethiopia_relevance": ethiopia_relevance(
                    src.get("url", ""),
                    src.get("source_org", ""),
                    src.get("title", ""),
                    piece,
                ),
                "risk_tags": risk_tags(piece),
                "text": piece,
                "char_count": len(piece),
                "word_count": len(piece.split()),
                "text_hash": h,
                "metadata": {
                    "local_path": src.get("local_path"),
                    "text_path": src.get("text_path"),
                    "download_status": src.get("download_status"),
                    "created_at": now_iso(),
                },
            }

            chunks.append(record)

    return chunks, rejected


def load_existing_chunks():
    if not EXISTING_CLEAN_CHUNKS.exists():
        print(f"No existing chunks found at {EXISTING_CLEAN_CHUNKS}. Only expansion chunks will be written.")
        return []

    records = []
    with open(EXISTING_CLEAN_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded existing cleaned chunks: {len(records)}")
    return records


def merge_chunks(existing, expansion):
    merged = []
    seen = set()

    for r in existing + expansion:
        text = r.get("text", "")
        key = re.sub(r"[^\w\u1200-\u137F]+", " ", text.lower())
        key = re.sub(r"\s+", " ", key).strip()

        if not key:
            continue

        h = sha256_text(key)

        if h in seen:
            continue

        seen.add(h)
        merged.append(r)

    return merged


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()

    print("Building missing Ethiopia agronomy KB expansion...")

    sources = build_source_list()

    source_manifest = LOG_DIR / "agronomy_expansion_sources.json"
    source_manifest.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nTotal sources selected: {len(sources)}")
    print(f"Source manifest: {source_manifest}")

    downloaded = []
    for i, src in enumerate(sources, start=1):
        downloaded.append(download_source(src, i))

    download_log = LOG_DIR / "agronomy_expansion_download_log.json"
    download_log.write_text(
        json.dumps(downloaded, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    extracted = []
    for d in downloaded:
        if d.get("download_status") in ["ok", "exists"]:
            item = extract_source(d)
            if item:
                extracted.append(item)

    expansion_chunks, rejected = make_chunks(extracted)

    existing_chunks = load_existing_chunks()
    combined_chunks = merge_chunks(existing_chunks, expansion_chunks)

    write_jsonl(EXPANSION_CHUNKS, expansion_chunks)
    write_jsonl(COMBINED_CHUNKS, combined_chunks)
    write_jsonl(REJECTED_CHUNKS, rejected)

    report = {
        "generated_at": now_iso(),
        "moa_manuals_page": MOA_MANUALS_PAGE,
        "sources_selected": len(sources),
        "downloaded_ok_or_exists": sum(1 for d in downloaded if d.get("download_status") in ["ok", "exists"]),
        "download_failed": sum(1 for d in downloaded if d.get("download_status") == "failed"),
        "extracted_sources": len(extracted),
        "existing_chunks": len(existing_chunks),
        "expansion_chunks": len(expansion_chunks),
        "combined_chunks": len(combined_chunks),
        "rejected_chunks": len(rejected),
        "output_expansion": str(EXPANSION_CHUNKS),
        "output_combined": str(COMBINED_CHUNKS),
        "output_rejected": str(REJECTED_CHUNKS),
        "download_log": str(download_log),
        "source_manifest": str(source_manifest),
    }

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nDONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nUse this enhanced file for the next RAG step:\n{COMBINED_CHUNKS}")


if __name__ == "__main__":
    main()