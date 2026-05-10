import os
import re
import csv
import json
import hashlib
import pathlib
import unicodedata
import subprocess
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text


BASE_DIR = pathlib.Path("data")
RAW_DIR = BASE_DIR / "raw" / "agronomy"
TEXT_DIR = BASE_DIR / "text" / "agronomy"
CLEAN_DIR = BASE_DIR / "clean"
CHUNKS_DIR = BASE_DIR / "chunks"
MANIFEST_DIR = BASE_DIR / "manifests"

MASTER_MD = CLEAN_DIR / "master_agronomy_kb.md"
MASTER_TXT = CLEAN_DIR / "master_agronomy_kb.txt"
CHUNKS_JSONL = CHUNKS_DIR / "agronomy_chunks.jsonl"
SOURCES_CSV = MANIFEST_DIR / "agronomy_sources.csv"
DOWNLOAD_LOG = BASE_DIR / "logs" / "download_log.jsonl"


SOURCES = [
    {
        "title": "Use of Extension Materials Manual",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202112_Use-of-Extension-Materials-Manual-1.pdf",
        "language": "English",
        "type": "PDF Manual",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Guide on Lowland Contextualized Crop Options",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202201_Guide-on-Lowland-Contexualized-Crop-Options-compressed-1.pdf",
        "language": "English",
        "type": "PDF Guide",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Training Package on Soil Fertility Management Technologies for Woreda Experts",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/201411_Training-Package-on-Soil-Fertility-Management-Technologies-for-Woreda-experts-compressed-1.pdf",
        "language": "English",
        "type": "PDF Training Package",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Soil and Water Conservation in Ethiopia Guideline for Development Agents",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/201804_Soil-and-water-Conservation-in-Ethiopia-Guideline-for-Development-agents-compressed-1.pdf",
        "language": "English",
        "type": "PDF Guideline",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Biological Measures for Woreda Experts TTLM",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202308_Biological-Measures-for-Woreda-Experts_TTLM-compressed-1.pdf",
        "language": "English",
        "type": "PDF Training Material",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "LandPKS Amharic Manual",
        "source_org": "LandPotential / Ministry of Agriculture Ethiopia",
        "url": "https://landpotential.org/wp-content/uploads/2020/07/LandPKS-Amharic-manual.pdf",
        "language": "Amharic",
        "type": "PDF Manual",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "ISFM Participatory Learning Field Guide",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2025/01/20160711_ISFMParticipatory-Learning_-Field-Guide_English.pdf",
        "language": "English",
        "type": "PDF Field Guide",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "ISFM Technical Manual",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2025/01/20201124_ISFM_Technical-Manual_English_updated.pdf",
        "language": "English",
        "type": "PDF Technical Manual",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Agroecology Extension Training Manual",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2025/01/20220325_Agroecology-Manual.pdf",
        "language": "English",
        "type": "PDF Manual",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Post-harvest Manual FAO Amharic",
        "source_org": "FDRE Ministry of Agriculture / FAO",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/08/Post-harvest-manual-FAO-Amharic-3-compressed.pdf",
        "language": "Amharic",
        "type": "PDF Manual",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Postharvest Management Strategy in Grains in Ethiopia",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/08/PHM-strategy-Federal-Democratic-Republic-Ethiopia-compressed.pdf",
        "language": "English",
        "type": "PDF Strategy",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Plant Guide Trees Herbs and Grasses in the Lowlands of Ethiopia",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202203_Plant-Guide-Trees-Herbs-and-Grasses-in-the-Lowlands-of-Ethiopia_EN-compressed-1.pdf",
        "language": "English",
        "type": "PDF Plant Guide",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Irrigation Agronomy and Development Plan Training Material Part 3",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/202210_Training-Material-for-trainers-on-Irrigation-Agronomy-and-development-plan-Part3-compressed-1.pdf",
        "language": "English",
        "type": "PDF Training Material",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Pest and Vector Management Plan for Ethiopia Wheat Value Chain Development Project",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/07/pvmp-ecs-wvcdp.pdf",
        "language": "English",
        "type": "PDF Pest Management Plan",
        "kb": "agronomy",
        "priority": "high",
    },
    {
        "title": "Standardized Methods of Analysis for Soil Water Plant and Fertilizer",
        "source_org": "FDRE Ministry of Agriculture / CIAT / SSHI",
        "url": "https://www.moa.gov.et/wp-content/uploads/2025/03/202110_A-guide-to-standardized-methods-of-analysis-for-soil-water-plant-and-fertilizer_CIAT-SSHI.pdf",
        "language": "English",
        "type": "PDF Technical Guide",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Agroecological Practices under Smallholder Management in the Horn of Africa",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/2023_Agroecological-Practices-under-smallholder-management-in-the-Horn-of-Africa-compressed-1.pdf",
        "language": "English",
        "type": "PDF Report",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Cattle Urine Collection Handling and Application User Manual",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2025/01/20240102_Cattle-Urine-Manual_English-version.pdf",
        "language": "English",
        "type": "PDF Manual",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Soil Conservation and Rehabilitation for Food Security Baseline Report",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/11/20160316_Soil-Conservation-and-Rehabilitation-for-Food-Security_BaselineReport-compressed-1.pdf",
        "language": "English",
        "type": "PDF Report",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Postharvest Loss Assessment of Maize Wheat Sorghum and Haricot Bean",
        "source_org": "FDRE Ministry of Agriculture",
        "url": "https://www.moa.gov.et/wp-content/uploads/2024/12/27.-Postharvest-loss-assessment-of-crop.pdf",
        "language": "English",
        "type": "PDF Report",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Ethiopian Soil Types WRBRSG Dataset Page",
        "source_org": "Ethiopian National Agri Data Hub",
        "url": "https://data.moa.gov.et/dataset/ethiopian-soil-types-wrbrsg",
        "language": "English",
        "type": "HTML Dataset Page",
        "kb": "agronomy",
        "priority": "medium",
    },
    {
        "title": "Soil Nutrients Dataset Page",
        "source_org": "Ethiopian National Agri Data Hub",
        "url": "https://data.moa.gov.et/dataset/soil-nutrients",
        "language": "English",
        "type": "HTML Dataset Page",
        "kb": "agronomy",
        "priority": "medium",
    },
]


def ensure_dirs():
    for d in [RAW_DIR, TEXT_DIR, CLEAN_DIR, CHUNKS_DIR, MANIFEST_DIR, BASE_DIR / "logs"]:
        d.mkdir(parents=True, exist_ok=True)


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:120] or "document"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def guess_extension(url, content_type=""):
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return ".pdf"
    if "pdf" in content_type.lower():
        return ".pdf"
    if "html" in content_type.lower():
        return ".html"
    return ".html"


def save_sources_csv():
    with open(SOURCES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "source_org", "url", "language", "type", "kb", "priority"],
        )
        writer.writeheader()
        for row in SOURCES:
            writer.writerow(row)


def download_sources():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "ethiopia-agronomy-rag-builder/1.0"
    })

    downloaded = []

    for idx, source in enumerate(SOURCES, start=1):
        title = source["title"]
        url = source["url"]
        print(f"\n[{idx}/{len(SOURCES)}] Downloading: {title}")

        try:
            response = session.get(url, timeout=90, allow_redirects=True, verify=False)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            ext = guess_extension(url, content_type)

            filename = f"{idx:03d}_{slugify(title)}{ext}"
            path = RAW_DIR / filename

            if path.exists() and path.stat().st_size > 0:
                print(f"  already exists: {path}")
            else:
                path.write_bytes(response.content)
                print(f"  saved: {path}")

            log_record = {
                **source,
                "local_path": str(path),
                "downloaded_at": datetime.utcnow().isoformat(),
                "content_type": content_type,
                "status": "ok",
                "sha256_raw": sha256_bytes(path.read_bytes()),
            }

            downloaded.append(log_record)

        except Exception as e:
            print(f"  FAILED: {e}")
            log_record = {
                **source,
                "local_path": None,
                "downloaded_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e),
            }

        with open(DOWNLOAD_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return downloaded


def extract_html(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n")
    return clean_text(text)


def extract_pdf(path):
    try:
        text = pdf_extract_text(str(path)) or ""
        return clean_text(text)
    except Exception as e:
        print(f"  PDF extraction failed for {path}: {e}")
        return ""


def extract_all(downloaded):
    extracted = []

    for item in downloaded:
        if item.get("status") != "ok":
            continue

        path = pathlib.Path(item["local_path"])
        print(f"\nExtracting: {path.name}")

        if path.suffix.lower() == ".pdf":
            text = extract_pdf(path)
        else:
            text = extract_html(path)

        text_path = TEXT_DIR / f"{path.stem}.txt"
        meta_path = TEXT_DIR / f"{path.stem}.meta.json"

        text_path.write_text(text, encoding="utf-8")

        meta = {
            **item,
            "text_path": str(text_path),
            "char_count": len(text),
            "sha256_text": sha256_text(text),
            "needs_review": len(text) < 500,
        }

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  text chars: {len(text)}")
        if len(text) < 500:
            print("  WARNING: extracted text is short. This PDF may need OCR.")

        extracted.append({
            "meta": meta,
            "text": text,
        })

    return extracted


def normalize_amharic_common(text):
    """
    Conservative Amharic normalization.
    This helps remove duplicate-looking text without changing too much meaning.
    """
    replacements = {
        "ሃ": "ሀ",
        "ኅ": "ሀ",
        "ኃ": "ሀ",
        "ሐ": "ሀ",
        "ሓ": "ሀ",
        "ኻ": "ሀ",
        "ሗ": "ኋ",
        "ሠ": "ሰ",
        "ሡ": "ሱ",
        "ሢ": "ሲ",
        "ሣ": "ሳ",
        "ሤ": "ሴ",
        "ሥ": "ስ",
        "ሦ": "ሶ",
        "ሧ": "ሷ",
        "ዐ": "አ",
        "ዑ": "ኡ",
        "ዒ": "ኢ",
        "ዓ": "አ",
        "ዔ": "ኤ",
        "ዕ": "እ",
        "ዖ": "ኦ",
        "ጸ": "ፀ",
        "ጹ": "ፁ",
        "ጺ": "ፂ",
        "ጻ": "ፃ",
        "ጼ": "ፄ",
        "ጽ": "ፅ",
        "ጾ": "ፆ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # remove repeated page markers and isolated page numbers
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        if re.search(r"^\s*page\s+\d+\s*$", line, flags=re.I):
            continue

        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = normalize_amharic_common(text)

    return text.strip()


def paragraph_key(paragraph):
    p = paragraph.lower()
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"[^\w\u1200-\u137F ]+", "", p)
    return p.strip()


def is_near_duplicate(a, b, threshold=0.92):
    if not a or not b:
        return False

    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.35:
        return False

    return SequenceMatcher(None, a, b).ratio() >= threshold


def dedupe_paragraphs(documents):
    """
    Removes exact and near-duplicate paragraphs across all documents.
    Keeps source metadata.
    """
    seen_hashes = set()
    kept_norms = []
    cleaned_docs = []

    total_before = 0
    total_after = 0

    for doc in documents:
        meta = doc["meta"]
        text = doc["text"]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        cleaned_paragraphs = []

        for p in paragraphs:
            total_before += 1

            if len(p) < 80:
                continue

            key = paragraph_key(p)
            if not key:
                continue

            h = sha256_text(key)

            if h in seen_hashes:
                continue

            duplicate = False
            for old_key in kept_norms[-2000:]:
                if is_near_duplicate(key, old_key):
                    duplicate = True
                    break

            if duplicate:
                continue

            seen_hashes.add(h)
            kept_norms.append(key)
            cleaned_paragraphs.append(p)
            total_after += 1

        cleaned_docs.append({
            "meta": meta,
            "paragraphs": cleaned_paragraphs,
        })

    report = {
        "paragraphs_before": total_before,
        "paragraphs_after": total_after,
        "removed": total_before - total_after,
        "generated_at": datetime.utcnow().isoformat(),
    }

    (CLEAN_DIR / "dedupe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDeduplication report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return cleaned_docs


def build_master_documents(cleaned_docs):
    md_parts = []
    txt_parts = []

    md_parts.append("# Ethiopia Agronomy Knowledge Base\n")
    md_parts.append(f"Generated at: {datetime.utcnow().isoformat()} UTC\n")
    md_parts.append("This document is automatically built from curated Ethiopian agronomy sources.\n")

    txt_parts.append("ETHIOPIA AGRONOMY KNOWLEDGE BASE")
    txt_parts.append(f"Generated at: {datetime.utcnow().isoformat()} UTC")
    txt_parts.append("")

    for idx, doc in enumerate(cleaned_docs, start=1):
        meta = doc["meta"]
        paragraphs = doc["paragraphs"]

        title = meta["title"]
        source_org = meta["source_org"]
        url = meta["url"]
        language = meta["language"]
        doc_type = meta["type"]

        md_parts.append("\n---\n")
        md_parts.append(f"\n## {idx}. {title}\n")
        md_parts.append(f"- Source organization: {source_org}\n")
        md_parts.append(f"- URL: {url}\n")
        md_parts.append(f"- Language: {language}\n")
        md_parts.append(f"- Type: {doc_type}\n")
        md_parts.append(f"- Local file: {meta.get('local_path')}\n")
        md_parts.append(f"- Extracted characters: {meta.get('char_count')}\n\n")

        txt_parts.append("\n" + "=" * 80)
        txt_parts.append(f"{idx}. {title}")
        txt_parts.append(f"Source organization: {source_org}")
        txt_parts.append(f"URL: {url}")
        txt_parts.append(f"Language: {language}")
        txt_parts.append(f"Type: {doc_type}")
        txt_parts.append("=" * 80)
        txt_parts.append("")

        for p in paragraphs:
            md_parts.append(p + "\n\n")
            txt_parts.append(p)
            txt_parts.append("")

    MASTER_MD.write_text("".join(md_parts), encoding="utf-8")
    MASTER_TXT.write_text("\n".join(txt_parts), encoding="utf-8")

    print(f"\nMaster Markdown saved to: {MASTER_MD}")
    print(f"Master TXT saved to: {MASTER_TXT}")


def split_sentences(text):
    parts = re.split(r"(?<=[.!?።፧፨])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def infer_language(text):
    am = sum(1 for c in text if "\u1200" <= c <= "\u137F")
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z"))

    if am > latin * 1.5:
        return "am"
    if latin > am * 1.5:
        return "en"
    return "mixed"


def chunk_text(text, chunk_size=1800, overlap=250):
    sentences = split_sentences(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)

            tail = current[-overlap:] if current else ""
            current = (tail + " " + s).strip()

    if current:
        chunks.append(current)

    return chunks


def build_chunks(cleaned_docs):
    records = []

    for doc_index, doc in enumerate(cleaned_docs, start=1):
        meta = doc["meta"]
        full_text = "\n\n".join(doc["paragraphs"])
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            record = {
                "id": f"agronomy_{doc_index:03d}_{i:04d}",
                "kb": "agronomy",
                "title": meta["title"],
                "source_org": meta["source_org"],
                "source_url": meta["url"],
                "language": meta["language"],
                "language_segment": infer_language(chunk),
                "doc_type": meta["type"],
                "chunk_index": i,
                "text": chunk,
                "metadata": {
                    "local_path": meta.get("local_path"),
                    "text_path": meta.get("text_path"),
                    "priority": meta.get("priority"),
                    "sha256_text": sha256_text(chunk),
                    "created_at": datetime.utcnow().isoformat(),
                },
            }

            records.append(record)

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nRAG chunks saved to: {CHUNKS_JSONL}")
    print(f"Total chunks: {len(records)}")


def write_readme():
    readme = CLEAN_DIR / "README.md"
    readme.write_text(
        f"""# Ethiopia Agronomy RAG Corpus

Generated at: {datetime.utcnow().isoformat()} UTC

## Output files

- `data/clean/master_agronomy_kb.md`
- `data/clean/master_agronomy_kb.txt`
- `data/chunks/agronomy_chunks.jsonl`
- `data/clean/dedupe_report.json`
- `data/manifests/agronomy_sources.csv`

## Notes

Some PDFs may need OCR if extracted text is too short. Check `.meta.json` files under:

`data/text/agronomy/`

If `needs_review` is true, run OCR manually or inspect the PDF.
""",
        encoding="utf-8",
    )


def main():
    print("Building Ethiopia Agronomy Knowledge Base")
    ensure_dirs()
    save_sources_csv()

    downloaded = download_sources()
    extracted = extract_all(downloaded)
    cleaned = dedupe_paragraphs(extracted)

    build_master_documents(cleaned)
    build_chunks(cleaned)
    write_readme()

    print("\nDONE.")
    print(f"Master file: {MASTER_MD}")
    print(f"RAG chunks: {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()