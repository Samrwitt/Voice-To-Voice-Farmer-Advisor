import json
import re
import hashlib
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/chunks/agronomy_chunks.jsonl")
OUTPUT_FILE = Path("data/chunks/agronomy_chunks_clean.jsonl")
REJECTED_FILE = Path("data/chunks/agronomy_chunks_rejected.jsonl")
REPORT_FILE = Path("data/chunks/agronomy_cleaning_report.json")

MIN_CHARS = 300
MAX_CHARS = 3000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_amharic_common(text: str) -> str:
    replacements = {
        "ሃ": "ሀ", "ኅ": "ሀ", "ኃ": "ሀ", "ሐ": "ሀ", "ሓ": "ሀ",
        "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
        "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
        "ጸ": "ፀ", "ጹ": "ፁ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ጽ": "ፅ", "ጾ": "ፆ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_text(text: str) -> str:
    text = text or ""

    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # Fix broken hyphenated line endings
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated dotted leaders
    text = re.sub(r"\.{5,}", " ", text)

    text = normalize_amharic_common(text)

    return text.strip()


def infer_language(text: str) -> str:
    amharic_chars = sum(1 for c in text if "\u1200" <= c <= "\u137F")
    latin_chars = sum(1 for c in text if "a" <= c.lower() <= "z")

    if amharic_chars > latin_chars * 1.5:
        return "am"
    if latin_chars > amharic_chars * 1.5:
        return "en"
    return "mixed"


def looks_like_table_of_contents(text: str) -> bool:
    lower = text.lower()

    dotted_page_refs = len(re.findall(r"\.{3,}\s*\d+", text))
    numbered_toc_lines = len(re.findall(r"\n\s*\d+(\.\d+)*\s+.+\s+\d+\s*$", text))

    toc_terms = [
        "table of contents",
        "contents",
        "list of figures",
        "list of tables",
        "abbreviations",
    ]

    if dotted_page_refs >= 4:
        return True

    if numbered_toc_lines >= 5:
        return True

    if any(term in lower for term in toc_terms) and dotted_page_refs >= 2:
        return True

    return False


def looks_like_reference_section(text: str) -> bool:
    lower = text.lower()

    # Keep useful sections, but remove pure bibliography chunks
    if lower.strip().startswith("references") or lower.strip().startswith("bibliography"):
        citation_like = len(re.findall(r"\(\d{4}\)|\b\d{4}\b|et al\.|http|www\.", lower))
        sentence_like = len(re.findall(r"[.!?።፧፨]", text))

        if citation_like >= 5 and sentence_like >= 5:
            return True

    return False


def has_too_much_noise(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return True

    # Too many page-layout artifacts
    if text.count("|") > 20:
        return True

    if text.count("_") > 20:
        return True

    # Too many single-character lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        short_lines = sum(1 for line in lines if len(line) <= 3)
        if short_lines / max(len(lines), 1) > 0.45:
            return True

    # Too little actual sentence structure
    sentence_marks = len(re.findall(r"[.!?።፧፨]", text))
    words = len(text.split())

    if words > 80 and sentence_marks == 0:
        return True

    return False


def fix_midword_start(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    # If the chunk starts with a broken lowercase fragment, remove up to first sentence boundary
    if text[0].islower() or text[0] in ",.;:)]}":
        match = re.search(r"(?<=[.!?።፧፨])\s+", text[:400])
        if match:
            return text[match.end():].strip()

    return text


def fix_midword_end(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    # If it ends with a very likely broken sentence, trim after last full sentence
    if text[-1] not in ".!?።፧፨":
        matches = list(re.finditer(r"[.!?።፧፨]", text))
        if matches:
            last = matches[-1].end()
            if len(text) - last < 300:
                return text[:last].strip()

    return text


def split_long_chunk(text: str, max_chars: int = MAX_CHARS):
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = (current + "\n\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?።፧፨])\s+", chunk)
            current = ""

            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = (current + " " + sentence).strip()
                else:
                    if current:
                        final_chunks.append(current)
                    current = sentence.strip()

            if current:
                final_chunks.append(current)

    return final_chunks


def normalized_duplicate_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u1200-\u137F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_relevance(record, text: str) -> str:
    source_org = str(record.get("source_org", "")).lower()
    source_url = str(record.get("source_url", "")).lower()
    title = str(record.get("title", "")).lower()

    if "moa.gov.et" in source_url or "ministry of agriculture" in source_org:
        return "ethiopia_direct"

    if "data.moa.gov.et" in source_url:
        return "ethiopia_direct"

    if "landpks" in title or "landpotential" in source_url:
        return "ethiopia_related"

    if "ethiopia" in text.lower() or "ethiopia" in title:
        return "ethiopia_related"

    return "general_methodological"


def clean_record(record):
    text = record.get("text", "")
    text = normalize_text(text)
    text = fix_midword_start(text)
    text = fix_midword_end(text)
    text = normalize_text(text)

    return text


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded chunks: {len(records)}")

    cleaned_records = []
    rejected_records = []
    seen_hashes = set()

    stats = Counter()

    for record in records:
        original_id = record.get("id", "")
        text = clean_record(record)

        reject_reason = None

        if looks_like_table_of_contents(text):
            reject_reason = "table_of_contents"
        elif looks_like_reference_section(text):
            reject_reason = "reference_section"
        elif has_too_much_noise(text):
            reject_reason = "too_short_or_noisy"

        if reject_reason:
            rejected = dict(record)
            rejected["reject_reason"] = reject_reason
            rejected["cleaned_preview"] = text[:500]
            rejected_records.append(rejected)
            stats[f"rejected_{reject_reason}"] += 1
            continue

        pieces = split_long_chunk(text)

        for i, piece in enumerate(pieces):
            piece = normalize_text(piece)

            if has_too_much_noise(piece):
                rejected = dict(record)
                rejected["reject_reason"] = "split_piece_too_short_or_noisy"
                rejected["cleaned_preview"] = piece[:500]
                rejected_records.append(rejected)
                stats["rejected_split_piece_too_short_or_noisy"] += 1
                continue

            duplicate_key = normalized_duplicate_key(piece)
            text_hash = sha256_text(duplicate_key)

            if text_hash in seen_hashes:
                stats["rejected_duplicate"] += 1
                rejected = dict(record)
                rejected["reject_reason"] = "duplicate"
                rejected["cleaned_preview"] = piece[:500]
                rejected_records.append(rejected)
                continue

            seen_hashes.add(text_hash)

            new_record = dict(record)

            if len(pieces) > 1:
                new_record["parent_id"] = original_id
                new_record["id"] = f"{original_id}_clean_{i:02d}"

            new_record["text"] = piece
            new_record["text_hash"] = text_hash
            new_record["cleaning_status"] = "cleaned"
            new_record["language_segment"] = infer_language(piece)
            new_record["ethiopia_relevance"] = classify_relevance(record, piece)
            new_record["char_count"] = len(piece)
            new_record["word_count"] = len(piece.split())

            cleaned_records.append(new_record)
            stats["kept"] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(REJECTED_FILE, "w", encoding="utf-8") as f:
        for record in rejected_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_counts = Counter(r.get("source_org", "unknown") for r in cleaned_records)
    title_counts = Counter(r.get("title", "unknown") for r in cleaned_records)
    language_counts = Counter(r.get("language_segment", "unknown") for r in cleaned_records)
    relevance_counts = Counter(r.get("ethiopia_relevance", "unknown") for r in cleaned_records)

    report = {
        "input_chunks": len(records),
        "cleaned_chunks": len(cleaned_records),
        "rejected_chunks": len(rejected_records),
        "stats": dict(stats),
        "language_counts": dict(language_counts),
        "ethiopia_relevance_counts": dict(relevance_counts),
        "source_counts": dict(source_counts),
        "top_documents": dict(title_counts.most_common(30)),
        "output_file": str(OUTPUT_FILE),
        "rejected_file": str(REJECTED_FILE),
    }

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nCleaning complete.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nClean file: {OUTPUT_FILE}")
    print(f"Rejected file: {REJECTED_FILE}")
    print(f"Report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()