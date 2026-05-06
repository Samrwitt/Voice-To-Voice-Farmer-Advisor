#!/usr/bin/env python3
"""
Ingest KB files (PDF, Word, plain text) into Postgres + pgvector.

Supported extensions: .pdf, .docx, .txt, .md

Usage (from repo root, with postgres up):
  docker compose run --rm logic_service python scripts/ingest_kb_folder.py

Or locally with POSTGRES_URL set:
  set POSTGRES_URL=postgresql://kb:kb@localhost:5432/advisor_kb
  python scripts/ingest_kb_folder.py --folder ./kb_documents/amharic

Default folder in container: /app/kb_documents/amharic

Note: Scanned PDFs (images only) need OCR first; pdfminer only reads text layers.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# logic_service code root (Docker: /app)
LOGIC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(LOGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOGIC_ROOT))


KB_FILE_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_file(path: Path) -> str:
    """Load text from PDF, Word, or UTF-8 text/markdown."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return clean_extracted_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        from pdfminer.high_level import extract_text as pdf_extract_text

        raw = pdf_extract_text(str(path)) or ""
        return clean_extracted_text(raw)
    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return clean_extracted_text("\n".join(parts))
    raise ValueError(f"Unsupported file type: {suffix}")


def iter_kb_files(folder: Path) -> list[Path]:
    """All supported KB files under folder (no README.md, no dotfiles)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for ext in KB_FILE_EXTENSIONS:
        for p in folder.glob(f"*{ext}"):
            if p.name.upper() == "README.MD" or p.name.startswith("."):
                continue
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def default_ingest_folder() -> str:
    env = os.environ.get("KB_INGEST_DIR")
    if env:
        return env
    docker_path = Path("/app/kb_documents/amharic")
    if docker_path.is_dir():
        return str(docker_path)
    return str(REPO_ROOT / "kb_documents" / "amharic")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest KB files (.pdf, .docx, .txt, .md) into Postgres+pgvector"
    )
    parser.add_argument(
        "--folder",
        default=default_ingest_folder(),
        help="Directory containing KB files",
    )
    parser.add_argument("--source-org", default=os.environ.get("KB_SOURCE_ORG", "አካባቢ ሰነድ"))
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all kb_documents/kb_chunks before ingest (dangerous)",
    )
    args = parser.parse_args()

    import psycopg

    from rag_pg import (
        POSTGRES_URL,
        chunk_amharic_text,
        embed_texts,
        init_pg_schema,
        kb_pg_enabled,
    )

    if not kb_pg_enabled():
        print("POSTGRES_URL is not set or psycopg is missing. Cannot ingest.", file=sys.stderr)
        sys.exit(1)

    init_pg_schema()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    files = iter_kb_files(folder)
    if not files:
        print(
            f"No supported files in {folder}. "
            f"Expected one of: {', '.join(KB_FILE_EXTENSIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    with psycopg.connect(POSTGRES_URL, autocommit=False) as conn:
        if args.wipe:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE kb_chunks, kb_documents CASCADE;")
            conn.commit()
            print("Wiped kb_chunks and kb_documents.")

        for path in files:
            try:
                text = extract_text_from_file(path)
            except Exception as exc:
                print(f"Skip (read error) {path.name}: {exc}", file=sys.stderr)
                continue
            if len(text) < 50:
                print(
                    f"Skip (too little text — scanned PDF?): {path.name}",
                    file=sys.stderr,
                )
                continue

            title = path.stem.replace("_", " ").replace("-", " ")
            chunks = chunk_amharic_text(text)
            if not chunks:
                print(f"Skip (no chunks): {path.name}")
                continue

            embeddings = embed_texts(chunks)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kb_documents
                        (title, source_org, source_url, language, status, original_filename, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id;
                    """,
                    (
                        title,
                        args.source_org,
                        None,
                        "am",
                        "approved",
                        path.name,
                        "{}",
                    ),
                )
                doc_id = cur.fetchone()[0]

                for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
                    lit = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                    cur.execute(
                        """
                        INSERT INTO kb_chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s::vector);
                        """,
                        (doc_id, idx, content, lit),
                    )

            conn.commit()
            print(f"Ingested: {path.name} -> {len(chunks)} chunks (document_id={doc_id})")

    print("Done.")


if __name__ == "__main__":
    main()
