from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.utils.config import load_config, resolve_path

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pypdf is required for build_pdf_knowledge_base.py. "
        "Install it in the target environment first."
    ) from exc


TEXTBOOK_PAGE_RANGES = [
    (880, 915),
]
MIN_CHUNK_CHARS = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown knowledge base from PDF sepsis references")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--max-pages-per-pdf", type=int, default=0)
    parser.add_argument("--skip-large-textbook", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def normalize_stem(stem: str) -> str:
    stem = re.sub(r"\(1\)$", "", stem).strip()
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_").lower()


def canonical_sort_key(path: Path) -> tuple[str, int, str]:
    return (normalize_stem(path.stem), 1 if path.stem.endswith("(1)") else 0, path.name.lower())


def cleanup_text(text: str) -> str:
    replacements = {
        "\u00ad": "",
        "\u00a0": " ",
        "\uf0b7": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "聽": " ",
        "鈥": '"',
        "€™": "'",
        "€œ": '"',
        "€": '"',
        "Â©": " ",
        "©": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current and len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current)
        if chunk_overlap > 0 and chunks:
            tail = current[-chunk_overlap:].strip()
            current = f"{tail} {sentence}".strip() if tail else sentence
        else:
            current = sentence

    if current and len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)
    return chunks


def textbook_page_allowed(page_number_1based: int) -> bool:
    return any(start <= page_number_1based <= end for start, end in TEXTBOOK_PAGE_RANGES)


def extract_pdf_text(pdf_path: Path, max_pages: int, skip_large_textbook: bool) -> tuple[str, dict[str, int | list[list[int]]]]:
    reader = PdfReader(str(pdf_path))
    selected_pages = 0
    skipped_pages = 0
    page_texts: list[str] = []
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages > 0 else total_pages
    is_textbook = "textbook_of_critical_care" in pdf_path.name.lower()

    for page_idx in range(limit):
        page_number = page_idx + 1
        raw = reader.pages[page_idx].extract_text() or ""
        cleaned = cleanup_text(raw)
        if not cleaned:
            skipped_pages += 1
            continue
        if is_textbook and skip_large_textbook:
            skipped_pages += 1
            continue
        if is_textbook and not textbook_page_allowed(page_number):
            skipped_pages += 1
            continue
        page_texts.append(cleaned)
        selected_pages += 1

    return "\n\n".join(page_texts), {
        "total_pages": total_pages,
        "selected_pages": selected_pages,
        "skipped_pages": skipped_pages,
        "textbook_page_ranges": [[start, end] for start, end in TEXTBOOK_PAGE_RANGES] if is_textbook else [],
    }


def write_chunks(
    output_dir: Path,
    pdf_path: Path,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_name = normalize_stem(pdf_path.stem)
    clean_title = re.sub(r"\(1\)$", "", pdf_path.stem).strip()
    for old in output_dir.glob(f"{base_name}_chunk*.md"):
        old.unlink()

    for idx, chunk in enumerate(chunks, start=1):
        md_path = output_dir / f"{base_name}_chunk{idx:03d}.md"
        md_path.write_text(
            "\n".join(
                [
                    f"Title: {clean_title}",
                    f"Source: {pdf_path.name}",
                    f"Chunk: {idx}/{len(chunks)}",
                    "",
                    chunk,
                ]
            ),
            encoding="utf-8",
        )
    return len(chunks)


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.md"):
        path.unlink()
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    rag_cfg = cfg.get("rag", {})
    input_dir = resolve_path(PROJECT_DIR, args.input_dir or rag_cfg.get("literature_dir", "资料"))
    output_dir = resolve_path(PROJECT_DIR, args.output_dir or rag_cfg.get("global_docs_dir", "knowledge_base/global_docs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        clear_output_dir(output_dir)

    seen: set[str] = set()
    manifest_rows: list[dict[str, object]] = []
    total_chunks = 0
    processed = 0
    skipped_duplicates = 0

    for pdf_path in sorted(input_dir.glob("*.pdf"), key=canonical_sort_key):
        canonical = normalize_stem(pdf_path.stem)
        if canonical in seen:
            skipped_duplicates += 1
            continue
        seen.add(canonical)

        text, stats = extract_pdf_text(
            pdf_path,
            max_pages=args.max_pages_per_pdf,
            skip_large_textbook=args.skip_large_textbook,
        )
        if not text:
            print(f"skip_empty: {pdf_path.name}", flush=True)
            continue

        chunk_count = write_chunks(
            output_dir=output_dir,
            pdf_path=pdf_path,
            text=text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        processed += 1
        total_chunks += chunk_count
        manifest_rows.append(
            {
                "pdf_name": pdf_path.name,
                "canonical_name": canonical,
                "chunk_count": chunk_count,
                **stats,
            }
        )
        print(
            (
                f"built: {pdf_path.name} | total_pages={stats['total_pages']} "
                f"selected_pages={stats['selected_pages']} skipped_pages={stats['skipped_pages']} chunks={chunk_count}"
            ),
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "processed_pdfs": processed,
                "skipped_duplicates": skipped_duplicates,
                "total_chunks": total_chunks,
                "files": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"output_dir: {output_dir}", flush=True)
    print(f"processed_pdfs: {processed}", flush=True)
    print(f"skipped_duplicates: {skipped_duplicates}", flush=True)
    print(f"total_chunks: {total_chunks}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
