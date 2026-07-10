"""Split extracted document text into section-wise chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.output_utils import build_output_path, run_timestamp

REVISION_PATTERNS = re.compile(
    r"(revision history|document history|change history|version history|revision record)",
    re.IGNORECASE,
)
APPROVAL_PATTERNS = re.compile(
    r"(document review and approval|signature verification|approved by|authorised by|authorized by)",
    re.IGNORECASE,
)
TOC_PATTERNS = re.compile(r"table of contents", re.IGNORECASE)
FOOTER_PATTERNS = re.compile(
    r"(confidential|page\s+\d+\s+of\s+\d+)",
    re.IGNORECASE,
)
STRUCTURE_HEADING = re.compile(
    r"\b(introduction|scope|responsibilities|references|procedure|objective)\b",
    re.IGNORECASE,
)
TOP_LEVEL_SECTION = re.compile(
    r"^\d+\s+[A-Z][A-Z0-9\s/&,\-]{2,}$",
    re.IGNORECASE,
)
MAJOR_SECTION = re.compile(
    r"^(REVISION HISTORY|DOCUMENT REVIEW AND APPROVAL|TABLE OF CONTENTS|"
    r"SIGNATURE VERIFICATION LOG|IQ PROTOCOL|PQ PROTOCOL|REFERENCES|RESPONSIBILITIES)\b",
    re.IGNORECASE,
)

TABLE_HEADER_LINES = {
    "name",
    "date",
    "signature",
    "approver(s)",
    "title / department",
    "revision",
    "no.",
    "description of change",
    "name / title / department",
    "title/company",
    "initials",
    "term",
    "expansion",
    "objective:",
    "acceptance criteria:",
    "results:",
}


def load_extracted(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_page_lines(page_text: str, page_number: int) -> list[str]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if lines and lines[0] == str(page_number):
        lines = lines[1:]
    return merge_split_headings(lines)


def merge_split_headings(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if re.fullmatch(r"\d+(?:\.\d+)*", line) and re.match(r"[A-Za-z<]", next_line):
                merged.append(f"{line} {next_line}")
                index += 2
                continue
        merged.append(line)
        index += 1
    return merged


def is_section_boundary(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    if normalized.lower() in TABLE_HEADER_LINES:
        return False
    if len(normalized) <= 3 and normalized.isupper():
        return False
    if re.search(r"\.{5,}", normalized):
        return False
    if MAJOR_SECTION.match(normalized):
        return True
    if TOP_LEVEL_SECTION.match(normalized):
        return True
    return False


def detect_section_type(text: str, page_number: int, heading: str) -> str:
    probe = f"{heading}\n{text}"
    if page_number == 1:
        return "first_page"
    if REVISION_PATTERNS.search(probe):
        return "revision_history"
    if APPROVAL_PATTERNS.search(probe):
        return "approval"
    if TOC_PATTERNS.search(probe):
        return "structure"
    if TOP_LEVEL_SECTION.match(heading) or MAJOR_SECTION.match(heading):
        return "structure"
    if STRUCTURE_HEADING.search(heading):
        return "structure"
    if FOOTER_PATTERNS.search(probe):
        return "footer"
    return "body"


def split_page_lines(lines: list[str], page_number: int) -> list[dict]:
    if not lines:
        return []

    if page_number == 1:
        text = "\n".join(lines).strip()
        title_line = next(
            (line for line in lines if line.isupper() and len(line) > 3 and line != "FOR"),
            f"Page {page_number}",
        )
        return [
            {
                "chunk_id": f"p{page_number}_1",
                "section_type": "first_page",
                "heading": title_line,
                "page_start": page_number,
                "page_end": page_number,
                "text": text,
            }
        ]

    page_text = "\n".join(lines)
    if TOC_PATTERNS.search(page_text):
        return [
            {
                "chunk_id": f"p{page_number}_1",
                "section_type": "structure",
                "heading": "TABLE OF CONTENTS",
                "page_start": page_number,
                "page_end": page_number,
                "text": page_text.strip(),
            }
        ]

    chunks: list[dict] = []
    current_heading = f"Page {page_number}"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_lines
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        chunks.append(
            {
                "chunk_id": f"p{page_number}_{len(chunks) + 1}",
                "section_type": detect_section_type(text, page_number, current_heading),
                "heading": current_heading,
                "page_start": page_number,
                "page_end": page_number,
                "text": text,
            }
        )
        current_lines = []

    for line in lines:
        if is_section_boundary(line):
            flush()
            current_heading = line
            current_lines = [line]
        else:
            if not current_lines:
                current_heading = f"Page {page_number}"
            current_lines.append(line)

    flush()
    return chunks


def chunk_pages_only(pages: list[dict], page_count: int) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        page_number = page.get("page", 1)
        text = page.get("text", "").strip()
        if not text:
            continue
        heading = f"Page {page_number}"
        if page_number == 1:
            section_type = "first_page"
            first_line = text.splitlines()[0] if text.splitlines() else heading
            heading = first_line[:80]
        else:
            section_type = detect_section_type(text, page_number, heading)
        chunks.append(
            {
                "chunk_id": f"p{page_number}_1",
                "section_type": section_type,
                "heading": heading,
                "page_start": page_number,
                "page_end": page_number,
                "text": text,
            }
        )
    return chunks


def preprocess_pages_for_chunking(extracted: dict) -> list[dict]:
    is_ocr = extracted.get("extraction_method") == "ocr"
    pages = extracted.get("pages", [])
    if not is_ocr:
        return pages

    from src.ocr.config import load_ocr_config
    from src.ocr.text_cleanup import normalize_ocr_page_text

    load_ocr_config()
    processed = []
    for page in pages:
        text = normalize_ocr_page_text(page.get("text", ""))
        processed.append({**page, "text": text})
    return processed


def chunk_document(extracted: dict) -> dict:
    is_ocr = extracted.get("extraction_method") == "ocr"
    pages = preprocess_pages_for_chunking(extracted)
    chunks: list[dict] = []

    use_page_chunks = False
    if is_ocr:
        from src.ocr.config import load_ocr_config

        use_page_chunks = load_ocr_config().get("fallback_to_page_chunks", True)

    if use_page_chunks:
        chunks.extend(chunk_pages_only(pages, extracted.get("page_count", len(pages))))
    else:
        for page in pages:
            page_number = page.get("page", 1)
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            lines = normalize_page_lines(page_text, page_number)
            chunks.extend(split_page_lines(lines, page_number))

    full_text = extracted.get("full_text", "")
    if is_ocr:
        from src.ocr.text_cleanup import normalize_ocr_page_text

        full_text = normalize_ocr_page_text(full_text)

    if not chunks and full_text:
        chunks.append(
            {
                "chunk_id": "full_1",
                "section_type": "full",
                "heading": "Full Document",
                "page_start": 1,
                "page_end": extracted.get("page_count", 1),
                "text": full_text,
            }
        )

    chunks.append(
        {
            "chunk_id": "full_document",
            "section_type": "full",
            "heading": "Complete Document",
            "page_start": 1,
            "page_end": extracted.get("page_count", 1),
            "text": full_text,
        }
    )

    return {
        "source_path": extracted.get("source_path"),
        "file_name": extracted.get("file_name"),
        "file_stem": extracted.get("file_stem"),
        "page_count": extracted.get("page_count", 0),
        "extraction_method": extracted.get("extraction_method", "native"),
        "ocr_engine": extracted.get("ocr_engine"),
        "chunk_strategy": "page" if use_page_chunks else "section",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create section-wise chunks from extracted JSON.")
    parser.add_argument("extracted_json", help="Path to extracted document JSON")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: output/chunks/<file_stem>_<timestamp>_chunks.json)",
    )
    args = parser.parse_args()

    extracted_path = Path(args.extracted_json)
    if not extracted_path.exists():
        raise SystemExit(f"File not found: {extracted_path}")

    extracted = load_extracted(extracted_path)
    run_ts = run_timestamp()
    result = chunk_document(extracted)
    result["run_timestamp"] = run_ts

    stem = extracted.get("file_stem") or extracted_path.stem
    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(PROJECT_ROOT / "output" / "chunks", stem, "chunks", run_ts)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created {result['chunk_count']} chunk(s) for {result.get('file_name', stem)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
