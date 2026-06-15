"""Run the document pipeline: validate, extract text, then chunk by section."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from chunk_document import chunk_document
from extract_document import extract_document
from src.output_utils import build_output_path, run_timestamp
from validate_document import validate_document

# Set the document path here (relative to project root or absolute path)
DOCUMENT_PATH = PROJECT_ROOT / "docs" / "Deployment Report_v0 - filled.pdf"
RULES_PATH = None

# Extraction mode: "native" (pdfplumber/docx/txt) or "ocr" (PDF -> images -> EasyOCR)
EXTRACTION_MODE = "ocr"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(document_path: Path, extraction_mode: str = EXTRACTION_MODE) -> dict:
    document_path = document_path.resolve()
    extraction_mode = extraction_mode.lower()

    stem = document_path.stem
    run_ts = run_timestamp()
    validation_output = build_output_path(
        PROJECT_ROOT / "output" / "validation", stem, "validation", run_ts
    )
    extracted_output = build_output_path(
        PROJECT_ROOT / "output" / "extracted", stem, "extracted", run_ts
    )
    chunks_output = build_output_path(
        PROJECT_ROOT / "output" / "chunks", stem, "chunks", run_ts
    )

    print("=" * 60)
    print("Document Reviewer Pipeline")
    print("=" * 60)
    print(f"Input:     {document_path}")
    print(f"Mode:      {extraction_mode}")
    print(f"Run time:  {run_ts}")
    print()

    # Step 1 — Validate form type and file rules
    print("[Step 1/3] Validating document...")
    validation = validate_document(
        document_path,
        rules_path=RULES_PATH,
        extraction_mode=extraction_mode,
    )
    validation["run_timestamp"] = run_ts
    save_json(validation, validation_output)
    print(f"  Form type: {validation['form_type']}")
    print(f"  Size:      {validation['file_size_mb']} MB")
    print(f"  Valid:     {validation['valid']}")
    print(f"  Saved:     {validation_output}")

    if not validation["valid"]:
        print()
        for error in validation.get("errors", []):
            print(f"  {error}")
        print()
        raise ValueError("Document validation failed. Fix the issues above and retry.")
    print()

    suffix = document_path.suffix.lower()
    if extraction_mode == "native" and suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for extraction: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Step 2 — Extract text (native or OCR)
    print(f"[Step 2/3] Extracting text ({extraction_mode})...")
    extracted = extract_document(document_path, mode=extraction_mode)
    extracted["run_timestamp"] = run_ts
    save_json(extracted, extracted_output)
    print(f"  Pages:  {extracted['page_count']}")
    if extracted.get("ocr_engine"):
        print(f"  OCR:    {extracted['ocr_engine']}")
    print(f"  Saved:  {extracted_output}")

    if not extracted.get("full_text", "").strip():
        raise ValueError(
            "Text extraction returned no content. "
            "For scanned/image PDFs, set EXTRACTION_MODE = 'ocr' in main.py."
        )
    print()

    # Step 3 — Chunk extracted text by section
    print("[Step 3/3] Creating section chunks...")
    chunks = chunk_document(extracted)
    chunks["run_timestamp"] = run_ts
    save_json(chunks, chunks_output)
    print(f"  Strategy: {chunks.get('chunk_strategy', 'section')}")
    print(f"  Chunks:   {chunks['chunk_count']}")
    print(f"  Saved:    {chunks_output}")
    print()

    print("Pipeline complete.")
    print("=" * 60)

    return {
        "document_path": str(document_path),
        "extraction_mode": extraction_mode,
        "run_timestamp": run_ts,
        "validation_json": str(validation_output),
        "extracted_json": str(extracted_output),
        "chunks_json": str(chunks_output),
        "page_count": extracted["page_count"],
        "chunk_count": chunks["chunk_count"],
    }


if __name__ == "__main__":
    try:
        result = run_pipeline(DOCUMENT_PATH, extraction_mode=EXTRACTION_MODE)
        print(f"Validation JSON: {result['validation_json']}")
        print(f"Extracted JSON:  {result['extracted_json']}")
        print(f"Chunks JSON:     {result['chunks_json']}")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
