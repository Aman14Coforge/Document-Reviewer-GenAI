"""Extract text from PDF and DOCX files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.output_utils import build_output_path, run_timestamp
sys.path.insert(0, str(PROJECT_ROOT))


# def extract_pdf(path: Path) -> dict:
#     import pdfplumber

#     pages = []
#     with pdfplumber.open(path) as pdf:
#         for index, page in enumerate(pdf.pages, start=1):
#             text = (
#                 page.extract_text(x_tolerance=10, y_tolerance=5) or ""
#             ).strip()
#             if not text:
#                 text = _extract_tables_as_text(page)

#             pages.append(
#                 {
#                     "page": index,
#                     "text": text,
#                 }
#             )

#     full_text = "\n\n".join(
#         f"--- Page {page['page']} ---\n{page['text']}" for page in pages if page["text"]
#     )
#     return {
#         "page_count": len(pages),
#         "pages": pages,
#         "full_text": full_text.strip(),
#     }

def extract_pdf_with_fonts(path: Path) -> dict:
    import pdfplumber
    import fitz

    #  open fitz separately for fonts
    fitz_doc = fitz.open(path)

    pages = []
    all_fonts = set()  # document-level fonts

    #  fonts to ignore for GDP-08 (icons, symbols)
    IGNORE_FONTS = {"wingdings", "symbol"}

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):

            #  ORIGINAL TEXT EXTRACTION (UNCHANGED)
            text = (
                page.extract_text(x_tolerance=10, y_tolerance=5) or ""
            ).strip()

            if not text:
                text = _extract_tables_as_text(page)

            #  FONT EXTRACTION USING FITZ
            fonts = set()

            fitz_page = fitz_doc[index - 1]
            text_dict = fitz_page.get_text("dict")

            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):

                        font = span.get("font")

                        if font:
                            #  base normalization
                            font = font.split("+")[-1].lower()

                            #  remove style variations
                            for token in ["-bold", "-italic", "bold", "italic", "mt", "-regular"]:
                                font = font.replace(token, "")

                            font = font.strip()

                            # ignore non-content fonts
                            if font in IGNORE_FONTS:
                                continue

                            fonts.add(font)
                            all_fonts.add(font)

            #  STORE PAGE
            pages.append(
                {
                    "page": index,
                    "text": text,
                    "fonts": list(fonts),
                }
            )

    #  ORIGINAL FULL TEXT (UNCHANGED)
    full_text = "\n\n".join(
        f"--- Page {page['page']} ---\n{page['text']}"
        for page in pages if page["text"]
    )

    return {
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text.strip(),
        "document_fonts": list(all_fonts),
    }

def _extract_tables_as_text(page) -> str:
    """Fallback when page text is empty but tables exist (e.g. scanned forms)."""
    blocks: list[str] = []
    for table in page.extract_tables() or []:
        rows = [
            " | ".join(str(cell or "").strip() for cell in row)
            for row in table
            if any(cell for cell in row)
        ]
        if rows:
            blocks.append("\n".join(rows))
    return "\n\n".join(blocks).strip()


def extract_docx(path: Path) -> dict:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    document = Document(path)
    pages: list[dict] = []
    current_page = 1
    current_lines: list[str] = []

    def flush_page() -> None:
        nonlocal current_page, current_lines
        text = "\n".join(current_lines).strip()
        pages.append({"page": current_page, "text": text})
        current_page += 1
        current_lines = []

    for block in document.element.body:
        if block.tag == qn("w:p"):
            paragraph = Paragraph(block, document)
            for run in paragraph.runs:
                for child in run._element:
                    if child.tag == qn("w:br") and child.get(qn("w:type")) == "page":
                        if paragraph.text.strip():
                            current_lines.append(paragraph.text.strip())
                        flush_page()
                        break
            else:
                if paragraph.text.strip():
                    current_lines.append(paragraph.text.strip())
        elif block.tag == qn("w:tbl"):
            rows = []
            for row in block.findall(".//" + qn("w:tr")):
                cells = []
                for cell in row.findall(".//" + qn("w:t")):
                    if cell.text:
                        cells.append(cell.text.strip())
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                current_lines.append("\n".join(rows))

    if current_lines or not pages:
        flush_page()

    if len(pages) == 1 and not pages[0]["text"]:
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        pages = [{"page": 1, "text": "\n".join(paragraphs)}]

    full_text = "\n\n".join(
        f"--- Page {page['page']} ---\n{page['text']}" for page in pages if page["text"]
    )
    return {
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text.strip(),
    }


def extract_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "page_count": 1,
        "pages": [{"page": 1, "text": text}],
        "full_text": text,
    }


def extract_document(
    path: Path,
    mode: str = "native",
    ocr_config_path: Path | None = None,
) -> dict:
    suffix = path.suffix.lower()
    mode = mode.lower()

    if mode not in {"native", "ocr"}:
        raise ValueError(f"Unsupported extraction mode: {mode}. Use 'native' or 'ocr'.")

    if mode == "ocr":
        if suffix != ".pdf":
            raise ValueError("OCR mode currently supports PDF files only.")
        from src.ocr.extract_ocr import extract_pdf_ocr

        extracted = extract_pdf_ocr(path, ocr_config_path)
    elif suffix == ".pdf":
        extracted = extract_pdf_with_fonts(path)
    elif suffix == ".docx":
        extracted = extract_docx(path)
    elif suffix == ".txt":
        extracted = extract_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")

    if "extraction_method" not in extracted:
        extracted["extraction_method"] = "native"
    if "ocr_engine" not in extracted:
        extracted["ocr_engine"] = None

    return {
        "source_path": str(path.resolve()),
        "file_name": path.name,
        "file_stem": path.stem,
        "file_type": suffix.lstrip("."),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        **extracted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF, DOCX, or TXT files.")
    parser.add_argument("document_path", help="Path to a .pdf, .docx, or .txt file")
    parser.add_argument(
        "--mode",
        choices=["native", "ocr"],
        default="native",
        help="native: pdfplumber/docx/txt | ocr: PDF pages to images then EasyOCR",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: output/extracted/<file_stem>_<timestamp>_extracted.json)",
    )
    args = parser.parse_args()

    source = Path(args.document_path)
    if not source.exists():
        raise SystemExit(f"File not found: {source}")

    run_ts = run_timestamp()
    result = extract_document(source, mode=args.mode)
    result["run_timestamp"] = run_ts

    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(
            PROJECT_ROOT / "output" / "extracted", source.stem, "extracted", run_ts
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Mode:     {result['extraction_method']}")
    print(f"Extracted {result['page_count']} page(s) from {source.name}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
