"""Extract text from PDF pages using OCR."""

from __future__ import annotations

from pathlib import Path

from src.ocr.config import load_ocr_config
from src.ocr.engines import run_ocr_on_image
from src.ocr.pdf_to_images import pdf_pages_to_images


def extract_pdf_ocr(path: Path, ocr_config_path: Path | None = None) -> dict:
    config = load_ocr_config(ocr_config_path)
    dpi = config.get("dpi", 300)
    engine = config.get("engine", "easyocr")

    pages: list[dict] = []
    page_images = pdf_pages_to_images(path, dpi=dpi)

    for page_number, image in page_images:
        text = run_ocr_on_image(image, config)
        pages.append({"page": page_number, "text": text})

    full_text = "\n\n".join(
        f"--- Page {page['page']} ---\n{page['text']}" for page in pages if page["text"]
    )

    return {
        "page_count": len(pages),
        "pages": pages,
        "full_text": full_text.strip(),
        "extraction_method": "ocr",
        "ocr_engine": engine,
    }
