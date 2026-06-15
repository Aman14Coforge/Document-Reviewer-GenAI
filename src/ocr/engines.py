"""Dispatch OCR requests to the configured engine."""

from __future__ import annotations

from .cloud_ocr import cloud_ocr_extract
from .easyocr_engine import easyocr_extract


def run_ocr_on_image(image, config: dict) -> str:
    engine = config.get("engine", "easyocr").lower()

    if engine == "easyocr":
        return easyocr_extract(
            image,
            config.get("languages", ["en"]),
            gpu=config.get("gpu", False),
        )

    if engine == "cloud":
        return cloud_ocr_extract(image, config)

    raise ValueError(f"Unknown OCR engine: {engine}. Use 'easyocr' or 'cloud'.")
