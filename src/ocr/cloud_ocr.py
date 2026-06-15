"""Placeholder for future cloud OCR providers."""

from __future__ import annotations


def cloud_ocr_extract(image, config: dict) -> str:
    """
    Dummy stub for future cloud OCR integration.

    Planned providers: Azure Document Intelligence, Google Vision, AWS Textract, etc.
    """
    provider = config.get("cloud_provider", "unspecified")
    raise NotImplementedError(
        f"Cloud OCR is not configured yet (provider={provider}). "
        "Set engine to 'easyocr' in config/ocr.json for now."
    )
