"""Convert PDF pages to image arrays for OCR."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def pdf_pages_to_images(path: Path, dpi: int = 300) -> list[tuple[int, np.ndarray]]:
    import fitz

    document = fitz.open(path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[tuple[int, np.ndarray]] = []

    for index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix)
        channels = pixmap.n
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, channels
        )
        if channels == 4:
            image = image[:, :, :3]
        pages.append((index, image))

    document.close()
    return pages
