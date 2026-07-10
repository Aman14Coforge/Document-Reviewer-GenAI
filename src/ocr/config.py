"""Load OCR configuration."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCR_CONFIG_PATH = PROJECT_ROOT / "config" / "ocr.json"


def load_ocr_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_OCR_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"OCR config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
