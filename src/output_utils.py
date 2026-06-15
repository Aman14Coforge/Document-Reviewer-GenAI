"""Helpers for versioned output file paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_output_path(
    directory: Path,
    stem: str,
    label: str,
    timestamp: str | None = None,
) -> Path:
    ts = timestamp or run_timestamp()
    return directory / f"{stem}_{ts}_{label}.json"
