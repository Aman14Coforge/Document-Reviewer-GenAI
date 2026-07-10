"""Central logging setup for the document compliance pipeline.

Log output goes to the console and to ``output/logs/compliance.log`` by default.
Set ``LOG_LEVEL=DEBUG`` in the environment for verbose RAG / checker detail.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "output" / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    *,
    level: str | None = None,
    log_file: Path | bool | None = True,
) -> None:
    """Configure root logging once for CLI, Streamlit, and scripts."""
    global _configured
    if _configured:
        return

    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    if log_file is True:
        log_file = DEFAULT_LOG_DIR / "compliance.log"
    if isinstance(log_file, Path):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", "") == str(log_file.resolve())
            for handler in root.handlers
        ):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    _configured = True
    logging.getLogger("compliance").info(
        "Logging initialized (level=%s, file=%s)",
        log_level_name,
        log_file if isinstance(log_file, Path) else "disabled",
    )


def get_logger(module: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``compliance.pipeline``."""
    if not _configured:
        setup_logging()
    return logging.getLogger(f"compliance.{module}")
