"""EasyOCR engine wrapper."""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

_reader: Any = None
_reader_languages: tuple[str, ...] | None = None


def get_easyocr_reader(languages: list[str], *, gpu: bool = False):
    global _reader, _reader_languages

    language_key = tuple(languages)
    if _reader is None or _reader_languages != language_key:
        import easyocr

        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        buffer = StringIO()
        with redirect_stdout(buffer):
            _reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        _reader_languages = language_key
    return _reader


def easyocr_extract(image, languages: list[str], *, gpu: bool = False) -> str:
    reader = get_easyocr_reader(languages, gpu=gpu)
    lines = reader.readtext(image, detail=0, paragraph=True)
    if isinstance(lines, list):
        return "\n".join(line.strip() for line in lines if str(line).strip()).strip()
    return str(lines).strip()
