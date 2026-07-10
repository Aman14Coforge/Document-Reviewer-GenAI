"""Validate uploaded form/document before processing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.output_utils import build_output_path, run_timestamp

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "validation.json"

EXTENSION_TO_FORM_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
}


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Validation config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect_form_type(path: Path) -> str:
    return EXTENSION_TO_FORM_TYPE.get(path.suffix.lower(), "unknown")


def _message(config: dict, key: str) -> str:
    return config.get("messages", {}).get(key, key)


def _check(name: str, passed: bool, message: str, user_message: str | None = None) -> dict:
    return {
        "name": name,
        "passed": passed,
        "message": message,
        "user_message": user_message or message,
    }


def _try_read_content(path: Path, form_type: str) -> str:
    if form_type == "text":
        return path.read_text(encoding="utf-8", errors="strict").strip()

    if form_type == "pdf":
        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = (page.extract_text(x_tolerance=10, y_tolerance=5) or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    if form_type == "docx":
        from extract_document import extract_docx

        extracted = extract_docx(path)
        return extracted.get("full_text", "").strip()

    raise ValueError(f"Unsupported form type: {form_type}")


def _is_supported_type(path: Path, config: dict) -> bool:
    form_type = detect_form_type(path)
    allowed_extensions = [ext.lower() for ext in config.get("allowed_extensions", [])]
    allowed_form_types = config.get("allowed_form_types", [])
    return form_type in allowed_form_types and path.suffix.lower() in allowed_extensions


def validate_rules_file(rules_path: Path, config: dict) -> list[dict]:
    checks: list[dict] = []
    rules_path = rules_path.resolve()
    min_rules_chars = config.get("min_rules_content_characters", 20)
    allowed_extensions = [ext.lower() for ext in config.get("rules_file_allowed_extensions", [])]

    if not rules_path.exists():
        checks.append(
            _check(
                "rules_file_exists",
                False,
                f"Rules file not found: {rules_path}",
                _message(config, "rules_file_no_content"),
            )
        )
        return checks

    form_type = detect_form_type(rules_path)
    if rules_path.suffix.lower() not in allowed_extensions or form_type == "unknown":
        checks.append(
            _check(
                "rules_file_type_supported",
                False,
                f"Unsupported rules file type: {rules_path.suffix.lower()}",
                _message(config, "rules_file_no_content"),
            )
        )
        return checks

    try:
        content = _try_read_content(rules_path, form_type)
        content_length = len(content)
        passed = content_length >= min_rules_chars
        checks.append(
            _check(
                "rules_file_has_content",
                passed,
                f"Rules file contains {content_length} character(s)."
                if passed
                else f"Rules file content is too short ({content_length} chars).",
                _message(config, "rules_file_no_content") if not passed else "Rules file content looks readable.",
            )
        )
    except Exception:
        checks.append(
            _check(
                "rules_file_readable",
                False,
                "Could not read rules file content.",
                _message(config, "rules_file_no_content"),
            )
        )

    return checks


def validate_document(
    path: Path,
    config_path: Path | None = None,
    rules_path: Path | None = None,
    extraction_mode: str = "native",
) -> dict:
    path = path.resolve()
    config = load_config(config_path)
    messages = config.get("messages", {})
    extraction_mode = extraction_mode.lower()

    checks: list[dict] = []
    form_type = detect_form_type(path)
    file_size_bytes = path.stat().st_size if path.exists() else 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    max_file_size_mb = config.get("max_file_size_mb", 50)
    min_file_size_bytes = config.get("min_file_size_bytes", 1)
    require_non_empty_content = config.get("require_non_empty_content", True)
    min_content_characters = config.get("min_content_characters", 1)
    enable_rules_file_validation = config.get("enable_rules_file_validation", False)
    ocr_mode = extraction_mode == "ocr"

    if not path.exists():
        checks.append(
            _check(
                "file_corrupt_or_unreadable",
                False,
                f"File not found: {path}",
                messages.get("file_corrupt_or_unreadable", "File not found."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    if not _is_supported_type(path, config):
        checks.append(
            _check(
                "file_type_not_supported",
                False,
                f"Unsupported file type: {path.suffix.lower()} ({form_type})",
                messages.get("file_type_not_supported", "Unsupported file type."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    if ocr_mode and form_type != "pdf":
        checks.append(
            _check(
                "ocr_pdf_only",
                False,
                f"OCR mode supports PDF only, got: {form_type}",
                messages.get("file_type_not_supported", "Unsupported file type."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    if file_size_bytes < min_file_size_bytes:
        checks.append(
            _check(
                "extraction_empty",
                False,
                "File is empty.",
                messages.get("extraction_empty", "Document is empty."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    if file_size_mb > max_file_size_mb:
        checks.append(
            _check(
                "file_too_large",
                False,
                f"File size is {file_size_mb} MB (limit: {max_file_size_mb} MB).",
                messages.get("file_too_large", "File is too large."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    content = ""
    try:
        if ocr_mode:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
            checks.append(
                _check(
                    "file_readable",
                    True,
                    f"PDF opened successfully ({page_count} page(s)).",
                    "File is readable.",
                )
            )
            checks.append(
                _check(
                    "ocr_mode_ready",
                    True,
                    "OCR mode enabled. Native text check skipped; OCR runs during extraction.",
                    "OCR mode ready.",
                )
            )
        else:
            content = _try_read_content(path, form_type)
            checks.append(
                _check(
                    "file_readable",
                    True,
                    "File opened and read successfully.",
                    "File is readable.",
                )
            )
    except Exception as error:
        checks.append(
            _check(
                "file_corrupt_or_unreadable",
                False,
                f"Could not read file: {error}",
                messages.get("file_corrupt_or_unreadable", "Could not read file."),
            )
        )
        return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)

    if require_non_empty_content and not ocr_mode:
        content_length = len(content)
        passed = content_length >= min_content_characters
        checks.append(
            _check(
                "extraction_not_empty",
                passed,
                f"Extracted {content_length} character(s) from document."
                if passed
                else f"Text extraction returned {content_length} character(s).",
                messages.get("extraction_empty", "Document extraction returned no content.")
                if not passed
                else "Document contains extractable text.",
            )
        )

    if enable_rules_file_validation and rules_path is not None:
        checks.extend(validate_rules_file(rules_path, config))
    elif enable_rules_file_validation and rules_path is None:
        checks.append(
            _check(
                "rules_file_provided",
                False,
                "Rules file validation is enabled but no rules file was uploaded.",
                messages.get("rules_file_no_content", "Rules file is missing."),
            )
        )

    return _build_report(path, form_type, file_size_mb, checks, rules_path, config, extraction_mode)


def _build_report(
    path: Path,
    form_type: str,
    file_size_mb: float,
    checks: list[dict],
    rules_path: Path | None,
    config: dict,
    extraction_mode: str = "native",
) -> dict:
    passed = all(check["passed"] for check in checks)
    errors = [check["user_message"] for check in checks if not check["passed"]]

    return {
        "valid": passed,
        "file_name": path.name,
        "file_path": str(path),
        "form_type": form_type,
        "file_size_mb": file_size_mb,
        "extraction_mode": extraction_mode,
        "rules_file_validation_enabled": config.get("enable_rules_file_validation", False),
        "rules_file_path": str(rules_path.resolve()) if rules_path else None,
        "errors": errors,
        "checks": checks,
    }


def save_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# Set paths here (relative to project root or absolute path)
DOCUMENT_PATH = PROJECT_ROOT / "docs" / "Deployment Report_v0 - filled.pdf"
RULES_PATH = None


if __name__ == "__main__":
    document_path = DOCUMENT_PATH
    run_ts = run_timestamp()
    output_path = build_output_path(
        PROJECT_ROOT / "output" / "validation", document_path.stem, "validation", run_ts
    )

    try:
        report = validate_document(document_path, rules_path=RULES_PATH)
        report["run_timestamp"] = run_ts
        save_report(report, output_path)

        print("=" * 60)
        print("Document Validation")
        print("=" * 60)
        print(f"File:      {report['file_name']}")
        print(f"Form type: {report['form_type']}")
        print(f"Size:      {report['file_size_mb']} MB")
        print(f"Valid:     {report['valid']}")
        print()

        for check in report["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['message']}")
            if not check["passed"]:
                print(f"         -> {check['user_message']}")

        if report["errors"]:
            print()
            print("Errors:")
            for error in report["errors"]:
                print(f"  - {error}")

        print()
        print(f"Report saved: {output_path}")
        print("=" * 60)

        if not report["valid"]:
            sys.exit(1)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
