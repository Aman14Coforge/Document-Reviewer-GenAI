"""Hybrid compliance check: semantic rules via LLM, deterministic rules via Python."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from check_compliance import check_compliance_whole_document, load_rules
from extract_document import extract_document
from src.logging_config import setup_logging
from src.output_utils import build_output_path, run_timestamp

WHOLE_DOC_PROMPT = PROJECT_ROOT / "prompts" / "compliance_check_whole_doc.txt"
DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid compliance check: semantic rules → Model Garden LLM, "
            "deterministic rules → Python logic."
        )
    )
    parser.add_argument("document", help="Path to PDF/DOCX/TXT")
    parser.add_argument("--rules", help="Path to rules JSON (default: rules/rules.json)")
    parser.add_argument(
        "--extraction",
        choices=["native", "ocr"],
        default="native",
        help="Text extraction mode (default: native)",
    )
    parser.add_argument(
        "--llm",
        choices=["dummy", "model_garden"],
        default="model_garden",
        help="LLM engine for semantic rules (default: model_garden)",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Send all rules to LLM (disable Python deterministic checks)",
    )
    parser.add_argument("-o", "--output", help="Output report JSON path")
    args = parser.parse_args()

    document_path = Path(args.document).resolve()
    rules_path = Path(args.rules or DEFAULT_RULES_PATH)
    if not document_path.exists():
        raise SystemExit(f"Document not found: {document_path}")
    if not rules_path.exists():
        raise SystemExit(f"Rules file not found: {rules_path}")

    extracted = extract_document(document_path, mode=args.extraction)
    if not extracted.get("full_text", "").strip():
        raise SystemExit("Text extraction returned no content.")

    rules = load_rules(rules_path)
    run_ts = run_timestamp()
    report = check_compliance_whole_document(
        extracted,
        rules,
        file_name=document_path.name,
        prompt_path=WHOLE_DOC_PROMPT,
        model=None,
        use_dummy_llm=(args.llm == "dummy"),
        hybrid=not args.llm_only,
    )
    report["run_timestamp"] = run_ts
    report["rules_path"] = str(rules_path)

    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(
            PROJECT_ROOT / "output" / "reports",
            document_path.stem,
            "hybrid_report",
            run_ts,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    print(f"Mode:          {report['mode']}")
    print(f"Semantic:      {report.get('semantic_rule_count', 0)} rules (LLM)")
    print(f"Deterministic: {report.get('deterministic_rule_count', 0)} rules (Python)")
    print(f"Existential:   {report.get('existential_rule_count', 0)} rules (external JSON)")
    print(f"Overall:       {summary['overall_status']}")
    print(
        "Passed: {passed} | Failed: {failed} | N/A: {not_applicable} | Needs review: {insufficient_evidence}".format(
            **summary
        )
    )
    print(f"Report:        {output_path}")


if __name__ == "__main__":
    main()
