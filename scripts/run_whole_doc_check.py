"""Flow 1: validate → extract → whole doc + all rules → LLM compliance check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_pipeline import run_compliance_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-document compliance check (all 13 rules sent to LLM)."
    )
    parser.add_argument("document", help="Path to PDF/DOCX/TXT")
    parser.add_argument(
        "--extraction",
        choices=["native", "ocr"],
        default="native",
        help="Extraction engine (default: native)",
    )
    parser.add_argument(
        "--llm",
        choices=["dummy", "model_garden"],
        default="dummy",
        help="LLM engine (default: dummy)",
    )
    args = parser.parse_args()

    result = run_compliance_pipeline(
        Path(args.document),
        extraction_mode=args.extraction,
        compliance_mode="whole_doc",
        use_dummy_llm=(args.llm == "dummy"),
    )

    if not result.get("success"):
        raise SystemExit(f"Pipeline failed at stage: {result.get('stage')}")

    report = result["report"]
    summary = report["summary"]
    print(f"Mode:     {report['mode']} ({report['rule_retrieval']} rules)")
    print(f"Status:   {summary['overall_status']}")
    print(f"Report:   {result['paths']['report']}")


if __name__ == "__main__":
    main()
