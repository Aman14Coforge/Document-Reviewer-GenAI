"""Flow 2: validate → extract → chunk → RAG match → chunk LLM compliance check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_pipeline import run_compliance_pipeline
from src.logging_config import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Chunk + RAG compliance check (vector-matched rules per chunk)."
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
        help="LLM engine (default: dummy for local test)",
    )
    parser.add_argument(
        "--rules",
        help="Path to rules JSON file (default: rules/rules.json)",
    )
    args = parser.parse_args()

    result = run_compliance_pipeline(
        Path(args.document),
        extraction_mode=args.extraction,
        compliance_mode="chunk_rag",
        use_dummy_llm=(args.llm == "dummy"),
        rules_path=Path("output/rules/generated_rules_20260630_112055.json"
    )

    if not result.get("success"):
        raise SystemExit(f"Pipeline failed at stage: {result.get('stage')}")

    report = result["report"]
    summary = report["summary"]
    matches = result.get("chunk_rule_matches", {})
    print(f"Mode:     {report['mode']} ({report['rule_retrieval']} rules)")
    print(f"Chunks:   {result['chunks']['chunk_count']}")
    print(f"Matched:  {len(matches.get('matches', []))} chunk(s)")
    print(f"Status:   {summary['overall_status']}")
    print(f"Matches:  {result['paths']['chunk_rules']}")
    print(f"Report:   {result['paths']['report']}")


if __name__ == "__main__":
    main()
