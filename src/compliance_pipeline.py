"""End-to-end compliance pipeline for CLI and Streamlit UI.

Stages:
  validate → extract → [chunk → RAG match] → hybrid compliance → report JSON

Hybrid mode splits rules by ``rule_type``:
  - semantic    → LLM (whole doc or per-chunk after RAG)
  - deterministic → Python heuristics on full extracted text
  - existential   → Python checks against external JSON registries
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from check_compliance import (
    check_compliance,
    check_compliance_whole_document,
    load_rules,
)
from chunk_document import chunk_document
from extract_document import extract_document
from src.logging_config import get_logger, setup_logging
from src.output_utils import build_output_path, run_timestamp
from src.rag.chunk_matcher import match_rules_to_chunks, save_chunk_rule_matches
from validate_document import validate_document

logger = get_logger("pipeline")

DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "compliance.json"
WHOLE_DOC_PROMPT = PROJECT_ROOT / "prompts" / "compliance_check_whole_doc.txt"
CHUNK_RAG_PROMPT = PROJECT_ROOT / "prompts" / "compliance_check_chunk.txt"


def load_compliance_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def is_hybrid_enabled(compliance_mode: str) -> bool:
    config = load_compliance_config()
    section = config.get(compliance_mode, {})
    return bool(section.get("hybrid", True))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_rules_path(rules_path: Path | None) -> Path:
    return rules_path if rules_path else DEFAULT_RULES_PATH


def resolve_prompt_path(compliance_mode: str) -> Path:
    config = load_compliance_config()
    if compliance_mode == "whole_doc":
        prompt = config.get("whole_doc", {}).get("prompt", str(WHOLE_DOC_PROMPT))
        return PROJECT_ROOT / prompt
    prompt = config.get("chunk_rag", {}).get("prompt", str(CHUNK_RAG_PROMPT))
    return PROJECT_ROOT / prompt


def run_compliance_pipeline(
    document_path: Path,
    *,
    extraction_mode: str = "native",
    compliance_mode: str = "whole_doc",
    rules_path: Path | None = None,
    use_dummy_llm: bool = True,
    run_ts: str | None = None,
) -> dict:
    document_path = document_path.resolve()
    extraction_mode = extraction_mode.lower()
    compliance_mode = compliance_mode.lower().replace("-", "_")
    if compliance_mode in {"per_chunk", "chunk"}:
        compliance_mode = "chunk_rag"
    if compliance_mode not in {"whole_doc", "chunk_rag"}:
        raise ValueError("compliance_mode must be 'whole_doc' or 'chunk_rag'")

    run_ts = run_ts or run_timestamp()
    stem = document_path.stem
    resolved_rules = resolve_rules_path(rules_path)
    prompt_path = resolve_prompt_path(compliance_mode)

    setup_logging()
    logger.info(
        "Starting pipeline run_id=%s doc=%s mode=%s extraction=%s llm=%s rules=%s",
        run_ts,
        document_path.name,
        compliance_mode,
        extraction_mode,
        "dummy" if use_dummy_llm else "model_garden",
        resolved_rules,
    )

    validation_output = build_output_path(
        PROJECT_ROOT / "output" / "validation", stem, "validation", run_ts
    )
    extracted_output = build_output_path(
        PROJECT_ROOT / "output" / "extracted", stem, "extracted", run_ts
    )
    chunks_output = build_output_path(
        PROJECT_ROOT / "output" / "chunks", stem, "chunks", run_ts
    )
    matches_output = build_output_path(
        PROJECT_ROOT / "output" / "matches", stem, "chunk_rules", run_ts
    )
    report_output = build_output_path(
        PROJECT_ROOT / "output" / "reports", stem, "report", run_ts
    )

    validation = validate_document(
        document_path,
        rules_path=rules_path,
        extraction_mode=extraction_mode,
    )
    validation["run_timestamp"] = run_ts
    save_json(validation, validation_output)
    if not validation["valid"]:
        logger.warning("Validation failed: %s", validation.get("errors"))
        return {
            "success": False,
            "run_timestamp": run_ts,
            "stage": "validation",
            "validation": validation,
            "errors": validation.get("errors", []),
        }

    extracted = extract_document(document_path, mode=extraction_mode)
    extracted["run_timestamp"] = run_ts
    save_json(extracted, extracted_output)
    page_count = extracted.get("page_count", 0)
    logger.info(
        "Extraction complete: pages=%s chars=%s",
        page_count,
        len(extracted.get("full_text", "")),
    )
    if not extracted.get("full_text", "").strip():
        logger.error("Extraction returned empty text")
        return {
            "success": False,
            "run_timestamp": run_ts,
            "stage": "extraction",
            "validation": validation,
            "extracted": extracted,
            "errors": ["Text extraction returned no content."],
        }

    rules = load_rules(resolved_rules)
    hybrid = is_hybrid_enabled(compliance_mode)
    logger.info("Loaded %s rule(s), hybrid=%s", len(rules), hybrid)

    chunks = None
    chunk_rule_matches = None
        
    if compliance_mode == "whole_doc":
        logger.info("Running whole-document hybrid compliance check")
        report = check_compliance_whole_document(
            extracted,
            rules,
            file_name=document_path.name,
            prompt_path=prompt_path,
            model=None,
            use_dummy_llm=use_dummy_llm,
            hybrid=hybrid,
        )
    else:
        
        chunks = chunk_document(extracted)
        chunks["run_timestamp"] = run_ts
        chunks["pages"] = extracted.get("pages", [])
        chunks["page_count"] = extracted.get("page_count", 1)
        save_json(chunks, chunks_output)
        logger.info("Chunking complete: %s chunk(s)", chunks.get("chunk_count", 0))

        chunk_rule_matches = match_rules_to_chunks(chunks, rules)
        chunk_rule_matches["run_timestamp"] = run_ts
        save_chunk_rule_matches(chunk_rule_matches, matches_output)
        logger.info(
            "RAG matching complete: %s chunk(s) with rule matches",
            len(chunk_rule_matches.get("matches", [])),
        )

        report = check_compliance(
            chunks,
            rules,
            mode="chunk_rag",
            file_name=document_path.name,
            prompt_path=prompt_path,
            model=None,
            include_all_rules=True,
            use_dummy_llm=use_dummy_llm,
            chunk_rule_matches=chunk_rule_matches,
            extracted=extracted,
            hybrid=hybrid,
        )

    report["run_timestamp"] = run_ts
    report["source_run_timestamp"] = run_ts
    report["extraction_mode"] = extraction_mode
    report["llm_engine"] = "dummy" if use_dummy_llm else "model_garden"
    if chunk_rule_matches:
        report["chunk_rule_matches_path"] = str(matches_output)
    save_json(report, report_output)
    summary = report.get("summary", {})
    logger.info(
        "Pipeline complete run_id=%s status=%s passed=%s failed=%s needs_review=%s report=%s",
        run_ts,
        summary.get("overall_status"),
        summary.get("passed"),
        summary.get("failed"),
        summary.get("insufficient_evidence"),
        report_output,
    )

    result = {
        "success": True,
        "run_timestamp": run_ts,
        "stage": "complete",
        "validation": validation,
        "extracted": extracted,
        "report": report,
        "paths": {
            "validation": str(validation_output),
            "extracted": str(extracted_output),
            "report": str(report_output),
        },
    }
    if chunks is not None:
        result["chunks"] = chunks
        result["paths"]["chunks"] = str(chunks_output)
    if chunk_rule_matches is not None:
        result["chunk_rule_matches"] = chunk_rule_matches
        result["paths"]["chunk_rules"] = str(matches_output)
    return result
