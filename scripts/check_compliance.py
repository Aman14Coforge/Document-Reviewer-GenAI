"""Run compliance checks on document chunks using an LLM.

Supports three evaluation paths (hybrid mode):
  - semantic rules     → LLM per chunk or whole document
  - deterministic rules → Python heuristics on full extracted text
  - existential rules   → Python checks vs external JSON registries

Chunk-level semantic results are merged with ``pass_if_min_chunk_passes``
from config (default: pass when 2+ chunks return passed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMResponseValidationError, call_llm_json, load_prompt
from src.logging_config import get_logger, setup_logging
from src.llm_schemas import ComplianceCheckResponse
from src.output_utils import build_output_path, run_timestamp
from src.deterministic_checker import (
    enrich_deterministic_results,
    evaluate_deterministic_rules,
    split_rules_by_type,
)
from src.existential_checker import (
    enrich_existential_results,
    evaluate_existential_rules,
)

try:
    from src.rag.chunk_matcher import load_chunk_rag_config
except ImportError:
    load_chunk_rag_config = lambda: {}

try:
    from src.rag.rule_store import load_compliance_config
except ImportError:
    def load_compliance_config() -> dict:
        config_path = PROJECT_ROOT / "config" / "compliance.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}


def load_aggregation_config() -> dict:
    return load_compliance_config().get(
        "aggregation",
        {
            "pass_if_min_chunk_passes": 2,
            "strategy": "pass_threshold",
        },
    )

logger = get_logger("check")

STATUS_PRIORITY = {
    "failed": 4,
    "insufficient_evidence": 3,
    "not_applicable": 2,
    "passed": 1,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(path: Path) -> list[dict]:
    data = load_json(path)
    rules = data.get("rules", data)
    if not isinstance(rules, list):
        raise ValueError("Rules file must contain a top-level 'rules' array.")
    return rules


def select_rules_for_chunk(rules: list[dict], chunk: dict, *, include_all: bool) -> list[dict]:
    if include_all:
        return rules

    section_type = chunk.get("section_type", "body")
    selected = []
    for rule in rules:
        sections = rule.get("applies_to_sections", [])
        if rule.get("always_include") or section_type in sections or "full" in sections:
            selected.append(rule)
    return selected or rules


def normalize_chunk_results(
    results: list[dict],
    selected_rules: list[dict],
    *,
    chunk_id: str,
    section_type: str,
) -> tuple[list[dict], dict]:
    """Keep only matched rules, fill missing rule IDs, dedupe by worst status."""
    expected_ids = [rule["rule_id"] for rule in selected_rules]
    expected_set = set(expected_ids)
    stats = {"dropped_unmatched_rule_ids": [], "filled_missing_rule_ids": []}

    by_rule_id: dict[str, dict] = {}
    for item in results:
        rule_id = item.get("rule_id")
        if not rule_id or rule_id not in expected_set:
            if rule_id:
                stats["dropped_unmatched_rule_ids"].append(rule_id)
            continue

        incoming_status = item.get("status", "insufficient_evidence")
        current = by_rule_id.get(rule_id)
        if not current or STATUS_PRIORITY.get(incoming_status, 0) > STATUS_PRIORITY.get(
            current.get("status", "insufficient_evidence"), 0
        ):
            by_rule_id[rule_id] = item

    normalized = []
    for rule_id in expected_ids:
        if rule_id in by_rule_id:
            normalized.append(by_rule_id[rule_id])
            continue
        stats["filled_missing_rule_ids"].append(rule_id)
        normalized.append(
            {
                "rule_id": rule_id,
                "status": "insufficient_evidence",
                "reason": "No LLM verdict returned for this matched rule on this chunk.",
                "evidence": "",
                "confidence": 0.0,
                "chunk_id": chunk_id,
                "section_type": section_type,
            }
        )

    return normalized, stats


def slim_rules_for_prompt(rules: list[dict]) -> list[dict]:
    return [
        {
            "rule_id": rule.get("rule_id"),
            "title": rule.get("title"),
            "verifiable_criteria": rule.get("verifiable_criteria"),
        }
        for rule in rules
    ]


def _call_compliance_llm(
    prompt: str,
    *,
    model: str | None,
    use_dummy_llm: bool,
    chunk_id: str,
) -> dict:
    try:
        return call_llm_json(
            prompt,
            model=model,
            use_dummy=use_dummy_llm,
            response_model=ComplianceCheckResponse,
        )
    except LLMResponseValidationError as error:
        logger.warning(
            "LLM response failed schema validation for chunk=%s, retrying once: %s",
            chunk_id,
            error,
        )
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous JSON was invalid. Return ONLY valid JSON with this shape:\n"
            '{"results": [{"rule_id": "GDP-XX", "status": "passed|failed|not_applicable|insufficient_evidence", '
            '"reason": "...", "evidence": "...", "confidence": 0.0}]}\n'
            "Do not add markdown fences or extra keys."
        )
        return call_llm_json(
            retry_prompt,
            model=model,
            use_dummy=use_dummy_llm,
            response_model=ComplianceCheckResponse,
        )


def evaluate_chunk(
    chunk: dict,
    rules: list[dict],
    *,
    file_name: str,
    prompt_path: Path,
    model: str | None,
    include_all_rules: bool,
    use_dummy_llm: bool = False,
    chunk_rules: list[dict] | None = None,
) -> dict:
    if chunk_rules is not None:
        selected_rules = chunk_rules
    else:
        selected_rules = select_rules_for_chunk(rules, chunk, include_all=include_all_rules)

    document_content = chunk.get("text", "")
    if chunk.get("chunk_id") != "whole_document":
        document_content = (
            f"Chunk ID: {chunk.get('chunk_id')}\n"
            f"Section Type: {chunk.get('section_type')}\n"
            f"Heading: {chunk.get('heading')}\n"
            f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n\n"
            f"{chunk.get('text', '')}"
        )

    rag_config = load_chunk_rag_config()
    rules_for_prompt = (
        slim_rules_for_prompt(selected_rules)
        if rag_config.get("slim_rules_in_prompt", True) and chunk_rules is not None
        else selected_rules
    )

    prompt = load_prompt(
        prompt_path,
        FILE_NAME=file_name,
        CHUNK_ID=str(chunk.get("chunk_id", "")),
        RULE_COUNT=str(len(selected_rules)),
        RULE_IDS=", ".join(rule["rule_id"] for rule in selected_rules),
        RULES_JSON=json.dumps(rules_for_prompt, indent=2, ensure_ascii=False),
        DOCUMENT_CONTENT=document_content,
    )
    normalize_results = rag_config.get("normalize_llm_results", True)
    retry_incomplete = rag_config.get("retry_incomplete_llm_response", False)

    rule_ids = [rule["rule_id"] for rule in selected_rules]
    logger.debug(
        "LLM evaluate chunk=%s rules=%s engine=%s",
        chunk.get("chunk_id"),
        rule_ids,
        "dummy" if use_dummy_llm else "model_garden",
    )

    response = _call_compliance_llm(
        prompt,
        model=model,
        use_dummy_llm=use_dummy_llm,
        chunk_id=str(chunk.get("chunk_id", "")),
    )
    results = response.get("results", [])
    normalization_stats = {}

    if normalize_results and chunk_rules is not None:
        results, normalization_stats = normalize_chunk_results(
            results,
            selected_rules,
            chunk_id=str(chunk.get("chunk_id", "")),
            section_type=str(chunk.get("section_type", "")),
        )

        if retry_incomplete and normalization_stats.get("filled_missing_rule_ids"):
            retry_prompt = (
                f"{prompt}\n\n"
                f"Your previous response was incomplete. "
                f"Return exactly {len(selected_rules)} results, one for each rule ID: "
                f"{', '.join(rule['rule_id'] for rule in selected_rules)}."
            )
            retry_response = _call_compliance_llm(
                retry_prompt,
                model=model,
                use_dummy_llm=use_dummy_llm,
                chunk_id=str(chunk.get("chunk_id", "")),
            )
            results, normalization_stats = normalize_chunk_results(
                retry_response.get("results", []),
                selected_rules,
                chunk_id=str(chunk.get("chunk_id", "")),
                section_type=str(chunk.get("section_type", "")),
            )

    for item in results:
        item["chunk_id"] = chunk.get("chunk_id")
        item["section_type"] = chunk.get("section_type")

    output = {
        "chunk_id": chunk.get("chunk_id"),
        "section_type": chunk.get("section_type"),
        "rules_evaluated": [rule["rule_id"] for rule in selected_rules],
        "results": results,
    }
    if normalization_stats:
        output["normalization"] = normalization_stats
    return output


def combine_hybrid_results(
    rule_catalog: list[dict],
    deterministic_results: list[dict],
    llm_results: list[dict],
    existential_results: list[dict] | None = None,
) -> list[dict]:
    det_map = {item["rule_id"]: item for item in deterministic_results}
    ex_map = {item["rule_id"]: item for item in (existential_results or [])}
    llm_map = {item["rule_id"]: item for item in llm_results}
    combined: list[dict] = []

    for rule in rule_catalog:
        rule_id = rule["rule_id"]
        if rule_id in det_map:
            combined.append(det_map[rule_id])
            continue
        if rule_id in ex_map:
            combined.append(ex_map[rule_id])
            continue
        if rule_id in llm_map:
            item = dict(llm_map[rule_id])
            item["check_method"] = "llm"
            combined.append(item)
            continue
        combined.append(
            {
                "rule_id": rule_id,
                "title": rule.get("title", ""),
                "status": "insufficient_evidence",
                "reason": "Rule was not evaluated.",
                "evidence": "",
                "confidence": 0.0,
                "severity": rule.get("severity", ""),
                "recommendation": rule.get("recommendation", ""),
                "rule_type": rule.get("rule_type", ""),
                "evidence_chunks": [],
            }
        )
    return combined


def extracted_from_chunks(chunks_data: dict) -> dict:
    chunks = chunks_data.get("chunks", [])
    full_text = chunks_data.get("full_text", "")
    for chunk in chunks:
        if chunk.get("chunk_id") == "full_document":
            full_text = chunk.get("text", full_text)
            break
    return {
        "full_text": full_text,
        "pages": chunks_data.get("pages", []),
        "page_count": chunks_data.get("page_count", 1),
        "source_path": chunks_data.get("source_path"),
        "file_name": chunks_data.get("file_name"),
    }


def aggregate_chunk_votes(
    rule_id: str,
    votes: list[dict],
    rule_map: dict[str, dict],
    *,
    pass_threshold: int,
    strategy: str,
) -> dict:
    """Merge per-chunk LLM votes into one rule-level result.

    When ``strategy`` is ``pass_threshold`` and at least ``pass_threshold`` chunks
    returned ``passed``, the rule passes regardless of failures on other chunks.
    Otherwise the worst status across chunks wins.
    """
    evidence_chunks: list[str] = []
    for vote in votes:
        chunk_id = vote.get("chunk_id")
        if chunk_id and chunk_id not in evidence_chunks:
            evidence_chunks.append(chunk_id)

    rule = rule_map.get(rule_id, {})
    rule_type = rule.get("rule_type", "semantic")

    pass_votes = [vote for vote in votes if vote.get("status") == "passed"]

    # Semantic rules are existence-based: a single pass is sufficient
    if rule_type == "semantic" and pass_votes:
        logger.debug(
            "Rule %s passed via semantic existence (%s/%s chunks passed)",
            rule_id,
            len(pass_votes),
            len(votes),
        )
        best = max(pass_votes, key=lambda vote: float(vote.get("confidence") or 0))
        evidence_parts = [vote.get("evidence", "") for vote in pass_votes if vote.get("evidence")]
        return {
            **best,
            "rule_id": rule_id,
            "status": "passed",
            "title": rule_map.get(rule_id, {}).get("title", best.get("title", "")),
            "severity": rule_map.get(rule_id, {}).get("severity", best.get("severity", "")),
            "recommendation": rule_map.get(rule_id, {}).get(
                "recommendation", best.get("recommendation", "")
            ),
            "reason": best.get(
                "reason",
                f"Rule passed in {len(pass_votes)} chunk(s) (threshold: {pass_threshold}).",
            ),
            "evidence": "; ".join(evidence_parts) if evidence_parts else best.get("evidence", ""),
            "evidence_chunks": evidence_chunks,
            "chunk_pass_count": len(pass_votes),
            "chunk_eval_count": len(votes),
        }

    merged = None
    for vote in votes:
        incoming_status = vote.get("status", "insufficient_evidence")
        if merged is None:
            merged = {
                **vote,
                "title": rule_map.get(rule_id, {}).get("title", ""),
                "severity": rule_map.get(rule_id, {}).get("severity", ""),
                "recommendation": rule_map.get(rule_id, {}).get("recommendation", ""),
                "evidence_chunks": evidence_chunks.copy(),
            }
            continue

        current_status = merged.get("status", "insufficient_evidence")
        if STATUS_PRIORITY.get(incoming_status, 0) > STATUS_PRIORITY.get(current_status, 0):
            merged.update(
                {
                    "status": incoming_status,
                    "reason": vote.get("reason", merged.get("reason", "")),
                    "evidence": vote.get("evidence", merged.get("evidence", "")),
                    "confidence": vote.get("confidence", merged.get("confidence", 0)),
                }
            )
        elif incoming_status == current_status and vote.get("evidence"):
            merged["evidence"] = (
                f"{merged.get('evidence', '')}; {vote.get('evidence', '')}".strip("; ")
            )

    merged = merged or {}
    merged["chunk_pass_count"] = len(pass_votes)
    merged["chunk_eval_count"] = len(votes)
    if merged.get("status") != "passed":
        logger.debug(
            "Rule %s aggregated as %s from %s chunk vote(s), %s pass(es)",
            rule_id,
            merged.get("status"),
            len(votes),
            len(pass_votes),
        )
    return merged


def merge_results(
    rule_catalog: list[dict],
    chunk_outputs: list[dict],
) -> list[dict]:
    rule_map = {rule["rule_id"]: rule for rule in rule_catalog}
    agg_config = load_aggregation_config()
    pass_threshold = int(agg_config.get("pass_if_min_chunk_passes", 2))
    strategy = str(agg_config.get("strategy", "pass_threshold"))

    votes_by_rule: dict[str, list[dict]] = {}
    for chunk_output in chunk_outputs:
        allowed_rule_ids = set(chunk_output.get("rules_evaluated", []))
        for item in chunk_output.get("results", []):
            rule_id = item.get("rule_id")
            if not rule_id or rule_id not in allowed_rule_ids:
                continue
            # Ignore not_applicable votes; they must not influence aggregation
            if item.get("status") == "not_applicable":
                continue
            votes_by_rule.setdefault(rule_id, []).append(item)

    merged: dict[str, dict] = {}
    for rule_id, votes in votes_by_rule.items():
        merged[rule_id] = aggregate_chunk_votes(
            rule_id,
            votes,
            rule_map,
            pass_threshold=pass_threshold,
            strategy=strategy,
        )

    ordered = []
    for rule in rule_catalog:
        rule_id = rule["rule_id"]
        if rule_id in merged:
            ordered.append(merged[rule_id])
        else:
            ordered.append(
                {
                    "rule_id": rule_id,
                    "title": rule.get("title", ""),
                    "status": "insufficient_evidence",
                    "reason": "No chunk provided enough evidence to evaluate this rule.",
                    "evidence": "",
                    "confidence": 0.0,
                    "severity": rule.get("severity", ""),
                    "recommendation": rule.get("recommendation", ""),
                    "evidence_chunks": [],
                    "chunk_pass_count": 0,
                    "chunk_eval_count": 0,
                }
            )
    return ordered


def summarize(results: list[dict]) -> dict:
    counts = {"passed": 0, "failed": 0, "not_applicable": 0, "insufficient_evidence": 0}
    for item in results:
        status = item.get("status", "insufficient_evidence")
        counts[status] = counts.get(status, 0) + 1

    overall_status = "compliant"
    if counts["failed"] > 0:
        overall_status = "non_compliant"
    elif counts["insufficient_evidence"] > 0:
        overall_status = "needs_review"

    return {
        "overall_status": overall_status,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "not_applicable": counts["not_applicable"],
        "insufficient_evidence": counts["insufficient_evidence"],
        "total_rules": len(results),
    }


def check_compliance_whole_document(
    extracted: dict,
    rules: list[dict],
    *,
    file_name: str | None,
    prompt_path: Path,
    model: str | None,
    use_dummy_llm: bool = False,
    hybrid: bool = True,
) -> dict:
    resolved_file_name = file_name or extracted.get("file_name", "unknown")
    semantic_rules, deterministic_rules, existential_rules = split_rules_by_type(rules)
    logger.info(
        "Whole-doc check: semantic=%s deterministic=%s existential=%s hybrid=%s llm=%s",
        len(semantic_rules),
        len(deterministic_rules),
        len(existential_rules),
        hybrid,
        "dummy" if use_dummy_llm else "model_garden",
    )

    deterministic_results: list[dict] = []
    if hybrid and deterministic_rules:
        deterministic_results = enrich_deterministic_results(
            evaluate_deterministic_rules(
                extracted,
                deterministic_rules,
                file_name=resolved_file_name,
            ),
            rules,
        )

    existential_results: list[dict] = []
    if hybrid and existential_rules:
        existential_results = enrich_existential_results(
            evaluate_existential_rules(
                extracted,
                existential_rules,
                file_name=resolved_file_name,
            ),
            rules,
        )

    semantic_chunk_output = None
    semantic_merged: list[dict] = []
    if semantic_rules:
        whole_chunk = {
            "chunk_id": "whole_document",
            "section_type": "full",
            "heading": "Complete Document",
            "page_start": 1,
            "page_end": extracted.get("page_count", 1),
            "text": extracted.get("full_text", ""),
        }
        semantic_chunk_output = evaluate_chunk(
            whole_chunk,
            semantic_rules,
            file_name=resolved_file_name,
            prompt_path=prompt_path,
            model=model,
            include_all_rules=True,
            use_dummy_llm=use_dummy_llm,
            chunk_rules=semantic_rules,
        )
        semantic_merged = merge_results(semantic_rules, [semantic_chunk_output])
        for item in semantic_merged:
            item["check_method"] = "llm"
            item["rule_type"] = "semantic"

    if hybrid:
        merged_results = combine_hybrid_results(
            rules,
            deterministic_results,
            semantic_merged,
            existential_results,
        )
        chunk_outputs = [output for output in [semantic_chunk_output] if output]
    else:
        whole_chunk = {
            "chunk_id": "whole_document",
            "section_type": "full",
            "heading": "Complete Document",
            "page_start": 1,
            "page_end": extracted.get("page_count", 1),
            "text": extracted.get("full_text", ""),
        }
        chunk_output = evaluate_chunk(
            whole_chunk,
            rules,
            file_name=resolved_file_name,
            prompt_path=prompt_path,
            model=model,
            include_all_rules=True,
            use_dummy_llm=use_dummy_llm,
            chunk_rules=rules,
        )
        merged_results = merge_results(rules, [chunk_output])
        chunk_outputs = [chunk_output]

    summary = summarize(merged_results)
    logger.info(
        "Whole-doc summary: %s (passed=%s failed=%s needs_review=%s)",
        summary["overall_status"],
        summary["passed"],
        summary["failed"],
        summary["insufficient_evidence"],
    )

    return {
        "source_path": extracted.get("source_path"),
        "file_name": resolved_file_name,
        "mode": "whole_doc_hybrid" if hybrid else "whole_doc",
        "rule_retrieval": "all",
        "hybrid": hybrid,
        "semantic_rule_count": len(semantic_rules),
        "deterministic_rule_count": len(deterministic_rules),
        "existential_rule_count": len(existential_rules),
        "summary": summary,
        "deterministic_results": deterministic_results,
        "existential_results": existential_results,
        "chunk_outputs": chunk_outputs,
        "results": merged_results,
    }


def check_compliance_with_rag(
    chunks_data: dict,
    rules: list[dict],
    chunk_rule_matches: dict,
    *,
    file_name: str | None,
    prompt_path: Path,
    model: str | None,
    use_dummy_llm: bool = False,
    extracted: dict | None = None,
    hybrid: bool = True,
) -> dict:
    resolved_file_name = file_name or chunks_data.get("file_name", "unknown")
    semantic_rules, deterministic_rules, existential_rules = split_rules_by_type(rules)
    semantic_rule_ids = {rule["rule_id"] for rule in semantic_rules}
    logger.info(
        "Chunk-RAG check: semantic=%s deterministic=%s existential=%s chunks_matched=%s",
        len(semantic_rules),
        len(deterministic_rules),
        len(existential_rules),
        len(chunk_rule_matches.get("matches", [])),
    )

    doc = extracted or extracted_from_chunks(chunks_data)

    deterministic_results: list[dict] = []
    if hybrid and deterministic_rules:
        deterministic_results = enrich_deterministic_results(
            evaluate_deterministic_rules(
                doc,
                deterministic_rules,
                file_name=resolved_file_name,
            ),
            rules,
        )

    existential_results: list[dict] = []
    if hybrid and existential_rules:
        existential_results = enrich_existential_results(
            evaluate_existential_rules(
                doc,
                existential_rules,
                file_name=resolved_file_name,
            ),
            rules,
        )

    match_map = {
        match["chunk_id"]: [
            rule for rule in match["matched_rules"] if rule["rule_id"] in semantic_rule_ids
        ]
        for match in chunk_rule_matches.get("matches", [])
    }

    chunk_outputs = []
    for chunk in chunks_data.get("chunks", []):
        chunk_id = chunk.get("chunk_id")
        chunk_rules = match_map.get(chunk_id, [])
        if not chunk_rules:
            continue
        chunk_outputs.append(
            evaluate_chunk(
                chunk,
                semantic_rules,
                file_name=resolved_file_name,
                prompt_path=prompt_path,
                model=model,
                include_all_rules=True,
                use_dummy_llm=use_dummy_llm,
                chunk_rules=chunk_rules,
            )
        )

    semantic_merged = merge_results(semantic_rules, chunk_outputs) if semantic_rules else []
    for item in semantic_merged:
        item["check_method"] = "llm"
        item["rule_type"] = "semantic"

    merged_results = (
        combine_hybrid_results(
            rules,
            deterministic_results,
            semantic_merged,
            existential_results,
        )
        if hybrid
        else merge_results(rules, chunk_outputs)
    )
    summary = summarize(merged_results)
    logger.info(
        "Chunk-RAG summary: %s (passed=%s failed=%s needs_review=%s)",
        summary["overall_status"],
        summary["passed"],
        summary["failed"],
        summary["insufficient_evidence"],
    )

    return {
        "source_path": chunks_data.get("source_path"),
        "file_name": resolved_file_name,
        "mode": "chunk_rag_hybrid" if hybrid else "chunk_rag",
        "rule_retrieval": "rag",
        "hybrid": hybrid,
        "semantic_rule_count": len(semantic_rules),
        "deterministic_rule_count": len(deterministic_rules),
        "existential_rule_count": len(existential_rules),
        "chunk_rule_matches": chunk_rule_matches,
        "deterministic_results": deterministic_results,
        "existential_results": existential_results,
        "summary": summary,
        "chunk_outputs": chunk_outputs,
        "results": merged_results,
    }


def check_compliance(
    chunks_data: dict,
    rules: list[dict],
    *,
    mode: str,
    file_name: str | None,
    prompt_path: Path,
    model: str | None,
    include_all_rules: bool,
    use_dummy_llm: bool = False,
    chunk_rule_matches: dict | None = None,
    extracted: dict | None = None,
    hybrid: bool = True,
) -> dict:
    chunks = chunks_data.get("chunks", [])
    if not chunks:
        raise ValueError("Chunk file does not contain any chunks.")

    resolved_file_name = file_name or chunks_data.get("file_name", "unknown")

    if mode == "chunk_rag":
        if not chunk_rule_matches:
            raise ValueError("chunk_rag mode requires chunk_rule_matches from vector retrieval.")
        return check_compliance_with_rag(
            chunks_data,
            rules,
            chunk_rule_matches,
            file_name=resolved_file_name,
            prompt_path=prompt_path,
            model=model,
            use_dummy_llm=use_dummy_llm,
            extracted=extracted,
            hybrid=hybrid,
        )

    if mode == "all":
        combined_chunk = {
            "chunk_id": "all_chunks",
            "section_type": "full",
            "heading": "All Chunks Combined",
            "page_start": 1,
            "page_end": chunks_data.get("page_count", 1),
            "text": "\n\n".join(
                (
                    f"[{chunk.get('chunk_id')} | {chunk.get('section_type')} | "
                    f"{chunk.get('heading')}]\n{chunk.get('text', '')}"
                )
                for chunk in chunks
            ),
        }
        chunk_outputs = [
            evaluate_chunk(
                combined_chunk,
                rules,
                file_name=resolved_file_name,
                prompt_path=prompt_path,
                model=model,
                include_all_rules=True,
                use_dummy_llm=use_dummy_llm,
            )
        ]
    else:
        chunk_outputs = []
        for chunk in chunks:
            if chunk.get("chunk_id") == "full_document" and mode == "per-chunk":
                continue
            chunk_outputs.append(
                evaluate_chunk(
                    chunk,
                    rules,
                    file_name=resolved_file_name,
                    prompt_path=prompt_path,
                    model=model,
                    include_all_rules=include_all_rules,
                    use_dummy_llm=use_dummy_llm,
                )
            )

    merged_results = merge_results(rules, chunk_outputs)
    summary = summarize(merged_results)

    return {
        "source_path": chunks_data.get("source_path"),
        "file_name": resolved_file_name,
        "mode": mode,
        "summary": summary,
        "chunk_outputs": chunk_outputs,
        "results": merged_results,
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Check document compliance using an LLM.")
    parser.add_argument("chunks_json", help="Path to chunk JSON produced by chunk_document.py")
    parser.add_argument(
        "--rules",
        help="Path to rules JSON (default: rules/rules.json)",
    )
    parser.add_argument(
        "--prompt",
        help="Path to compliance prompt template (default: prompts/compliance_check.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["per-chunk", "all"],
        default="per-chunk",
        help="per-chunk: one LLM call per chunk; all: one LLM call with all chunks combined",
    )
    parser.add_argument(
        "--include-all-rules",
        action="store_true",
        help="Send all rules on every chunk instead of section-based filtering",
    )
    parser.add_argument("--file-name", help="Override file name used for GDP-01")
    parser.add_argument(
        "-o",
        "--output",
        help="Output report JSON path (default: output/reports/<file_stem>_<timestamp>_report.json)",
    )
    parser.add_argument("--model", help="Override LLM_MODEL")
    args = parser.parse_args()

    chunks_path = Path(args.chunks_json)
    rules_path = Path(args.rules or PROJECT_ROOT / "rules" / "rules.json")
    prompt_path = Path(args.prompt or PROJECT_ROOT / "prompts" / "compliance_check.txt")

    if not chunks_path.exists():
        raise SystemExit(f"Chunk file not found: {chunks_path}")
    if not rules_path.exists():
        raise SystemExit(f"Rules file not found: {rules_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    chunks_data = load_json(chunks_path)
    rules = load_rules(rules_path)

    report = check_compliance(
        chunks_data,
        rules,
        mode="all" if args.mode == "all" else "per-chunk",
        file_name=args.file_name,
        prompt_path=prompt_path,
        model=args.model,
        include_all_rules=args.include_all_rules,
    )

    run_ts = run_timestamp()
    report["run_timestamp"] = run_ts
    if chunks_data.get("run_timestamp"):
        report["source_run_timestamp"] = chunks_data["run_timestamp"]

    stem = chunks_data.get("file_stem") or chunks_path.stem
    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(PROJECT_ROOT / "output" / "reports", stem, "report", run_ts)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Mode: {args.mode}")
    print(f"Overall status: {report['summary']['overall_status']}")
    print(
        "Passed: {passed} | Failed: {failed} | N/A: {not_applicable} | Needs review: {insufficient_evidence}".format(
            **report["summary"]
        )
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
