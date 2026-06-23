"""Run compliance checks on document chunks using an LLM."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.llm_client import call_llm_json, load_prompt
from src.output_utils import build_output_path, run_timestamp
try:
   from src.rag.chunk_matcher import load_chunk_rag_config
except ImportError:
   load_chunk_rag_config = lambda: {}
# STATUS_PRIORITY = {
#    "failed": 4,
#    "insufficient_evidence": 3,
#    "not_applicable": 2,
#    "passed": 1,
# }


STATUS_PRIORITY = {
    "passed": 4,
    "failed": 3,
    "insufficient_evidence": 2,
    "not_applicable": 1
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
           #"keywords": rule.get("keywords"),   # ADD THIS
       }
       for rule in rules
   ]

# def evaluate_chunk(
#    chunk: dict,
#    rules: list[dict],
#    *,
#    file_name: str,
#    prompt_path: Path,
#    model: str | None,
#    include_all_rules: bool,
#    use_dummy_llm: bool = False,
#    chunk_rules: list[dict] | None = None,
# ) -> dict:
#    if chunk_rules is not None:
#        selected_rules = chunk_rules
#    else:
#        selected_rules = select_rules_for_chunk(rules, chunk, include_all=include_all_rules)
#    document_content = chunk.get("text", "")
#    if chunk.get("chunk_id") != "whole_document":
#        document_content = (
#            f"Chunk ID: {chunk.get('chunk_id')}\n"
#            f"Section Type: {chunk.get('section_type')}\n"
#            f"Heading: {chunk.get('heading')}\n"
#            f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n\n"
#            f"{chunk.get('text', '')}"
#        )
#    rag_config = load_chunk_rag_config()
#    rules_for_prompt = (
#        slim_rules_for_prompt(selected_rules)
#        if rag_config.get("slim_rules_in_prompt", True) and chunk_rules is not None
#        else selected_rules
#    )
#    prompt = load_prompt(
#        prompt_path,
#        FILE_NAME=file_name,
#        CHUNK_ID=str(chunk.get("chunk_id", "")),
#        RULE_COUNT=str(len(selected_rules)),
#        RULE_IDS=", ".join(rule["rule_id"] for rule in selected_rules),
#        RULES_JSON=json.dumps(rules_for_prompt, indent=2, ensure_ascii=False),
#        DOCUMENT_CONTENT=document_content,
#    )
#    normalize_results = rag_config.get("normalize_llm_results", True)
#    retry_incomplete = rag_config.get("retry_incomplete_llm_response", False)
#    response = call_llm_json(prompt, model=model, use_dummy=use_dummy_llm)
#    results = response.get("results", [])
#    normalization_stats = {}
#    if normalize_results and chunk_rules is not None:
#        results, normalization_stats = normalize_chunk_results(
#            results,
#            selected_rules,
#            chunk_id=str(chunk.get("chunk_id", "")),
#            section_type=str(chunk.get("section_type", "")),
#        )
#        if retry_incomplete and normalization_stats.get("filled_missing_rule_ids"):
#            retry_prompt = (
#                f"{prompt}\n\n"
#                f"Your previous response was incomplete. "
#                f"Return exactly {len(selected_rules)} results, one for each rule ID: "
#                f"{', '.join(rule['rule_id'] for rule in selected_rules)}."
#            )
#            retry_response = call_llm_json(retry_prompt, model=model, use_dummy=use_dummy_llm)
#            results, normalization_stats = normalize_chunk_results(
#                retry_response.get("results", []),
#                selected_rules,
#                chunk_id=str(chunk.get("chunk_id", "")),
#                section_type=str(chunk.get("section_type", "")),
#            )
#    for item in results:
#        item["chunk_id"] = chunk.get("chunk_id")
#        item["section_type"] = chunk.get("section_type")
#    output = {
#        "chunk_id": chunk.get("chunk_id"),
#        "section_type": chunk.get("section_type"),
#        "rules_evaluated": [rule["rule_id"] for rule in selected_rules],
#        "results": results,
#    }
#    if normalization_stats:
#        output["normalization"] = normalization_stats
#    return output

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
    chunks_data: dict,
) -> dict:

    #  STEP 1: Select rules
    if chunk_rules is not None:
        selected_rules = chunk_rules
    else:
        selected_rules = select_rules_for_chunk(
            rules, chunk, include_all=include_all_rules
        )

    # STEP 2: SECTION-BASED FILTERING (IMPORTANT FIX)
    section_type = chunk.get("section_type")

    filtered_rules = []
    for rule in selected_rules:
        applies = rule.get("applies_to_sections", [])
        #if section_type in applies:
        
        if (
            section_type in applies
            or "full" in applies
            or rule.get("always_include")
        ):

            filtered_rules.append(rule)

    selected_rules = filtered_rules
    
    #  DEBUG START
    print("\n-----------------------------")
    print("Chunk:", chunk["chunk_id"], "| Section:", chunk.get("section_type"))
    print("Selected Rules:", [r["rule_id"] for r in selected_rules])
    print("-----------------------------\n")
    #  DEBUG END

    
#  Create LLM-specific rules
    llm_rules = [
        rule for rule in selected_rules
        if rule["rule_id"] not in ["GDP-08", "GDP-12"]
    ]
    print("LLM Rules:", [r["rule_id"] for r in llm_rules])


    

    #  STEP 3: Default document content
    document_content = chunk.get("text", "")

    if chunk.get("chunk_id") != "whole_document":
        document_content = (
            f"Chunk ID: {chunk.get('chunk_id')}\n"
            f"Section Type: {section_type}\n"
            f"Heading: {chunk.get('heading')}\n"
            f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n\n"
            f"{chunk.get('text', '')}"
        )

    #  STEP 4: FOOTER AGGREGATION (GDP-11 FIX)
    # if section_type == "footer":

    #     footer_chunks = [
    #         c for c in chunks_data.get("chunks", [])
    #         if c.get("section_type") == "footer"
    #     ]

    #     footer_text = "\n".join(
    #         f"Page {c.get('page_start')}: {c.get('text', '')}"
    #         for c in footer_chunks
    #     )

    #     document_content = (
    #         "This section contains page numbers from all document pages. "
    #         "Check if page numbers are sequential.\n\n"
    #         f"{footer_text}"
    #     )
    if section_type == "footer":

        

        footer_chunks = [
            c for c in chunks_data.get("chunks", [])
            if c.get("section_type") == "footer"
        ]

        footer_text = "\n".join(
            f"Page {c.get('page_start')}: {c.get('text', '')}"
            for c in footer_chunks
        )

        document_content = (
            "This is the footer section aggregated across all pages.\n"
            "Evaluate the rules based ONLY on the footer content.\n\n"
            "IMPORTANT:\n"
            "- For GDP-11: Check page numbers are sequential.\n"
            "- For GDP-13: PASS if ANY ONE of the following is present:\n"
            "  • Document ID\n"
            "  • Page Number\n"
            "  • Confidentiality statement\n"
            "- FAIL GDP-13 ONLY if NONE of the above are present.\n\n"
            f"{footer_text}"
        )



    #  STEP 5: Load config
    rag_config = load_chunk_rag_config()

    # rules_for_prompt = (
    #     slim_rules_for_prompt(selected_rules)
    #     if rag_config.get("slim_rules_in_prompt", True) and chunk_rules is not None
    #     else selected_rules
    # )
    rules_for_prompt = (
    slim_rules_for_prompt(llm_rules)
    if rag_config.get("slim_rules_in_prompt", True) and chunk_rules is not None
    else llm_rules
)

    #  STEP 6: Build prompt
    prompt = load_prompt(
        prompt_path,
        FILE_NAME=file_name,
        CHUNK_ID=str(chunk.get("chunk_id", "")),
        #RULE_COUNT=str(len(selected_rules)),
        #RULE_IDS=", ".join(rule["rule_id"] for rule in selected_rules),
        RULE_COUNT=str(len(llm_rules)),
        RULE_IDS=", ".join(rule["rule_id"] for rule in llm_rules),
        RULES_JSON=json.dumps(rules_for_prompt, indent=2, ensure_ascii=False),
        DOCUMENT_CONTENT=document_content,
    )

    #  STEP 7: Call LLM
    normalize_results = rag_config.get("normalize_llm_results", True)
    retry_incomplete = rag_config.get("retry_incomplete_llm_response", False)

    response = call_llm_json(prompt, model=model, use_dummy=use_dummy_llm)
    results = response.get("results", [])
    normalization_stats = {}

    #  STEP 8: Normalize results
    if normalize_results and chunk_rules is not None:
        results, normalization_stats = normalize_chunk_results(
            results,
            #selected_rules,
            llm_rules,
            chunk_id=str(chunk.get("chunk_id", "")),
            section_type=str(section_type),
        )

        if retry_incomplete and normalization_stats.get("filled_missing_rule_ids"):
            retry_prompt = (
                f"{prompt}\n\n"
                f"Return exactly {len(selected_rules)} results for rules: "
                # f"{', '.join(rule['rule_id'] for rule in selected_rules)}."
                f"{', '.join(rule['rule_id'] for rule in llm_rules)}."

            )

            retry_response = call_llm_json(
                retry_prompt, model=model, use_dummy=use_dummy_llm
            )

            results, normalization_stats = normalize_chunk_results(
                retry_response.get("results", []),
                selected_rules,
                chunk_id=str(chunk.get("chunk_id", "")),
                section_type=str(section_type),
            )

    #  STEP 9: Attach metadata
    for item in results:
        item["chunk_id"] = chunk.get("chunk_id")
        item["section_type"] = section_type

    #  STEP 10: Build output
    output = {
        "chunk_id": chunk.get("chunk_id"),
        "section_type": section_type,
        "rules_evaluated": [rule["rule_id"] for rule in selected_rules],
        "results": results,
    }

    if normalization_stats:
        output["normalization"] = normalization_stats

    return output

# def evaluate_chunk(
#     chunk: dict,
#     rules: list[dict],
#     *,
#     file_name: str,
#     prompt_path: Path,
#     model: str | None,
#     include_all_rules: bool,
#     use_dummy_llm: bool = False,
#     chunk_rules: list[dict] | None = None,
#     chunks_data: dict,
# ) -> dict:

#     # STEP 1: Select rules
#     if chunk_rules is not None:
#         selected_rules = chunk_rules
#     else:
#         selected_rules = select_rules_for_chunk(
#             rules, chunk, include_all=include_all_rules
#         )

#     #  STEP 2: Section filtering
#     section_type = chunk.get("section_type")

#     filtered_rules = []
#     for rule in selected_rules:
#         applies = rule.get("applies_to_sections", [])
#         if section_type in applies:
#             filtered_rules.append(rule)

#     selected_rules = filtered_rules

#     #  STEP 2.5: Deterministic rules (GDP-08)
#     deterministic_results = []

#     for rule in selected_rules:
#         if rule["rule_id"] == "GDP-08":

#             fonts = set()
#             all_chunks = chunks_data.get("chunks", []) if chunks_data else []

#             for c in all_chunks:
#                 if c.get("section_type") not in ["body", "full"]:
#                     continue
#                 fonts.update(c.get("fonts", []))

#             #  FIX 1: HANDLE EMPTY FONT CASE
#             if not fonts:
#                 result = {
#                     "rule_id": "GDP-08",
#                     "status": "insufficient_evidence",
#                     "reason": "No font information available",
#                     "evidence": "",
#                     "confidence": 0.0,
#                 }

#             elif len(fonts) > 3:
#                 result = {
#                     "rule_id": "GDP-08",
#                     "status": "failed",
#                     "reason": "More than 3 fonts detected",
#                     "evidence": ", ".join(sorted(fonts)),
#                     "confidence": 0.95,
#                 }
#             else:
#                 result = {
#                     "rule_id": "GDP-08",
#                     "status": "passed",
#                     "reason": f"{len(fonts)} fonts used",
#                     "evidence": ", ".join(sorted(fonts)),
#                     "confidence": 0.95,
#                 }

#             result["chunk_id"] = chunk.get("chunk_id")
#             result["section_type"] = section_type

#             deterministic_results.append(result)

#     # REMOVE GDP-08 from LLM
#     selected_rules = [r for r in selected_rules if r["rule_id"] != "GDP-08"]

#     #  EARLY RETURN IF ONLY DETERMINISTIC RULE
#     if not selected_rules and deterministic_results:
#         for item in deterministic_results:
#             item["chunk_id"] = chunk.get("chunk_id")
#             item["section_type"] = section_type

#         return {
#             "chunk_id": chunk.get("chunk_id"),
#             "section_type": section_type,
#             "rules_evaluated": [r["rule_id"] for r in deterministic_results],
#             "results": deterministic_results,
#         }

#     #  STEP 3: Document content
#     document_content = chunk.get("text", "")

#     if chunk.get("chunk_id") != "whole_document":
#         document_content = (
#             f"Chunk ID: {chunk.get('chunk_id')}\n"
#             f"Section Type: {section_type}\n"
#             f"Heading: {chunk.get('heading')}\n"
#             f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}\n\n"
#             f"{chunk.get('text', '')}"
#         )

#     #  STEP 4: Footer aggregation
#     if section_type == "footer":
#         footer_chunks = [
#             c for c in chunks_data.get("chunks", [])
#             if c.get("section_type") == "footer"
#         ]

#         footer_text = "\n".join(
#             f"Page {c.get('page_start')}: {c.get('text', '')}"
#             for c in footer_chunks
#         )

#         document_content = (
#             "This section contains page numbers from all document pages.\n\n"
#             f"{footer_text}"
#         )

#     #  STEP 5: Config
#     rag_config = load_chunk_rag_config()

#     rules_for_prompt = (
#         slim_rules_for_prompt(selected_rules)
#         if rag_config.get("slim_rules_in_prompt", True) and chunk_rules is not None
#         else selected_rules
#     )

#     #  STEP 6: Prompt
#     prompt = load_prompt(
#         prompt_path,
#         FILE_NAME=file_name,
#         CHUNK_ID=str(chunk.get("chunk_id", "")),
#         RULE_COUNT=str(len(selected_rules)),
#         RULE_IDS=", ".join(rule["rule_id"] for rule in selected_rules),
#         RULES_JSON=json.dumps(rules_for_prompt, indent=2, ensure_ascii=False),
#         DOCUMENT_CONTENT=document_content,
#     )

#     #  STEP 7: LLM
#     results = []
#     normalization_stats = {}

#     if selected_rules:
#         response = call_llm_json(prompt, model=model, use_dummy=use_dummy_llm)
#         results = response.get("results", [])

#         normalize_results = rag_config.get("normalize_llm_results", True)
#         retry_incomplete = rag_config.get("retry_incomplete_llm_response", False)

#         if normalize_results and chunk_rules is not None:
#             results, normalization_stats = normalize_chunk_results(
#                 results,
#                 selected_rules,
#                 chunk_id=str(chunk.get("chunk_id", "")),
#                 section_type=str(section_type),
#             )

#     #  ADD deterministic results
#     results.extend(deterministic_results)

#     #  metadata
#     for item in results:
#         item["chunk_id"] = chunk.get("chunk_id")
#         item["section_type"] = section_type

#     #  FIX 2: REMOVE DUPLICATES
#     rules_evaluated = list({
#         r["rule_id"] for r in selected_rules
#     } | {
#         r["rule_id"] for r in deterministic_results
#     })

#     #  output
#     output = {
#         "chunk_id": chunk.get("chunk_id"),
#         "section_type": section_type,
#         "rules_evaluated": rules_evaluated,
#         "results": results,
#     }

#     if normalization_stats:
#         output["normalization"] = normalization_stats

#     return output
import textstat

def evaluate_font_consistency(chunks_data: dict) -> dict:
    fonts = chunks_data.get("document_fonts", [])
    unique_fonts = set(fonts)

    if not unique_fonts:
        return {
            "rule_id": "GDP-08",
            "status": "failed",
            "reason": "No font information detected.",
            "confidence": 0.9,
        }

    if len(unique_fonts) <= 3:
        return {
            "rule_id": "GDP-08",
            "status": "passed",
            "reason": f"{len(unique_fonts)} fonts used: {list(unique_fonts)}",
            "confidence": 0.95,
        }

    return {
        "rule_id": "GDP-08",
        "status": "failed",
        "reason": f"Too many fonts used: {list(unique_fonts)}",
        "confidence": 0.95,
    }


def evaluate_readability(chunks_data: dict) -> dict:
    text = " ".join(
        c.get("text", "") for c in chunks_data.get("chunks", [])
        if c.get("text")
    )

    if not text.strip():
        return {
            "rule_id": "GDP-12",
            "status": "insufficient_evidence",
            "reason": "No text available for readability.",
            "confidence": 0.0,
        }

    flesch = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    if flesch >= 25 and grade <= 14:
        status = "passed"
    else:
        status = "failed"

    return {
        "rule_id": "GDP-12",
        "status": status,
        "reason": f"Flesch: {flesch:.2f}, Grade: {grade:.2f}",
        "confidence": 0.9,
    }
# def merge_results(
#    rule_catalog: list[dict],
#    chunk_outputs: list[dict],
# ) -> list[dict]:
#    rule_map = {rule["rule_id"]: rule for rule in rule_catalog}
#    merged: dict[str, dict] = {}
#    for chunk_output in chunk_outputs:
#        allowed_rule_ids = set(chunk_output.get("rules_evaluated", []))
#        for item in chunk_output.get("results", []):
#            rule_id = item.get("rule_id")
#            if not rule_id or rule_id not in allowed_rule_ids:
#                continue
#            current = merged.get(rule_id)
#            incoming_status = item.get("status", "insufficient_evidence")
#            if not current:
#                merged[rule_id] = {
#                    **item,
#                    "title": rule_map.get(rule_id, {}).get("title", ""),
#                    "severity": rule_map.get(rule_id, {}).get("severity", ""),
#                    "recommendation": rule_map.get(rule_id, {}).get("recommendation", ""),
#                    "evidence_chunks": [item.get("chunk_id")],
#                }
#                continue
#            current_status = current.get("status", "insufficient_evidence")
#            if STATUS_PRIORITY.get(incoming_status, 0) > STATUS_PRIORITY.get(current_status, 0):
#                merged[rule_id].update(
#                    {
#                        "status": incoming_status,
#                        "reason": item.get("reason", current.get("reason", "")),
#                        "evidence": item.get("evidence", current.get("evidence", "")),
#                        "confidence": item.get("confidence", current.get("confidence", 0)),
#                    }
#                )
#            elif incoming_status == current_status and item.get("evidence"):
#                merged[rule_id]["evidence"] = (
#                    f"{current.get('evidence', '')}; {item.get('evidence', '')}".strip("; ")
#                )
#            chunk_id = item.get("chunk_id")
#            if chunk_id and chunk_id not in merged[rule_id]["evidence_chunks"]:
#                merged[rule_id]["evidence_chunks"].append(chunk_id)
#    ordered = []
#    for rule in rule_catalog:
#        rule_id = rule["rule_id"]
#        if rule_id in merged:
#            ordered.append(merged[rule_id])
#        else:
#            ordered.append(
#                {
#                    "rule_id": rule_id,
#                    "title": rule.get("title", ""),
#                    "status": "insufficient_evidence",
#                    "reason": "No chunk provided enough evidence to evaluate this rule.",
#                    "evidence": "",
#                    "confidence": 0.0,
#                    "severity": rule.get("severity", ""),
#                    "recommendation": rule.get("recommendation", ""),
#                    "evidence_chunks": [],
#                }
#            )
#    return ordered

def merge_results(
    rule_catalog: list[dict],
    chunk_outputs: list[dict],
) -> list[dict]:

    rule_map = {rule["rule_id"]: rule for rule in rule_catalog}
    merged: dict[str, dict] = {}

    def merge_rule(existing, incoming):
        if existing is None:
            return incoming

        existing_status = existing.get("status", "insufficient_evidence")
        incoming_status = incoming.get("status", "insufficient_evidence")

        # NEVER override PASS
        if existing_status == "passed":
            return existing

        if incoming_status == "passed":
            return {
                **existing,
                **incoming,
                "evidence_chunks": list(
                    set(existing.get("evidence_chunks", []) + [incoming.get("chunk_id")])
                ),
            }

        #  use GLOBAL STATUS_PRIORITY here
        if STATUS_PRIORITY.get(incoming_status, 0) > STATUS_PRIORITY.get(existing_status, 0):
            return {
                **existing,
                **incoming,
                "evidence_chunks": list(
                    set(existing.get("evidence_chunks", []) + [incoming.get("chunk_id")])
                ),
            }

        return existing

    #  MERGE LOOP
    for chunk_output in chunk_outputs:
        allowed_rule_ids = set(chunk_output.get("rules_evaluated", []))

        for item in chunk_output.get("results", []):
            rule_id = item.get("rule_id")

            if not rule_id or rule_id not in allowed_rule_ids:
                continue

            current = merged.get(rule_id)

            incoming_item = {
                **item,
                "title": rule_map.get(rule_id, {}).get("title", ""),
                "severity": rule_map.get(rule_id, {}).get("severity", ""),
                "recommendation": rule_map.get(rule_id, {}).get("recommendation", ""),
                "evidence_chunks": [item.get("chunk_id")],
            }

            merged[rule_id] = merge_rule(current, incoming_item)

    #  FINAL ORDERING
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
) -> dict:
   resolved_file_name = file_name or extracted.get("file_name", "unknown")
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
   summary = summarize(merged_results)
   return {
       "source_path": extracted.get("source_path"),
       "file_name": resolved_file_name,
       "mode": "whole_doc",
       "rule_retrieval": "all",
       "summary": summary,
       "chunk_outputs": [chunk_output],
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
) -> dict:
   resolved_file_name = file_name or chunks_data.get("file_name", "unknown")
   match_map = {
       match["chunk_id"]: match["matched_rules"]
       for match in chunk_rule_matches.get("matches", [])
   }
   chunk_outputs = []
   for chunk in chunks_data.get("chunks", []):
       chunk_id = chunk.get("chunk_id")
       if chunk_id not in match_map:
           continue
       chunk_outputs.append(
           evaluate_chunk(
               chunk,
               rules,
               file_name=resolved_file_name,
               prompt_path=prompt_path,
               model=model,
               include_all_rules=True,
               use_dummy_llm=use_dummy_llm,
               chunk_rules=match_map[chunk_id],
               chunks_data=chunks_data,
           )
       )
   #merged_results = merge_results(rules, chunk_outputs)
   #summary = summarize(merged_results)
   merged_results = merge_results(rules, chunk_outputs)

    #  GDP-08 override
   font_result = evaluate_font_consistency(chunks_data)

    #  GDP-12 override
   readability_result = evaluate_readability(chunks_data)

   for i, r in enumerate(merged_results):
        if r["rule_id"] == "GDP-08":
            merged_results[i] = {**r, **font_result}
        if r["rule_id"] == "GDP-12":
            merged_results[i] = {**r, **readability_result}

   summary = summarize(merged_results)

   return {
       "source_path": chunks_data.get("source_path"),
       "file_name": resolved_file_name,
       "mode": "chunk_rag",
       "rule_retrieval": "rag",
       "chunk_rule_matches": chunk_rule_matches,
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
                   chunks_data=chunks_data
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