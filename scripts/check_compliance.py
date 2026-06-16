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
   response = call_llm_json(prompt, model=model, use_dummy=use_dummy_llm)
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
           retry_response = call_llm_json(retry_prompt, model=model, use_dummy=use_dummy_llm)
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

def merge_results(
   rule_catalog: list[dict],
   chunk_outputs: list[dict],
) -> list[dict]:
   rule_map = {rule["rule_id"]: rule for rule in rule_catalog}
   merged: dict[str, dict] = {}
   for chunk_output in chunk_outputs:
       allowed_rule_ids = set(chunk_output.get("rules_evaluated", []))
       for item in chunk_output.get("results", []):
           rule_id = item.get("rule_id")
           if not rule_id or rule_id not in allowed_rule_ids:
               continue
           current = merged.get(rule_id)
           incoming_status = item.get("status", "insufficient_evidence")
           if not current:
               merged[rule_id] = {
                   **item,
                   "title": rule_map.get(rule_id, {}).get("title", ""),
                   "severity": rule_map.get(rule_id, {}).get("severity", ""),
                   "recommendation": rule_map.get(rule_id, {}).get("recommendation", ""),
                   "evidence_chunks": [item.get("chunk_id")],
               }
               continue
           current_status = current.get("status", "insufficient_evidence")
           if STATUS_PRIORITY.get(incoming_status, 0) > STATUS_PRIORITY.get(current_status, 0):
               merged[rule_id].update(
                   {
                       "status": incoming_status,
                       "reason": item.get("reason", current.get("reason", "")),
                       "evidence": item.get("evidence", current.get("evidence", "")),
                       "confidence": item.get("confidence", current.get("confidence", 0)),
                   }
               )
           elif incoming_status == current_status and item.get("evidence"):
               merged[rule_id]["evidence"] = (
                   f"{current.get('evidence', '')}; {item.get('evidence', '')}".strip("; ")
               )
           chunk_id = item.get("chunk_id")
           if chunk_id and chunk_id not in merged[rule_id]["evidence_chunks"]:
               merged[rule_id]["evidence_chunks"].append(chunk_id)
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
           )
       )
   merged_results = merge_results(rules, chunk_outputs)
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