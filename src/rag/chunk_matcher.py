"""Match document chunks to GDP rules using Chroma vector retrieval."""
from __future__ import annotations
#from src.rag.reranker import rerank_rules
import json
from pathlib import Path
from src.rag.rule_store import load_compliance_config, resolve_rules_by_ids, retrieve_rule_ids_for_text
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "compliance.json"

def load_chunk_rag_config() -> dict:
   config = load_compliance_config()
   return config.get(
       "chunk_rag",
       {
           "top_k_rules_per_chunk": 2,
           "use_vector_retrieval_only": True,
           "always_include_rule_ids": [],
           "skip_chunk_ids": ["full_document"],
           "fallback_to_all_rules_if_empty": False,
       },
   )

def enrich_query(text: str, section_type: str) -> str:
    if section_type == "first_page":
        return f"This is the first page containing document title and version. {text}"

    if section_type == "revision_history":
        return f"This section contains revision history with versions and dates. {text}"

    if section_type == "approval":
        return f"This section contains approval names, signatures and dates. {text}"

    if section_type == "structure":
        return f"This section represents document structure like table of contents. {text}"

    return text


def keyword_match_rules(text: str, all_rules: list[dict]) -> list[str]:
    matched = []
    text_lower = text.lower()

    for rule in all_rules:
        #  1. Keyword-based matching
        for keyword in rule.get("keywords", []):
            if keyword.lower() in text_lower:
                matched.append(rule["rule_id"])
                break  # stop after first match

        #  2. Typical phrase matching (strong signal)
        for phrase in rule.get("typical_phrases", []):
            if phrase.lower() in text_lower:
                matched.append(rule["rule_id"])
                break

    return list(set(matched))
# def match_rules_to_chunks(
#    chunks_data: dict,
#    all_rules: list[dict],
#    *,
#    top_k: int | None = None,
#    always_include_rule_ids: list[str] | None = None,
#    skip_chunk_ids: list[str] | None = None,
# ) -> dict:
#    rag_config = load_chunk_rag_config()
#    top_k = top_k or rag_config.get("top_k_rules_per_chunk", 2)
#    vector_only = rag_config.get("use_vector_retrieval_only", True)
#    always_include_rule_ids = (
#        [] if vector_only else (always_include_rule_ids or rag_config.get("always_include_rule_ids", []))
#    )
#    section_always_include = {} if vector_only else rag_config.get("section_always_include", {})
#    skip_chunk_ids = set(skip_chunk_ids or rag_config.get("skip_chunk_ids", ["full_document"]))
#    fallback_all = False if vector_only else rag_config.get("fallback_to_all_rules_if_empty", False)
#    rule_map = {rule["rule_id"]: rule for rule in all_rules}
#    always_rules = [rule_map[rule_id] for rule_id in always_include_rule_ids if rule_id in rule_map]
#    matches = []
#    for chunk in chunks_data.get("chunks", []):
#        chunk_id = chunk.get("chunk_id", "")
#        if chunk_id in skip_chunk_ids:
#            continue
#        retrieved_ids = retrieve_rule_ids_for_text(chunk.get("text", ""), top_k=top_k)
#        matched_rules = resolve_rules_by_ids(retrieved_ids, all_rules)
       
#        if not vector_only:
#            for rule in always_rules:
#                if rule["rule_id"] not in {item["rule_id"] for item in matched_rules}:
#                    matched_rules.append(rule)
#            section_type = chunk.get("section_type", "")
#            for rule_id in section_always_include.get(section_type, []):
#                if rule_id in rule_map and rule_id not in {item["rule_id"] for item in matched_rules}:
#                    matched_rules.append(rule_map[rule_id])
#        if not matched_rules and fallback_all:
#            matched_rules = list(all_rules)
#        matches.append(
#            {
#                "chunk_id": chunk_id,
#                "section_type": chunk.get("section_type"),
#                "heading": chunk.get("heading"),
#                "page_start": chunk.get("page_start"),
#                "page_end": chunk.get("page_end"),
#                "retrieved_rule_ids": retrieved_ids,
#                "matched_rule_ids": [rule["rule_id"] for rule in matched_rules],
#                #"matched_rule_ids": final_ids,
#                "matched_rules": matched_rules,
               
#            }
#        )
#    return {
#        "source_path": chunks_data.get("source_path"),
#        "file_name": chunks_data.get("file_name"),
#        "file_stem": chunks_data.get("file_stem"),
#        "top_k": top_k,
#        "use_vector_retrieval_only": vector_only,
#        "always_include_rule_ids": always_include_rule_ids,
#        "section_always_include": section_always_include,
#        "matches": matches,
#    }

def match_rules_to_chunks(
   chunks_data: dict,
   all_rules: list[dict],
   *,
   top_k: int | None = None,
   always_include_rule_ids: list[str] | None = None,
   skip_chunk_ids: list[str] | None = None,
) -> dict:
   rag_config = load_chunk_rag_config()
   top_k = top_k or rag_config.get("top_k_rules_per_chunk", 2)
   vector_only = rag_config.get("use_vector_retrieval_only", True)
   always_include_rule_ids = (
       [] if vector_only else (always_include_rule_ids or rag_config.get("always_include_rule_ids", []))
   )
   section_always_include = {} if vector_only else rag_config.get("section_always_include", {})
   skip_chunk_ids = set(skip_chunk_ids or rag_config.get("skip_chunk_ids", ["full_document"]))
   fallback_all = False if vector_only else rag_config.get("fallback_to_all_rules_if_empty", False)

   rule_map = {rule["rule_id"]: rule for rule in all_rules}
   always_rules = [rule_map[rule_id] for rule_id in always_include_rule_ids if rule_id in rule_map]

   matches = []

   for chunk in chunks_data.get("chunks", []):
       chunk_id = chunk.get("chunk_id", "")
       if chunk_id in skip_chunk_ids:
           continue

       section_type = chunk.get("section_type", "")
       text = chunk.get("text", "")

       #  1. FOOTER HANDLING (deterministic rules)
       if section_type == "footer":
           retrieved_ids = ["GDP-11", "GDP-13"]
           matched_rules = resolve_rules_by_ids(retrieved_ids, all_rules)

       else:
           #  2. Query enrichment
           enriched_text = enrich_query(text, section_type)

           #  3. Semantic retrieval
           retrieved_ids = retrieve_rule_ids_for_text(enriched_text, top_k=top_k)

           
           retrieved_ids = [
                rid for rid in retrieved_ids
                if rid not in ["GDP-11", "GDP-13"]
            ]


           #  4. Section filtering
           candidate_rule_ids = {
               r["rule_id"]
               for r in all_rules
               if section_type in r.get("applies_to_sections", [])
               or "full" in r.get("applies_to_sections", [])
           }

           retrieved_ids = [
               rid for rid in retrieved_ids
               if rid in candidate_rule_ids
           ]

           #  5. Keyword matching (hybrid enhancement)
           keyword_ids = keyword_match_rules(text, all_rules)
           keyword_ids = [
                rid for rid in keyword_match_rules(text, all_rules)
                if rid not in ["GDP-11", "GDP-13"]
            ]

           retrieved_ids = list(set(retrieved_ids + keyword_ids))

           #  Final resolve
           matched_rules = resolve_rules_by_ids(retrieved_ids, all_rules)

       #  Existing logic
       if not vector_only:
           for rule in always_rules:
               if rule["rule_id"] not in {item["rule_id"] for item in matched_rules}:
                   matched_rules.append(rule)

           for rule_id in section_always_include.get(section_type, []):
               if rule_id in rule_map and rule_id not in {item["rule_id"] for item in matched_rules}:
                   matched_rules.append(rule_map[rule_id])

       if not matched_rules and fallback_all:
           matched_rules = list(all_rules)

       matches.append(
           {
               "chunk_id": chunk_id,
               "section_type": section_type,
               "heading": chunk.get("heading"),
               "page_start": chunk.get("page_start"),
               "page_end": chunk.get("page_end"),
               "retrieved_rule_ids": retrieved_ids,
               "matched_rule_ids": [rule["rule_id"] for rule in matched_rules],
               "matched_rules": matched_rules,
           }
       )

   return {
       "source_path": chunks_data.get("source_path"),
       "file_name": chunks_data.get("file_name"),
       "file_stem": chunks_data.get("file_stem"),
       "top_k": top_k,
       "use_vector_retrieval_only": vector_only,
       "always_include_rule_ids": always_include_rule_ids,
       "section_always_include": section_always_include,
       "matches": matches,
   }

def save_chunk_rule_matches(matches: dict, output_path: Path) -> None:
   output_path.parent.mkdir(parents=True, exist_ok=True)
   output_path.write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")