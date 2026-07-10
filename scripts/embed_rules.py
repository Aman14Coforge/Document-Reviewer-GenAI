"""Embed GDP rules into ChromaDB (local MiniLM or Model Garden API)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rule_store import embed_rules, get_embedding_provider

DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed rules from rules.json into ChromaDB (local or Model Garden)."
    )
    parser.add_argument(
        "--rules",
        help="Path to rules JSON (default: rules/rules.json)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before embedding",
    )
    args = parser.parse_args()

    rules_path = Path(args.rules or DEFAULT_RULES_PATH)
    if not rules_path.exists():
        raise SystemExit(f"Rules file not found: {rules_path}")

    print("Embedding rules into ChromaDB...")
    print(f"Provider:        {get_embedding_provider()}")
    result = embed_rules(rules_path, rebuild=args.rebuild)

    summary_path = PROJECT_ROOT / "data" / "chroma" / "last_embed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Provider:        {result['embedding_provider']}")
    print(f"Collection:      {result['collection']}")
    print(f"Embedding model: {result['embedding_model']}")
    print(f"Model source:    {result['embedding_model_source']}")
    print(f"Embedding dim:   {result['embedding_dimension']}")
    if result.get("embedding_model_path"):
        print(f"Local path:      {result['embedding_model_path']}")
    print(f"Rules embedded:  {result['rule_count']}")
    print(f"Chroma path:     {result['chroma_path']}")
    print(f"Summary saved:   {summary_path}")


if __name__ == "__main__":
    main()
