"""Match chunk JSON to rules using Chroma vector retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from check_compliance import load_rules
from src.output_utils import build_output_path, run_timestamp
from src.rag.chunk_matcher import match_rules_to_chunks, save_chunk_rule_matches

DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Match chunks to rules using Chroma retrieval.")
    parser.add_argument("chunks_json", help="Path to chunk JSON")
    parser.add_argument("--rules", help="Path to rules JSON (default: rules/rules.json)")
    parser.add_argument("--top-k", type=int, help="Override top-k rules per chunk")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: output/matches/<stem>_<timestamp>_chunk_rules.json)",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks_json)
    rules_path = Path(args.rules or DEFAULT_RULES_PATH)
    if not chunks_path.exists():
        raise SystemExit(f"Chunk file not found: {chunks_path}")
    if not rules_path.exists():
        raise SystemExit(f"Rules file not found: {rules_path}")

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    rules = load_rules(rules_path)

    matches = match_rules_to_chunks(
        chunks_data,
        rules,
        top_k=args.top_k,
    )

    run_ts = run_timestamp()
    stem = chunks_data.get("file_stem") or chunks_path.stem.replace("_chunks", "").rsplit("_", 2)[0]
    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(PROJECT_ROOT / "output" / "matches", stem, "chunk_rules", run_ts)
    )
    matches["run_timestamp"] = run_ts
    save_chunk_rule_matches(matches, output_path)

    print(f"Matched {len(matches['matches'])} chunk(s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
