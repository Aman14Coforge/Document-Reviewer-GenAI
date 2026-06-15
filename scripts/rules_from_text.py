"""Convert plain-language rules text into structured rules JSON using an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import call_llm_json, load_prompt
from src.output_utils import run_timestamp


def rules_from_text(
    rules_text: str,
    *,
    prompt_path: Path,
    model: str | None = None,
) -> dict:
    prompt = load_prompt(prompt_path, RULES_TEXT=rules_text.strip())
    result = call_llm_json(prompt, model=model)
    if "rules" not in result or not isinstance(result["rules"], list):
        raise ValueError("LLM response must contain a 'rules' array.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert plain-language compliance rules into structured JSON."
    )
    parser.add_argument(
        "--rules-text",
        help="Path to plain text rules file (default: rules/rules_plain.txt)",
    )
    parser.add_argument(
        "--prompt",
        help="Path to prompt template (default: prompts/rules_to_json.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: output/rules/generated_rules_<timestamp>.json)",
    )
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    args = parser.parse_args()

    rules_text_path = Path(args.rules_text or PROJECT_ROOT / "rules" / "rules_plain.txt")
    prompt_path = Path(args.prompt or PROJECT_ROOT / "prompts" / "rules_to_json.txt")
    run_ts = run_timestamp()
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "output" / "rules" / f"generated_rules_{run_ts}.json"
    )

    if not rules_text_path.exists():
        raise SystemExit(f"Rules text file not found: {rules_text_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    rules_text = rules_text_path.read_text(encoding="utf-8")
    result = rules_from_text(rules_text, prompt_path=prompt_path, model=args.model)
    result["run_timestamp"] = run_ts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {len(result['rules'])} rule(s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
