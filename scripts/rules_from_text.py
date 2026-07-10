"""Convert plain-language rules text into structured rules JSON (Model Garden LLM).

Usage:
  python scripts/rules_from_text.py rules/rules_plain.txt
  python scripts/rules_from_text.py --text "GDP-01: Title must match file name."
  python scripts/rules_from_text.py -o output/rules/my_rules.json rules/rules_plain.txt
  type rules\\rules_plain.txt | python scripts/rules_from_text.py -
  python scripts/rules_from_text.py rules/rules_plain.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import DEFAULT_MODEL, LLMResponseValidationError, call_llm_json, load_prompt,call_llm
from src.llm_schemas import RulesGenerationResponse
from src.output_utils import run_timestamp

DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "rules_to_json.txt"
DEFAULT_INPUT = PROJECT_ROOT / "rules" / "rules_plain.txt"


def rules_from_text(
    rules_text: str,
    *,
    prompt_path: Path,
    model: str | None = None,
) -> dict:
    prompt = load_prompt(prompt_path, RULES_TEXT=rules_text.strip())
    try:
        result = call_llm(prompt, model=model, response_model=RulesGenerationResponse)
    except LLMResponseValidationError as error:
        raise ValueError(f"LLM response failed schema validation: {error}") from error
    return result


def read_rules_input(
    *,
    input_path: Path | None,
    inline_text: str | None,
) -> tuple[str, str]:
    if inline_text is not None:
        if not inline_text.strip():
            raise ValueError("Rules text is empty.")
        return inline_text, "<inline --text>"

    if input_path is None:
        input_path = DEFAULT_INPUT

    if str(input_path) == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("No rules text received on stdin.")
        return text, "stdin"

    if not input_path.exists():
        raise FileNotFoundError(f"Rules text file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Rules text file is empty: {input_path}")
    return text, str(input_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert plain-language compliance rules into structured JSON using Model Garden LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/rules_from_text.py rules/rules_plain.txt\n"
            '  python scripts/rules_from_text.py --text "GDP-01: Title must match file name."\n'
            "  python scripts/rules_from_text.py -o output/rules/my_rules.json rules/rules_plain.txt\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to plain-text rules file, or '-' to read from stdin "
        f"(default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--text",
        help="Rules text directly on the command line (instead of a file)",
    )
    parser.add_argument(
        "--rules-text",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT),
        help=f"Prompt template path (default: {DEFAULT_PROMPT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: output/rules/generated_rules_<timestamp>.json)",
    )
    parser.add_argument(
        "--model",
        help="Override LLM model (default: LLM_MODEL from .env)",
    )
    args = parser.parse_args()

    legacy_path = Path(args.rules_text) if args.rules_text else None
    input_path = legacy_path
    if args.input:
        input_path = Path(args.input)

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    try:
        rules_text, source_label = read_rules_input(
            input_path=input_path,
            inline_text=args.text,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    model = args.model or DEFAULT_MODEL
    run_ts = run_timestamp()
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "output" / "rules" / f"generated_rules_{run_ts}.json"
    )

    print(f"LLM:    {model}")
    print(f"Source: {source_label}")
    print("Calling Model Garden...")

    try:
        result = rules_from_text(rules_text, prompt_path=prompt_path, model=model)
    except RuntimeError as error:
        raise SystemExit(
            f"Model Garden call failed: {error}\n"
            "Check MODEL_GARDEN_BASE_URL, MODEL_GARDEN_API_KEY, and LLM_MODEL in .env"
        ) from error

    result["run_timestamp"] = run_ts
    result["source"] = source_label
    result["llm_model"] = model

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {len(result['rules'])} rule(s)")
    print(f"Saved:   {output_path}")


if __name__ == "__main__":
    main()
