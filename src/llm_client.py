"""Shared LLM client for document compliance scripts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.logging_config import get_logger

logger = get_logger("llm")


class LLMResponseValidationError(ValueError):
    """Raised when LLM JSON does not match the expected Pydantic schema."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

BASE_URL = os.getenv("MODEL_GARDEN_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).strip()
API_KEY = os.getenv("MODEL_GARDEN_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "")).strip()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client

    if _client is not None:
        return _client

    if not BASE_URL:
        raise RuntimeError(
            "MODEL_GARDEN_BASE_URL is not set. Copy .env.example to .env and configure it."
        )
    if not API_KEY:
        raise RuntimeError(
            "MODEL_GARDEN_API_KEY is not set. Copy .env.example to .env and configure it."
        )
    if not DEFAULT_MODEL:
        raise RuntimeError(
            "LLM_MODEL is not set. Copy .env.example to .env and configure it."
        )

    custom_http_client = httpx.Client(
        headers={
            "X-API-KEY": API_KEY,
            "Content-Type": "application/json",
        }
    )
    _client = OpenAI(
        base_url=BASE_URL,
        api_key="dummy",
        http_client=custom_http_client,
    )
    return _client


def load_prompt(path: str | Path, **replacements: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_llm_response(data: dict, response_model: type[BaseModel]) -> dict:
    try:
        parsed = response_model.model_validate(data)
    except ValidationError as error:
        logger.warning("LLM response failed schema validation: %s", error)
        raise LLMResponseValidationError(str(error)) from error
    return parsed.model_dump()


def call_llm(
    prompt: str,
    *,
    system_prompt: str = "You return strict JSON when asked. Follow instructions exactly.",
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty response.")
    return content.strip()


def dummy_call_llm_json(prompt: str) -> dict:
    """
    Dummy LLM for local/UI testing without Model Garden credentials.

    Uses keyword heuristics per rule; rules not in the hardcoded map auto-pass
    when chunk text is >= 80 chars. Not suitable for production sign-off.
    """
    rules = _extract_rules_from_prompt(prompt)
    document_text = _extract_document_from_prompt(prompt)
    file_name = _extract_file_name_from_prompt(prompt)
    logger.debug("Dummy LLM evaluating %s rule(s), text_len=%s", len(rules), len(document_text))

    results = []
    for rule in rules:
        rule_id = rule.get("rule_id", "UNKNOWN")
        status, reason, evidence, confidence = _dummy_rule_verdict(rule, document_text, file_name)
        results.append(
            {
                "rule_id": rule_id,
                "status": status,
                "reason": reason,
                "evidence": evidence,
                "confidence": confidence,
            }
        )
    return {"results": results}


def _extract_file_name_from_prompt(prompt: str) -> str:
    match = re.search(r"FILE NAME:\s*(.+)", prompt)
    return match.group(1).strip() if match else ""


def _extract_document_from_prompt(prompt: str) -> str:
    for label in ("CHUNK CONTENT:", "DOCUMENT CONTENT:"):
        match = re.search(rf"{re.escape(label)}\s*(.*)\Z", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _extract_rules_from_prompt(prompt: str) -> list[dict]:
    marker_match = re.search(
        r"(?:MATCHED RULES(?: FOR THIS CHUNK|\s*\(vector retrieval[^\)]*\))?|RULES TO EVALUATE)\s*:",
        prompt,
        re.IGNORECASE,
    )
    if not marker_match:
        return []

    rules_text = prompt[marker_match.end() :]
    for sep in ("CHUNK CONTENT:", "DOCUMENT CONTENT:"):
        if sep in rules_text:
            rules_text = rules_text.split(sep, maxsplit=1)[0]
            break
    rules_text = rules_text.strip()
    try:
        parsed = json.loads(rules_text)
        return parsed if isinstance(parsed, list) else parsed.get("rules", [])
    except json.JSONDecodeError:
        return []


def _dummy_rule_verdict(rule: dict, document_text: str, file_name: str) -> tuple[str, str, str, float]:
    rule_id = rule.get("rule_id", "")
    text_lower = document_text.lower()
    file_stem = Path(file_name).stem.lower().replace("_", " ").replace("-", " ")

    checks = {
        "GDP-01": (
            "deployment report" in text_lower and ("deployment report" in file_stem or "report" in file_stem),
            "Title appears aligned with file name.",
            "Deployment report title found on first page.",
        ),
        "GDP-02": (
            any(token in text_lower for token in ["author", "prepared by", "manager", "director"]),
            "Author or role information appears present.",
            "Author/role keywords found in document.",
        ),
        "GDP-05": ("revision history" in text_lower, "Revision history section found.", "Revision history heading present."),
        "GDP-06": (
            any(token in text_lower for token in ["signature", "approved by", "approver"]),
            "Signature or approval block found.",
            "Approval/signature keywords found.",
        ),
        "GDP-10": (
            all(token in text_lower for token in ["introduction", "scope", "references"])
            or "responsibilities" in text_lower,
            "Core structural sections appear present.",
            "Major section headings found.",
        ),
        "GDP-13": (
            any(token in text_lower for token in ["confidential", "page", "doc id", "document id"]),
            "Footer-related identifiers appear present.",
            "Footer keywords found in extracted text.",
        ),
    }

    if rule_id in checks:
        passed, reason_ok, evidence = checks[rule_id]
        if passed:
            return "passed", reason_ok, evidence, 0.88
        return "failed", f"{rule.get('title', rule_id)} was not clearly satisfied.", evidence, 0.72

    if len(document_text.strip()) < 80:
        return (
            "insufficient_evidence",
            "Not enough document content in this chunk to evaluate the rule.",
            "",
            0.4,
        )

    return (
        "passed",
        f"Dummy engine: no blocking issue detected for {rule_id}.",
        document_text[:120],
        0.75,
    )


def call_llm_json(
    prompt: str,
    *,
    system_prompt: str = "You return strict JSON when asked. Follow instructions exactly.",
    model: str | None = None,
    temperature: float = 0.0,
    use_dummy: bool = False,
    response_model: type[BaseModel] | None = None,
) -> dict:
    if use_dummy:
        logger.debug("Using dummy LLM engine")
        data = dummy_call_llm_json(prompt)
    else:
        logger.debug("Calling Model Garden LLM model=%s", model or DEFAULT_MODEL)
        raw = call_llm(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )
        data = extract_json(raw)

    if response_model is not None:
        return validate_llm_response(data, response_model)
    return data
