"""Python logic checks for deterministic compliance rules.

Each rule has a dedicated checker (or generic keyword fallback) that inspects
the full extracted document text — no LLM involved. Used in hybrid mode alongside
semantic (LLM) and existential (external registry) checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.logging_config import get_logger

logger = get_logger("deterministic")

DEFAULT_REQUIRED_SECTIONS = [
    "INTRODUCTION",
    "REVISION HISTORY",
    "REFERENCES",
]

PLACEHOLDER_PATTERNS = [
    re.compile(r"<[^>]{1,120}>"),
    re.compile(r"\bXX\b"),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bN/A\b"),
    re.compile(r"_+\s*$"),
]

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
]

PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,4}$")
DOC_ID_PATTERN = re.compile(
    r"\b(?:doc(?:ument)?\s*id|doc\s*#|ref(?:erence)?\s*(?:no|number)?)\s*[:#]?\s*[\w-]+",
    re.IGNORECASE,
)


def _result(
    rule: dict,
    *,
    status: str,
    reason: str,
    evidence: str = "",
    confidence: float = 1.0,
) -> dict:
    return {
        "rule_id": rule["rule_id"],
        "status": status,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "check_method": "deterministic",
        "chunk_id": "whole_document",
        "section_type": "full",
    }


def _full_text(extracted: dict) -> str:
    return extracted.get("full_text", "") or ""


def _pages(extracted: dict) -> list[dict]:
    return extracted.get("pages", []) or []


def _search_terms(rule: dict) -> list[str]:
    terms: list[str] = []
    for key in ("keywords", "typical_phrases"):
        values = rule.get(key) or []
        if isinstance(values, list):
            terms.extend(str(item) for item in values if item)
    return terms


def _contains_any(text: str, terms: list[str]) -> tuple[bool, str]:
    upper = text.upper()
    for term in terms:
        if term.upper() in upper:
            return True, term
    return False, ""


def check_gdp_08(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    pages = _pages(extracted)
    if not pages:
        return _result(rule, status="insufficient_evidence", reason="No page text available.")

    line_lengths: list[int] = []
    blank_runs = 0
    for page in pages:
        page_blank = 0
        for line in page.get("text", "").splitlines():
            stripped = line.strip()
            if not stripped:
                page_blank += 1
                continue
            line_lengths.append(len(stripped))
        if page_blank >= 3:
            blank_runs += 1

    if not line_lengths:
        return _result(rule, status="insufficient_evidence", reason="No readable lines found.")

    avg_len = sum(line_lengths) / len(line_lengths)
    outliers = [length for length in line_lengths if length < avg_len * 0.15 or length > avg_len * 3.5]
    issues = []
    if len(outliers) > len(line_lengths) * 0.25:
        issues.append("high line-length variance")
    if blank_runs > max(1, len(pages) // 3):
        issues.append("excessive blank lines")

    if issues:
        return _result(
            rule,
            status="failed",
            reason=f"Formatting inconsistency detected: {', '.join(issues)}.",
            evidence=f"Analyzed {len(line_lengths)} lines across {len(pages)} pages.",
            confidence=0.85,
        )
    return _result(
        rule,
        status="passed",
        reason="Line length and spacing appear consistent across pages.",
        evidence=f"Analyzed {len(line_lengths)} lines across {len(pages)} pages.",
        confidence=0.8,
    )


def check_gdp_09(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    text = _full_text(extracted)
    if not text.strip():
        return _result(rule, status="insufficient_evidence", reason="No document text available.")

    issues: list[str] = []
    if re.search(r"  +", text):
        issues.append("double spaces")
    if re.search(r"\b(\w+)\s+\1\b", text, re.IGNORECASE):
        issues.append("repeated words")
    if re.search(r"[a-z][A-Z]", text.replace("\n", " ")):
        issues.append("missing space before capitalized word")
    if re.search(r"\bteh\b|\brecieve\b|\boccured\b", text, re.IGNORECASE):
        issues.append("common misspellings")

    if issues:
        return _result(
            rule,
            status="failed",
            reason=f"Potential language issues: {', '.join(issues)}.",
            evidence="; ".join(issues),
            confidence=0.7,
        )
    return _result(
        rule,
        status="passed",
        reason="No obvious grammar or spelling patterns detected by heuristic checks.",
        confidence=0.65,
    )


def check_gdp_10(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    text = _full_text(extracted).upper()
    required = DEFAULT_REQUIRED_SECTIONS.copy()
    for term in rule.get("typical_phrases") or []:
        upper = str(term).upper()
        if upper and upper not in required:
            required.append(upper)

    missing = [section for section in required if section not in text]
    found = [section for section in required if section in text]

    if missing:
        return _result(
            rule,
            status="failed",
            reason=f"Missing required section heading(s): {', '.join(missing)}.",
            evidence=f"Found: {', '.join(found) or 'none'}",
            confidence=0.9,
        )
    return _result(
        rule,
        status="passed",
        reason="All required section headings were found.",
        evidence=f"Found: {', '.join(found)}",
        confidence=0.9,
    )


def check_gdp_11(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    pages = _pages(extracted)
    if not pages:
        return _result(rule, status="insufficient_evidence", reason="No pages available.")

    found_numbers: list[int] = []
    for page in pages:
        lines = [line.strip() for line in page.get("text", "").splitlines() if line.strip()]
        if not lines:
            continue
        last = lines[-1]
        if PAGE_NUMBER_PATTERN.match(last):
            found_numbers.append(int(last))

    expected = list(range(1, len(pages) + 1))
    if len(found_numbers) < max(2, len(pages) // 2):
        return _result(
            rule,
            status="failed",
            reason="Page numbers were not detected on most pages.",
            evidence=f"Detected trailing numbers: {found_numbers}",
            confidence=0.85,
        )

    if found_numbers != expected[: len(found_numbers)] and sorted(found_numbers) != found_numbers:
        return _result(
            rule,
            status="failed",
            reason="Page numbers are missing or not sequential.",
            evidence=f"Detected: {found_numbers}; expected 1..{len(pages)}",
            confidence=0.9,
        )

    return _result(
        rule,
        status="passed",
        reason="Page numbers appear present and sequential.",
        evidence=f"Detected: {found_numbers}",
        confidence=0.85,
    )


def check_gdp_13(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    text = _full_text(extracted)
    upper = text.upper()
    terms = _search_terms(rule) or ["CONFIDENTIAL", "DOC ID", "PAGE"]

    hits = [term for term in terms if term.upper() in upper]
    has_confidential = "CONFIDENTIAL" in upper
    has_page_ref = bool(re.search(r"\bpage\s+\d+\b", text, re.IGNORECASE) or PAGE_NUMBER_PATTERN.search(text))
    has_doc_id = bool(DOC_ID_PATTERN.search(text) or re.search(r"\bSOP-\d+\b", text, re.IGNORECASE))

    missing = []
    if not has_confidential and "CONFIDENTIAL" in [t.upper() for t in terms]:
        missing.append("confidentiality marker")
    if not has_page_ref:
        missing.append("page number reference")
    if not has_doc_id:
        missing.append("document ID")

    if len(hits) >= 2 and not missing:
        return _result(
            rule,
            status="passed",
            reason="Footer/control elements detected in document text.",
            evidence=f"Matched terms: {', '.join(hits)}",
            confidence=0.8,
        )
    if missing:
        return _result(
            rule,
            status="failed",
            reason=f"Missing footer element(s): {', '.join(missing)}.",
            evidence=f"Matched terms: {', '.join(hits) or 'none'}",
            confidence=0.85,
        )
    return _result(
        rule,
        status="insufficient_evidence",
        reason="Could not confirm all footer control fields from extracted text.",
        evidence=f"Matched terms: {', '.join(hits) or 'none'}",
        confidence=0.6,
    )


def check_gdp_14(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    text = _full_text(extracted)
    terms = _search_terms(rule) or ["informational purposes only", "regulatory"]
    found, matched = _contains_any(text, terms)
    if found:
        return _result(
            rule,
            status="passed",
            reason="Required compliance statement language was found.",
            evidence=f"Matched phrase: {matched}",
            confidence=0.9,
        )
    return _result(
        rule,
        status="failed",
        reason="Required compliance statement was not found.",
        evidence=f"Expected one of: {', '.join(terms)}",
        confidence=0.85,
    )


def check_gdp_17(rule: dict, extracted: dict, *, file_name: str | None = None) -> dict:
    text = _full_text(extracted)
    placeholders: list[str] = []
    for pattern in PLACEHOLDER_PATTERNS:
        placeholders.extend(match.group(0) for match in pattern.finditer(text))

    unique = sorted(set(placeholders))
    if unique:
        preview = ", ".join(unique[:8])
        suffix = "..." if len(unique) > 8 else ""
        return _result(
            rule,
            status="failed",
            reason=f"Found {len(unique)} unfilled placeholder or blank operational field(s).",
            evidence=f"{preview}{suffix}",
            confidence=0.95,
        )
    return _result(
        rule,
        status="passed",
        reason="No unfilled placeholders or blank operational fields detected.",
        confidence=0.9,
    )


def check_generic_keyword_rule(
    rule: dict,
    extracted: dict,
    *,
    file_name: str | None = None,
) -> dict:
    text = _full_text(extracted)
    terms = _search_terms(rule)
    if not terms:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="No deterministic checker or keywords configured for this rule.",
            confidence=0.0,
        )
    found, matched = _contains_any(text, terms)
    if found:
        return _result(
            rule,
            status="passed",
            reason="Deterministic keyword check passed.",
            evidence=f"Matched: {matched}",
            confidence=0.75,
        )
    return _result(
        rule,
        status="failed",
        reason="Deterministic keyword check failed.",
        evidence=f"Expected one of: {', '.join(terms)}",
        confidence=0.75,
    )


CHECKERS: dict[str, Callable[..., dict]] = {
    "GDP-08": check_gdp_08,
    "GDP-09": check_gdp_09,
    "GDP-10": check_gdp_10,
    "GDP-11": check_gdp_11,
    "GDP-13": check_gdp_13,
    "GDP-14": check_gdp_14,
    "GDP-17": check_gdp_17,
}


def split_rules_by_type(
    rules: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    semantic: list[dict] = []
    deterministic: list[dict] = []
    existential: list[dict] = []
    for rule in rules:
        rule_type = rule.get("rule_type", "semantic")
        if rule_type == "deterministic":
            deterministic.append(rule)
        elif rule_type == "existential":
            existential.append(rule)
        else:
            semantic.append(rule)
    return semantic, deterministic, existential


def evaluate_deterministic_rules(
    extracted: dict,
    rules: list[dict],
    *,
    file_name: str | None = None,
) -> list[dict]:
    logger.info("Evaluating %s deterministic rule(s)", len(rules))
    results: list[dict] = []
    for rule in rules:
        checker = CHECKERS.get(rule["rule_id"], check_generic_keyword_rule)
        result = checker(rule, extracted, file_name=file_name)
        logger.debug(
            "Deterministic %s → %s (%s)",
            rule["rule_id"],
            result["status"],
            result.get("reason", "")[:80],
        )
        results.append(result)
    passed = sum(1 for item in results if item["status"] == "passed")
    logger.info("Deterministic complete: %s passed, %s failed/other", passed, len(results) - passed)
    return results


def enrich_deterministic_results(results: list[dict], rule_catalog: list[dict]) -> list[dict]:
    rule_map = {rule["rule_id"]: rule for rule in rule_catalog}
    enriched: list[dict] = []
    for item in results:
        rule = rule_map.get(item["rule_id"], {})
        enriched.append(
            {
                **item,
                "title": rule.get("title", ""),
                "severity": rule.get("severity", ""),
                "recommendation": rule.get("recommendation", ""),
                "rule_type": rule.get("rule_type", "deterministic"),
                "check_method": "deterministic",
                "evidence_chunks": [item.get("chunk_id")] if item.get("chunk_id") else [],
            }
        )
    return enriched
