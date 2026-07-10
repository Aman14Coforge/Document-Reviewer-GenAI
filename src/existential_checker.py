"""Existential compliance rules validated against external JSON registries.

GDP-18 → reference_registry (DOC-xxx)
GDP-19 → audit_log (authorized editors)
GDP-20 → traceability_matrix (URS → FRS → TC)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger("existential")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "compliance.json"

DOC_REF_PATTERN = re.compile(r"\bDOC-\d+\b", re.IGNORECASE)
URS_PATTERN = re.compile(r"\bURS-\d+\b", re.IGNORECASE)
FRS_PATTERN = re.compile(r"\bFRS-\d+\b", re.IGNORECASE)
TC_PATTERN = re.compile(r"\bTC-\d+\b", re.IGNORECASE)
MODIFIED_BY_PATTERN = re.compile(
    r"(?:last\s+)?(?:modified|edited)\s+by\s*[:\-]?\s*([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)

DEFAULT_EXTERNAL_PATHS = {
    "audit_log": PROJECT_ROOT / "data" / "external" / "audit_log.json",
    "reference_registry": PROJECT_ROOT / "data" / "external" / "reference_registry.json",
    "traceability_matrix": PROJECT_ROOT / "data" / "external" / "traceability_matrix.json",
}


def load_external_config() -> dict[str, Path]:
    paths = dict(DEFAULT_EXTERNAL_PATHS)
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        configured = config.get("external_dependencies", {})
        for key, value in configured.items():
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths[key] = path
    return paths


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
        "check_method": "existential",
        "chunk_id": "whole_document",
        "section_type": "full",
    }


def _full_text(extracted: dict) -> str:
    return extracted.get("full_text", "") or ""


def _approved_users_from_audit_log(data: dict | None) -> set[str]:
    approved: set[str] = set()
    if not data:
        return approved
    for entry in data.get("logs", []):
        approved.update(str(user).lower() for user in entry.get("approved_users", []))
        user = entry.get("user")
        if user:
            approved.add(str(user).lower())
    return approved


def _registered_doc_ids(data: dict | None) -> set[str]:
    if not data:
        return set()
    return {
        str(item.get("doc_id", "")).upper()
        for item in data.get("documents", [])
        if item.get("doc_id")
    }


def check_gdp_18(
    rule: dict,
    extracted: dict,
    *,
    external_data: dict | None = None,
    file_name: str | None = None,
) -> dict:
    text = _full_text(extracted)
    refs = sorted({match.group(0).upper() for match in DOC_REF_PATTERN.finditer(text)})
    registry = _registered_doc_ids(external_data)

    if external_data is None:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="Reference registry file not found or not configured.",
            confidence=0.0,
        )
    if not refs:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="No document references (DOC-xxx) found in the document text.",
            confidence=0.5,
        )

    missing = [ref for ref in refs if ref not in registry]
    if missing:
        return _result(
            rule,
            status="failed",
            reason=f"{len(missing)} reference(s) not found in the reference registry.",
            evidence=f"Missing: {', '.join(missing)}; valid registry: {', '.join(sorted(registry))}",
            confidence=0.95,
        )
    return _result(
        rule,
        status="passed",
        reason="All document references exist in the reference registry.",
        evidence=f"Validated: {', '.join(refs)}",
        confidence=0.95,
    )


def check_gdp_19(
    rule: dict,
    extracted: dict,
    *,
    external_data: dict | None = None,
    file_name: str | None = None,
) -> dict:
    text = _full_text(extracted)
    editors = sorted(
        {
            match.group(1).lower()
            for match in MODIFIED_BY_PATTERN.finditer(text)
        }
    )
    approved = _approved_users_from_audit_log(external_data)

    if external_data is None:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="Audit log file not found or not configured.",
            confidence=0.0,
        )
    if not approved:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="Audit log contains no approved users.",
            confidence=0.0,
        )
    if not editors:
        return _result(
            rule,
            status="passed",
            reason="No edit-history markers found; no unauthorized edits detected.",
            evidence=f"Approved users: {', '.join(sorted(approved))}",
            confidence=0.7,
        )

    unauthorized = [user for user in editors if user not in approved]
    if unauthorized:
        return _result(
            rule,
            status="failed",
            reason="Unauthorized editor(s) detected against audit log.",
            evidence=f"Unauthorized: {', '.join(unauthorized)}; approved: {', '.join(sorted(approved))}",
            confidence=0.95,
        )
    return _result(
        rule,
        status="passed",
        reason="All detected editors are authorized in the audit log.",
        evidence=f"Editors: {', '.join(editors)}",
        confidence=0.9,
    )


def check_gdp_20(
    rule: dict,
    extracted: dict,
    *,
    external_data: dict | None = None,
    file_name: str | None = None,
) -> dict:
    text = _full_text(extracted)
    urs_ids = {match.group(0).upper() for match in URS_PATTERN.finditer(text)}
    frs_ids = {match.group(0).upper() for match in FRS_PATTERN.finditer(text)}
    tc_ids = {match.group(0).upper() for match in TC_PATTERN.finditer(text)}

    if external_data is None:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="Traceability matrix file not found or not configured.",
            confidence=0.0,
        )

    matrix = external_data.get("traceability_matrix", [])
    matrix_urs = {str(row.get("urs_id", "")).upper() for row in matrix if row.get("urs_id")}
    matrix_frs = {str(row.get("frs_id", "")).upper() for row in matrix if row.get("frs_id")}
    matrix_tc = {
        str(tc).upper()
        for row in matrix
        for tc in row.get("test_cases", [])
    }

    if not urs_ids and not frs_ids and not tc_ids:
        return _result(
            rule,
            status="insufficient_evidence",
            reason="No URS/FRS/TC identifiers found in the document.",
            confidence=0.5,
        )

    issues: list[str] = []
    for urs_id in sorted(urs_ids):
        row = next((item for item in matrix if str(item.get("urs_id", "")).upper() == urs_id), None)
        if not row:
            issues.append(f"{urs_id} missing from traceability matrix")
            continue
        if urs_id in frs_ids and str(row.get("frs_id", "")).upper() not in frs_ids:
            issues.append(f"{urs_id} not linked to expected FRS in document")
        for tc in row.get("test_cases", []):
            if str(tc).upper() not in tc_ids and tc_ids:
                issues.append(f"{urs_id} missing test case {tc} in document")

    for frs_id in sorted(frs_ids - matrix_frs):
        issues.append(f"{frs_id} not registered in traceability matrix")
    for tc_id in sorted(tc_ids - matrix_tc):
        issues.append(f"{tc_id} not registered in traceability matrix")

    if issues:
        return _result(
            rule,
            status="failed",
            reason="Traceability linkage gaps detected.",
            evidence="; ".join(issues[:10]),
            confidence=0.9,
        )

    return _result(
        rule,
        status="passed",
        reason="URS, FRS, and test case references align with the traceability matrix.",
        evidence=(
            f"URS: {', '.join(sorted(urs_ids)) or 'none'}; "
            f"FRS: {', '.join(sorted(frs_ids)) or 'none'}; "
            f"TC: {', '.join(sorted(tc_ids)) or 'none'}"
        ),
        confidence=0.9,
    )


CHECKERS = {
    "GDP-18": check_gdp_18,
    "GDP-19": check_gdp_19,
    "GDP-20": check_gdp_20,
}

DEPENDENCY_KEY = {
    "GDP-18": "reference_registry",
    "GDP-19": "audit_log",
    "GDP-20": "traceability_matrix",
}


def evaluate_existential_rules(
    extracted: dict,
    rules: list[dict],
    *,
    file_name: str | None = None,
    external_paths: dict[str, Path] | None = None,
) -> list[dict]:
    paths = external_paths or load_external_config()
    logger.info("Evaluating %s existential rule(s)", len(rules))
    cache: dict[str, dict | list | None] = {}
    results: list[dict] = []

    for rule in rules:
        rule_id = rule["rule_id"]
        checker = CHECKERS.get(rule_id)
        if not checker:
            logger.warning("No existential checker for %s", rule_id)
            results.append(
                _result(
                    rule,
                    status="insufficient_evidence",
                    reason=f"No existential checker implemented for {rule_id}.",
                    confidence=0.0,
                )
            )
            continue

        dep_key = rule.get("external_dependency") or DEPENDENCY_KEY.get(rule_id, "")
        if dep_key not in cache:
            dep_path = paths.get(dep_key, Path())
            cache[dep_key] = _load_json(dep_path)
            if cache[dep_key] is None:
                logger.warning("External dependency missing or unreadable: %s (%s)", dep_key, dep_path)
            else:
                logger.debug("Loaded external dependency %s from %s", dep_key, dep_path)

        result = checker(rule, extracted, external_data=cache[dep_key], file_name=file_name)
        logger.debug("Existential %s → %s (%s)", rule_id, result["status"], result.get("reason", "")[:80])
        results.append(result)

    passed = sum(1 for item in results if item["status"] == "passed")
    logger.info("Existential complete: %s passed, %s failed/other", passed, len(results) - passed)
    return results


def enrich_existential_results(results: list[dict], rule_catalog: list[dict]) -> list[dict]:
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
                "rule_type": rule.get("rule_type", "existential"),
                "check_method": "existential",
                "external_dependency": rule.get("external_dependency", ""),
                "evidence_chunks": [item.get("chunk_id")] if item.get("chunk_id") else [],
            }
        )
    return enriched
