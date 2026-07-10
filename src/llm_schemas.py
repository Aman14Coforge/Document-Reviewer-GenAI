"""Pydantic schemas for validating structured LLM JSON responses."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComplianceStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    not_applicable = "not_applicable"
    insufficient_evidence = "insufficient_evidence"


class ComplianceRuleResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rule_id: str = Field(min_length=1)
    status: ComplianceStatus
    reason: str = ""
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("rule_id", mode="before")
    @classmethod
    def strip_rule_id(cls, value: object) -> str:
        return str(value).strip()


class ComplianceCheckResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[ComplianceRuleResult]
    chunk_id: str | None = None


class RuleCategory(str, Enum):
    header = "header"
    revision_history = "revision_history"
    approval = "approval"
    formatting = "formatting"
    language = "language"
    structure = "structure"
    footer = "footer"


class RuleType(str, Enum):
    deterministic = "deterministic"
    semantic = "semantic"
    existential = "existential"


class ExternalDependency(str, Enum):
    audit_log = "audit_log"
    reference_registry = "reference_registry"
    traceability_matrix = "traceability_matrix"


class GeneratedRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rule_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: RuleCategory
    severity: Literal["low", "medium", "high"]
    rule_type: RuleType
    verifiable_criteria: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    applies_to_sections: list[str] = Field(min_length=1)
    requires_metadata: list[str] = Field(default_factory=list)
    external_dependency: ExternalDependency | None = None
    requires_supporting_documents: bool = False

    @field_validator("rule_id", mode="before")
    @classmethod
    def strip_rule_id(cls, value: object) -> str:
        return str(value).strip()


class RulesGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str = "1.0"
    rules: list[GeneratedRule] = Field(min_length=1)
