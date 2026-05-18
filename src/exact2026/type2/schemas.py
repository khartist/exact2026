"""Shared schemas for the Type 2 physics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnownValue:
    symbol: str
    value: float
    unit: str
    si_value: float
    si_unit: str
    raw: str


@dataclass(frozen=True)
class ParsedProblem:
    question: str
    topic: str
    knowns: dict[str, KnownValue]
    target: str
    target_unit: str
    possible_formulas: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Formula:
    id: str
    topic: str
    equation: str
    expression: str
    required_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    output_unit: str
    keywords: tuple[str, ...]
    description: str
    intermediates: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PlanStep:
    formula_id: str
    formula: str
    substitution: str
    target: str
    expression: str
    source: str = "formula_bank"


@dataclass(frozen=True)
class EquationPlan:
    formula_id: str
    steps: list[PlanStep]
    confidence: float
    source: str = "formula_bank"


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    answer: float | None
    unit: str
    intermediates: dict[str, float]
    trace: list[str]
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    confidence: float
    errors: list[str]


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    unit: str
    explanation: str
    cot: list[str]
    premises: list[str]
    confidence: float
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "unit": self.unit,
            "explanation": self.explanation,
            "cot": self.cot,
            "premises": self.premises,
            "confidence": self.confidence,
        }
