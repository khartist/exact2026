"""Shared schemas for the Type 2 physics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class LawStatement:
    id: str
    topic: str
    statement: str
    keywords: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    related_formula_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Type2SolverConfig:
    model: str = "gemma4:e2b-it-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 180.0


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    repair_prompt: str = ""


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    unit: str
    explanation: str
    cot: list[str]
    premises: list[str]
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "unit": self.unit,
            "explanation": self.explanation,
            "cot": self.cot,
            "premises": self.premises,
        }
