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
    category: str = ""
    assumptions: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()


@dataclass(frozen=True)
class LawStatement:
    id: str
    topic: str
    statement: str
    keywords: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    related_formula_ids: tuple[str, ...] = ()
    category: str = ""
    diagram_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolDefinition:
    symbol: str
    meaning: str
    si_unit: str
    aliases: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class PhysicalConstant:
    symbol: str
    value: str
    si_unit: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkedExample:
    id: str
    category: str
    title: str
    prompt: str
    givens: tuple[str, ...]
    steps: tuple[str, ...]
    answer: tuple[str, ...]
    related_formula_ids: tuple[str, ...] = ()
    related_law_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagram:
    id: str
    title: str
    mermaid: str
    related_formula_ids: tuple[str, ...] = ()
    related_law_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    summary: str
    formula_ids: tuple[str, ...]
    law_ids: tuple[str, ...]
    example_ids: tuple[str, ...] = ()
    symbol_ids: tuple[str, ...] = ()
    constant_ids: tuple[str, ...] = ()
    diagram_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeSearchConfig:
    enabled_types: tuple[str, ...] = (
        "formulas",
        "laws",
        "symbols",
        "constants",
        "examples",
        "categories",
        "diagrams",
    )
    top_k: int = 8
    max_examples: int = 2
    max_diagrams: int = 1


@dataclass(frozen=True)
class Type2SolverConfig:
    model: str = "gemma4:e2b-it-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: float = 180.0
    knowledge_context_types: tuple[str, ...] = KnowledgeSearchConfig().enabled_types
    knowledge_top_k: int = 8
    knowledge_max_examples: int = 2
    knowledge_max_diagrams: int = 1


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
