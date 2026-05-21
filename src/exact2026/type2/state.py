"""LangGraph state for the structured Type 2 solver."""

from __future__ import annotations

from typing import Any, TypedDict

from .schemas import PipelineResult, ValidationResult


class Type2State(TypedDict, total=False):
    question: str
    knowledge: dict[str, list[dict[str, Any]]] | list[dict[str, Any]]
    knowledge_config: Any
    search_query: str
    plan: dict[str, Any]
    planner_attempts: int
    planner_validation: ValidationResult
    code_payload: dict[str, Any]
    execution: dict[str, Any]
    code_attempts: int
    execution_validation: ValidationResult
    fallback_reason: str
    review: dict[str, Any]
    result: PipelineResult
