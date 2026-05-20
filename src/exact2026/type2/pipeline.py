"""Orchestration for the Type 2 physics MVP pipeline."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from typing import Any

from .json_utils import extract_json_like_fields, extract_json_object
from .llm import build_type2_llm
from .schemas import PipelineResult, Type2SolverConfig, ValidationResult

logger = logging.getLogger(__name__)

FallbackModelFn = Callable[[str], str]

FALLBACK_SYSTEM_PROMPT = """You solve physics problems.
Use only the stated problem information and standard physics formulas.
Return strict JSON only with keys: answer, unit, formula, explanation, cot, premises.
The answer must be numeric without the unit. Use LaTeX only when it improves readability, and escape backslashes so the response remains valid JSON. Do not use markdown."""


def solve_physics_question(
    question: str,
    config: Type2SolverConfig | None = None,
    llm: Any = None,
    fallback_model: FallbackModelFn | None = None,
) -> PipelineResult:
    from .graph import solve_with_structured_graph

    logger.info("type2.raw_question=%s", question)
    active_llm = llm if llm is not None else build_type2_llm(config) if config else None
    result = solve_with_structured_graph(question, llm=active_llm)
    if not result.answer and fallback_model is not None:
        result = solve_with_llm_fallback(question, fallback_model)
    logger.info("type2.final_response=%s", result)
    return result


def solve_with_llm_fallback(
    question: str,
    fallback_model: FallbackModelFn | None,
) -> PipelineResult:
    if fallback_model is None:
        return PipelineResult(
            answer="",
            unit="",
            explanation=(
                "The problem could not be solved by the current formula bank, and no "
                "fallback model was available to produce an answer."
            ),
            cot=[
                "Parsed the question.",
                "Checked the formula bank for a matching trusted formula.",
                "No matching formula or fallback model was available.",
            ],
            premises=[],
            metadata={"fallback": "unavailable"},
        )

    prompt = "\n".join(["Physics problem:", question, "", "Return strict JSON only."])
    try:
        raw = fallback_model(prompt)
    except Exception as exc:
        return PipelineResult(
            answer="",
            unit="",
            explanation=f"The fallback model failed: {exc}",
            cot=["Formula bank did not match.", "Fallback model call failed."],
            premises=[],
            raw_response=str(exc),
            metadata={"fallback": "error"},
        )
    parsed = extract_json_object(raw) or extract_json_like_fields(
        raw,
        ("answer", "unit", "explanation", "formula"),
    )
    answer = clean_scalar(parsed.get("answer"))
    unit = clean_scalar(parsed.get("unit"))
    verification = verify_llm_fallback(answer, unit)
    explanation = clean_scalar(parsed.get("explanation")) or "Solved using fallback model physics knowledge."
    cot = normalize_string_list(parsed.get("cot")) or cot_from_explanation(explanation)
    premises = normalize_string_list(parsed.get("premises"))
    if not premises and parsed.get("formula"):
        premises = [clean_scalar(parsed.get("formula"))]
    return PipelineResult(
        answer=answer,
        unit=unit,
        explanation=explanation,
        cot=cot,
        premises=premises,
        raw_response=raw,
        metadata={"fallback": "llm", "verified": verification.ok},
    )


def verify_llm_fallback(answer: str, unit: str) -> ValidationResult:
    errors: list[str] = []
    try:
        numeric = float(answer)
    except ValueError:
        numeric = None
        errors.append("Fallback answer is not numeric.")
    if numeric is not None and not math.isfinite(numeric):
        errors.append("Fallback answer is not finite.")
    if not unit:
        errors.append("Fallback unit is missing.")
    return ValidationResult(not errors, errors)


def format_number(value: float) -> str:
    rounded = round(value, 2)
    if abs(value - rounded) <= max(1e-9, abs(value) * 1e-4):
        text = f"{rounded:.2f}"
    else:
        text = f"{value:.6g}"
    return re.sub(r"\.?0+$", "", text) if "." in text else text


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := clean_scalar(item))]


def cot_from_explanation(explanation: str) -> list[str]:
    if not explanation:
        return [
            "Formula bank did not match.",
            "Used fallback model physics knowledge.",
        ]
    inline_steps = re.findall(
        r"(?:Step\s+\d+\s*:|\d+\.)\s+.*?(?=\s+(?:Step\s+\d+\s*:|\d+\.)\s+|$)",
        explanation,
        re.DOTALL,
    )
    if len(inline_steps) > 1:
        return [clean_scalar(item) for item in inline_steps]
    candidates = re.split(r"(?:^|\n)\s*(?=\d+\.\s+|Step\s+\d+\s*:)", explanation)
    steps = [clean_scalar(item) for item in candidates if clean_scalar(item)]
    if len(steps) > 1:
        return steps
    lines = [clean_scalar(line) for line in explanation.splitlines() if clean_scalar(line)]
    return lines if len(lines) > 1 else [explanation]


def dumps_response(result: PipelineResult) -> str:
    return json.dumps(result.to_api_dict(), ensure_ascii=False)
