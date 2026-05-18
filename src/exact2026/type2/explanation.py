"""Explanation generation for verified Type 2 solutions."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .formula_bank import get_formula
from .schemas import EquationPlan, ExecutionResult, ParsedProblem, VerificationResult

ModelCompleteFn = Callable[[str, str], str]

EXPLANATION_SYSTEM_PROMPT = """You write concise physics solution steps.
Use only the provided variables, formula, execution trace, answer, and unit.
Return strict JSON only with keys: explanation, cot, premises."""


def build_explanation(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
    verification: VerificationResult,
    answer_text: str,
    model_complete: ModelCompleteFn | None = None,
) -> tuple[str, list[str], list[str]]:
    generated = generate_explanation_with_model(
        parsed, plan, execution, answer_text, model_complete
    )
    if generated is not None:
        return generated
    return build_deterministic_explanation(parsed, plan, execution, verification, answer_text)


def generate_explanation_with_model(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
    answer_text: str,
    model_complete: ModelCompleteFn | None,
) -> tuple[str, list[str], list[str]] | None:
    if model_complete is None:
        return None
    formula = get_formula(plan.formula_id)
    payload = {
        "question": parsed.question,
        "knowns": {symbol: known.raw for symbol, known in parsed.knowns.items()},
        "si_values": {
            symbol: {"value": known.si_value, "unit": known.si_unit}
            for symbol, known in parsed.knowns.items()
        },
        "formula": formula.equation if formula else plan.steps[0].formula,
        "trace": execution.trace,
        "answer": answer_text,
        "unit": execution.unit,
    }
    try:
        raw = model_complete(EXPLANATION_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        parsed_json = parse_json_object(raw)
    except Exception:
        return None
    explanation = clean_scalar(parsed_json.get("explanation"))
    cot = clean_string_list(parsed_json.get("cot"))
    premises = clean_string_list(parsed_json.get("premises"))
    if not explanation or not cot or not premises:
        return None
    return explanation, cot, premises


def build_deterministic_explanation(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
    verification: VerificationResult,
    answer_text: str,
) -> tuple[str, list[str], list[str]]:
    formula = get_formula(plan.formula_id)
    formula_text = formula.description if formula else plan.steps[0].formula
    known_text = ", ".join(
        f"{symbol} = {known.raw}" for symbol, known in sorted(parsed.knowns.items())
    )
    conversion_text = ", ".join(
        f"{symbol} = {known.si_value:g} {known.si_unit}"
        for symbol, known in sorted(parsed.knowns.items())
        if known.unit and known.unit != known.si_unit
    )

    cot = [f"Identify known variables: {known_text}."]
    if conversion_text:
        cot.append(f"Convert to SI units: {conversion_text}.")
    cot.append(f"Select formula: {formula.equation if formula else plan.steps[0].formula}.")
    cot.extend(f"Execute step: {item}." for item in execution.trace)
    if not verification.passed:
        cot.append("Verification reported reduced confidence for this result.")

    explanation = " ".join(f"Step {index}: {step}" for index, step in enumerate(cot, start=1))
    premises = [formula_text]
    return explanation, cot, premises


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := clean_scalar(item))]
