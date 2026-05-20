"""Deterministic answer composition from validated Type 2 artifacts."""

from __future__ import annotations

import re
from typing import Any

from .schemas import PipelineResult


def compose_answer(
    question: str,
    plan: dict[str, Any],
    execution: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> PipelineResult:
    final = execution.get("final", {})
    value = float(final["value"])
    unit = str(final.get("unit", "")).strip()
    answer = format_number(value)
    trace = [item for item in execution.get("trace", []) if isinstance(item, dict)]
    cot = build_full_cot(plan, trace)
    premises = unique_preserved(
        str(step.get("premise", "")).strip()
        for step in plan.get("steps", [])
        if isinstance(step, dict) and str(step.get("premise", "")).strip()
    )
    explanation = " ".join(cot)
    if not explanation:
        explanation = f"The validated execution gives {answer} {unit}."
    return PipelineResult(
        answer=answer,
        unit=unit,
        explanation=explanation,
        cot=cot or [f"Final answer: {answer} {unit}."],
        premises=premises,
        metadata={
            "agent_loop": "structured_langgraph",
            "verified": True,
            "question": question,
            **(metadata or {}),
        },
    )


def build_full_cot(plan: dict[str, Any], trace: list[dict[str, Any]]) -> list[str]:
    cot: list[str] = []
    givens_text = format_givens(plan.get("givens", {}))
    if givens_text:
        cot.append(f"Step {len(cot) + 1}: Identify the given values: {givens_text}.")

    conversion_text = format_conversions(plan.get("givens", {}))
    if conversion_text:
        cot.append(f"Step {len(cot) + 1}: Convert quantities to SI units: {conversion_text}.")

    formulas = unique_preserved(
        str(step.get("formula", "")).strip()
        for step in plan.get("steps", [])
        if isinstance(step, dict) and str(step.get("formula", "")).strip()
    )
    if formulas:
        cot.append(
            f"Step {len(cot) + 1}: Use the calculation formulas: "
            + "; ".join(pretty_formula(item) for item in formulas)
            + "."
        )

    premises = unique_preserved(
        str(step.get("premise", "")).strip()
        for step in plan.get("steps", [])
        if isinstance(step, dict) and str(step.get("premise", "")).strip()
    )
    law_premises = [item for item in premises if item not in formulas]
    if law_premises:
        cot.append(
            f"Step {len(cot) + 1}: Apply the relevant physics premises: "
            + "; ".join(law_premises)
            + "."
        )

    for item in trace:
        cot.append(trace_to_cot(len(cot) + 1, item))
    return cot


def trace_to_cot(index: int, item: dict[str, Any]) -> str:
    output = str(item.get("output", "")).strip()
    substitution = pretty_numeric_text(str(item.get("substitution", "")).strip())
    value = format_number(float(item.get("value", 0)))
    unit = str(item.get("unit", "")).strip()
    if output:
        return f"Step {index}: Substitute and compute {output}: {substitution} = {format_value_unit(value, unit)}."
    return f"Step {index}: Substitute and compute: {substitution} = {format_value_unit(value, unit)}."


def trace_to_sentence(index: int, item: dict[str, Any]) -> str:
    output = str(item.get("output", "")).strip()
    substitution = pretty_numeric_text(str(item.get("substitution", "")).strip())
    value = format_number(float(item.get("value", 0)))
    unit = str(item.get("unit", "")).strip()
    label = "Finally" if item.get("is_final") else ("First" if index == 1 else "Then")
    if output:
        return f"{label}, compute {output}: {substitution} = {format_value_unit(value, unit)}."
    return f"{label}, compute {substitution} = {format_value_unit(value, unit)}."


def unique_preserved(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def format_givens(givens: Any) -> str:
    if not isinstance(givens, dict):
        return ""
    parts: list[str] = []
    for symbol, raw in sorted(givens.items()):
        if not isinstance(raw, dict):
            continue
        value = raw.get("value")
        unit = str(raw.get("unit", "")).strip()
        parts.append(f"{symbol} = {format_value(value)} {unit}".strip())
    return ", ".join(parts)


def format_conversions(givens: Any) -> str:
    if not isinstance(givens, dict):
        return ""
    parts: list[str] = []
    for symbol, raw in sorted(givens.items()):
        if not isinstance(raw, dict):
            continue
        unit = str(raw.get("unit", "")).strip()
        si_unit = str(raw.get("si_unit", "")).strip()
        value = raw.get("value")
        si_value = raw.get("si_value")
        if not unit or not si_unit:
            continue
        if unit == si_unit and same_number(value, si_value):
            continue
        parts.append(
            f"{symbol} = {format_value(value)} {unit} = {format_value(si_value)} {si_unit}"
        )
    return ", ".join(parts)


def pretty_formula(formula: str) -> str:
    return formula.replace("**2", "^2").replace("*", " × ")


def pretty_numeric_text(text: str) -> str:
    number = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:e[-+]?\d+)?"

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            return format_number(float(raw))
        except ValueError:
            return raw

    return re.sub(number, replace, text, flags=re.IGNORECASE)


def same_number(left: Any, right: Any) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return abs(float(left) - float(right)) <= 1e-12


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return format_number(float(value))
    return pretty_numeric_text(str(value))


def format_value_unit(value: str, unit: str) -> str:
    return f"{value} {unit}".strip()


def format_number(value: float) -> str:
    nearest_integer = round(value)
    if abs(value) >= 1e-6 and abs(value - nearest_integer) <= 1e-6:
        return str(nearest_integer)
    one_decimal = round(value, 1)
    if abs(value) >= 1e-6 and abs(value - one_decimal) <= 1e-6:
        return f"{one_decimal:.1f}".rstrip("0").rstrip(".")
    two_decimal = round(value, 2)
    if abs(value) >= 1e-6 and abs(value - two_decimal) <= max(1e-9, abs(value) * 1e-4):
        return f"{two_decimal:.2f}".rstrip("0").rstrip(".")
    compact = f"{value:.6g}"
    if "e" in compact or "E" in compact:
        text = compact
    else:
        text = compact
    return re.sub(r"\.?0+$", "", text) if "." in text else text
