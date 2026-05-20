"""Deterministic validation for planner calculation contracts."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

import sympy as sp

from ..schemas import ValidationResult

ALLOWED_FORMULA_NAMES = {
    name for name in vars(math) if not name.startswith("_")
} | {
    name for name in vars(sp) if not name.startswith("_")
} | {"math", "sp", "sympy"}


def validate_calculation_plan(plan: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return invalid(["Plan must be a JSON object."])

    target = plan.get("target")
    givens = plan.get("givens")
    steps = plan.get("steps")
    final_step = plan.get("final_step")
    missing = plan.get("missing_information", [])
    status = str(plan.get("status", "")).strip()
    direct_answer = clean_text(plan.get("answer"))
    direct_explanation = clean_text(plan.get("explanation"))

    if not isinstance(target, dict):
        errors.append("target must be an object.")
        target = {}
    target_symbol = clean_text(target.get("symbol"))
    target_unit = clean_text(target.get("unit"))
    if not target_symbol:
        errors.append("target.symbol is required.")
    if status == "READY" and not target_unit:
        errors.append("target.unit is required.")
    if status == "DIRECT_ANSWER":
        if not direct_answer:
            errors.append("answer is required when status is DIRECT_ANSWER.")
        if not direct_explanation:
            errors.append("explanation is required when status is DIRECT_ANSWER.")

    if not isinstance(givens, dict) or not givens:
        if status == "READY":
            errors.append("givens must be a non-empty object when status is READY.")
        givens = {}
    given_symbols = set(givens)
    for symbol, raw_given in givens.items():
        if not isinstance(raw_given, dict):
            errors.append(f"given {symbol} must be an object.")
            continue
        for field in ("value", "unit", "si_value", "si_unit"):
            if field not in raw_given:
                errors.append(f"given {symbol}.{field} is required.")
        if not isinstance(raw_given.get("value"), (int, float)) or isinstance(raw_given.get("value"), bool):
            errors.append(f"given {symbol}.value must be numeric.")
        if not clean_text(raw_given.get("unit")):
            errors.append(f"given {symbol}.unit is required.")
        if not isinstance(raw_given.get("si_value"), (int, float)) or isinstance(raw_given.get("si_value"), bool):
            errors.append(f"given {symbol}.si_value must be numeric.")
        if not clean_text(raw_given.get("si_unit")):
            errors.append(f"given {symbol}.si_unit is required.")

    if not isinstance(steps, list) or not steps:
        if status == "READY":
            errors.append("steps must be a non-empty list when status is READY.")
        if status == "DIRECT_ANSWER" and steps:
            errors.append("steps must be empty when status is DIRECT_ANSWER.")
        steps = []

    if status == "READY" and not final_step:
        errors.append("final_step is required.")
    if status == "DIRECT_ANSWER" and clean_text(final_step):
        errors.append("final_step must be empty when status is DIRECT_ANSWER.")

    if status == "READY" and missing:
        errors.append("missing_information must be empty when status is READY.")
    if status == "DIRECT_ANSWER" and missing:
        errors.append("missing_information must be empty when status is DIRECT_ANSWER.")

    available = set(given_symbols)
    outputs: set[str] = set()
    step_ids: set[str] = set()
    final_step_obj: dict[str, Any] | None = None

    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            errors.append(f"steps[{index}] must be an object.")
            continue
        step_id = clean_text(raw_step.get("id"))
        output = clean_text(raw_step.get("output"))
        formula = clean_text(raw_step.get("formula"))
        inputs = raw_step.get("inputs")
        unit = clean_text(raw_step.get("unit"))

        if not step_id:
            errors.append(f"steps[{index}].id is required.")
        elif step_id in step_ids:
            errors.append(f"Duplicate step id: {step_id}.")
        else:
            step_ids.add(step_id)

        if step_id == final_step:
            final_step_obj = raw_step

        if not output:
            errors.append(f"steps[{index}].output is required.")
        elif output in given_symbols:
            errors.append(f"Step {step_id or index} overwrites given variable {output}.")
        elif output in outputs:
            errors.append(f"Duplicate step output: {output}.")

        if not isinstance(inputs, list):
            errors.append(f"Step {step_id or index} inputs must be a list.")
            inputs = []
        input_symbols = [clean_text(item) for item in inputs if clean_text(item)]
        for symbol in input_symbols:
            if symbol not in available:
                errors.append(f"Step {step_id or index} input {symbol} is undefined.")

        formula_errors = validate_formula_text(formula, output, set(input_symbols))
        errors.extend(f"Step {step_id or index}: {error}" for error in formula_errors)

        if not unit and step_id == final_step:
            errors.append(f"Step {step_id or index} unit is required.")

        if output:
            outputs.add(output)
            available.add(output)

    if final_step and final_step not in step_ids:
        errors.append(f"final_step {final_step} does not exist.")
    if final_step_obj is not None:
        if clean_text(final_step_obj.get("output")) != target_symbol:
            errors.append("final step output must equal target symbol.")
        if clean_text(final_step_obj.get("unit")) != target_unit:
            errors.append("final step unit must equal target unit.")

    if status == "DIRECT_ANSWER":
        if steps:
            errors.append("steps must be empty when status is DIRECT_ANSWER.")
        if not direct_answer:
            errors.append("answer is required when status is DIRECT_ANSWER.")
        if not direct_explanation:
            errors.append("explanation is required when status is DIRECT_ANSWER.")

    return ValidationResult(not errors, errors, build_repair_prompt(errors))


def validate_formula_text(formula: str, output: str, inputs: set[str]) -> list[str]:
    if not formula:
        return ["formula is required."]
    if "=" not in formula:
        return ["formula must contain '='."]
    left, right = (part.strip() for part in formula.split("=", 1))
    if left != output:
        return [f"formula left side {left!r} must match output {output!r}."]
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", right))
    unknown = names - inputs - ALLOWED_FORMULA_NAMES
    if unknown:
        return [f"formula references undeclared names: {', '.join(sorted(unknown))}."]
    try:
        ast.parse(right, mode="eval")
    except SyntaxError as exc:
        return [f"formula expression is invalid Python syntax: {exc.msg}."]
    if "__" in right:
        return ["formula expression contains forbidden dunder token."]
    return []


def invalid(errors: list[str]) -> ValidationResult:
    return ValidationResult(False, errors, build_repair_prompt(errors))


def build_repair_prompt(errors: list[str]) -> str:
    if not errors:
        return ""
    return "Repair the calculation plan. Validator errors: " + "; ".join(errors)


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()
