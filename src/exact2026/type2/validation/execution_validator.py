"""Deterministic validation for structured execution results."""

from __future__ import annotations

import math
from typing import Any

from ..schemas import ValidationResult


def validate_structured_execution(
    plan: dict[str, Any],
    execution: dict[str, Any],
) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(execution, dict):
        return invalid(["Execution result must be a JSON object."])
    if execution.get("status") != "SUCCESS":
        errors.append(f"execution status must be SUCCESS, got {execution.get('status')!r}.")

    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        steps = []
        errors.append("Plan steps are missing.")

    trace = execution.get("trace")
    if not isinstance(trace, list):
        trace = []
        errors.append("execution trace must be a list.")
    if len(trace) != len(steps):
        errors.append("execution trace length must match planned step count.")

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if index >= len(trace) or not isinstance(trace[index], dict):
            errors.append(f"Trace item missing for planned step {step.get('id')}.")
            continue
        item = trace[index]
        expected_id = str(step.get("id", "")).strip()
        expected_output = str(step.get("output", "")).strip()
        expected_unit = str(step.get("unit", "")).strip()
        if item.get("step_id") != expected_id:
            errors.append(f"Trace item {index} step_id must be {expected_id}.")
        if item.get("output") != expected_output:
            errors.append(f"Trace item {index} output must be {expected_output}.")
        if item.get("unit") != expected_unit:
            errors.append(f"Trace item {index} unit must be {expected_unit}.")
        if not str(item.get("substitution", "")).strip():
            errors.append(f"Trace item {index} substitution is required.")
        if not is_finite_number(item.get("value")):
            errors.append(f"Trace item {index} value must be numeric and finite.")

    target = plan.get("target") if isinstance(plan, dict) else {}
    final = execution.get("final")
    if not isinstance(final, dict):
        final = {}
        errors.append("execution final must be an object.")
    if final.get("symbol") != target.get("symbol"):
        errors.append("final symbol must equal target symbol.")
    if final.get("unit") != target.get("unit"):
        errors.append("final unit must equal target unit.")
    if not is_finite_number(final.get("value")):
        errors.append("final value must be numeric and finite.")

    planned_outputs = {
        str(step.get("output", "")).strip()
        for step in steps
        if isinstance(step, dict) and step.get("output")
    }
    traced_outputs = {
        str(item.get("output", "")).strip()
        for item in trace
        if isinstance(item, dict) and item.get("output")
    }
    missing_outputs = planned_outputs - traced_outputs
    if missing_outputs:
        errors.append(
            "Trace is missing planned outputs: " + ", ".join(sorted(missing_outputs)) + "."
        )

    return ValidationResult(not errors, errors, build_repair_prompt(errors))


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def invalid(errors: list[str]) -> ValidationResult:
    return ValidationResult(False, errors, build_repair_prompt(errors))


def build_repair_prompt(errors: list[str]) -> str:
    if not errors:
        return ""
    return "Repair the generated code/execution result. Validator errors: " + "; ".join(errors)
