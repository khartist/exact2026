"""Code generator LangChain agent for validated Type 2 plans."""

from __future__ import annotations

import json
import re
from typing import Any

from ..prompts import load_prompt
from ..state import Type2State
from .common import invoke_json_agent


def code_generator_agent(state: Type2State, llm: Any) -> dict[str, Any]:
    attempts = state.get("code_attempts", 0) + 1
    payload = {
        "question": state["question"],
        "validated_plan": state["plan"],
        "previous_code_payload": state.get("code_payload"),
        "validator_errors": state.get("execution_validation").errors
        if state.get("execution_validation")
        else [],
        "attempt": attempts,
    }
    code_payload = invoke_json_agent(llm, load_prompt("code_generator.txt"), payload)
    if not isinstance(code_payload, dict) or not code_payload.get("code"):
        code_payload = deterministic_code_payload(state["plan"])
    return {"code_payload": code_payload, "code_attempts": attempts}


def deterministic_code_payload(plan: dict[str, Any]) -> dict[str, Any]:
    code = deterministic_code_from_plan(plan)
    return {
        "status": "SUCCESS" if code else "ERROR",
        "code": code,
        "trace": [],
        "final": {},
        "error": None if code else "Could not generate deterministic code.",
    }


def deterministic_code_from_plan(plan: dict[str, Any]) -> str:
    steps = plan.get("steps", [])
    target = plan.get("target", {})
    if not isinstance(steps, list) or not steps:
        return ""

    lines = ["trace = []"]
    for step in steps:
        if not isinstance(step, dict):
            return ""
        output = str(step.get("output", "")).strip()
        formula = str(step.get("formula", "")).strip()
        unit = str(step.get("unit", "")).strip()
        step_id = str(step.get("id", "")).strip()
        if not output or "=" not in formula:
            return ""
        _, expression = (part.strip() for part in formula.split("=", 1))
        lines.append(f"{output} = {expression}")
        substitution_code = substitution_expression(output, expression, step.get("inputs", []))
        lines.append(
            "trace.append({"
            f"'step_id': {step_id!r}, "
            f"'formula': {formula!r}, "
            f"'substitution': {substitution_code}, "
            f"'output': {output!r}, "
            f"'value': float({output}), "
            f"'unit': {unit!r}"
            "})"
        )

    final_symbol = str(target.get("symbol", "")).strip() or str(steps[-1].get("output", "")).strip()
    final_unit = str(target.get("unit", "")).strip() or str(steps[-1].get("unit", "")).strip()
    lines.append(
        f"final = {{'symbol': {final_symbol!r}, 'value': float({final_symbol}), 'unit': {final_unit!r}}}"
    )
    return "\n".join(lines)


def substitution_expression(output: str, expression: str, inputs: Any) -> str:
    if not isinstance(inputs, list):
        return repr(f"{output} = {expression}")
    rendered = expression
    for symbol in sorted((str(item) for item in inputs), key=len, reverse=True):
        if not symbol:
            continue
        rendered = re.sub(rf"\b{re.escape(symbol)}\b", f"{{{symbol}:g}}", rendered)
    if "{" not in rendered:
        return repr(f"{output} = {expression}")
    return "f" + json.dumps(f"{output} = {rendered}")
