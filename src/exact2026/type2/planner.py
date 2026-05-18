"""Deterministic formula planner for Type 2 physics."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .formula_bank import FORMULAS
from .schemas import EquationPlan, Formula, ParsedProblem, PlanStep

ModelCompleteFn = Callable[[str, str], str]


PLANNER_SYSTEM_PROMPT = """You select physics formulas from an allowed list.
Return strict JSON only with key formula_id. Do not invent formulas."""

CODE_PLAN_SYSTEM_PROMPT = """You write an equation execution plan for a physics formula.
Return strict JSON only:
{"steps":[{"target":"symbol","expression":"math expression"}]}
Use only the provided known symbols, prior step targets, numbers, and functions: sqrt, sin, cos, tan, Abs, pi.
Do not include Python code, imports, assignments, markdown, or explanation."""


def build_plan(
    parsed: ParsedProblem,
    model_complete: ModelCompleteFn | None = None,
) -> EquationPlan | None:
    candidates = rank_formulas(parsed)
    if not candidates:
        return None
    formula = choose_formula_with_model(parsed, candidates, model_complete) or candidates[0]
    steps = build_steps_with_model(parsed, formula, model_complete) or build_formula_steps(
        parsed, formula
    )
    return EquationPlan(
        formula_id=formula.id,
        steps=steps,
        confidence=0.95,
        source="slm_code_plan" if steps and steps[0].source == "slm" else "formula_bank",
    )


def rank_formulas(parsed: ParsedProblem) -> list[Formula]:
    known_symbols = set(parsed.knowns)
    question_words = set(parsed.question.lower().replace("-", " ").split())
    scored: list[tuple[int, Formula]] = []
    for formula in FORMULAS:
        if not requirements_available(formula, known_symbols):
            continue
        score = 0
        if parsed.target and parsed.target in formula.target_variables:
            score += 10
        if parsed.topic and parsed.topic == formula.topic:
            score += 7
        score += len(question_words.intersection(formula.keywords))
        if formula.id == "ac_real_power_vzr" and {"V", "Z", "R"}.issubset(known_symbols):
            score += 12
        if formula.id == "series_rlc_impedance" and {"R", "L", "C", "f"}.issubset(known_symbols):
            score += 5
        if formula.id == "net_coulomb_force_right_triangle" and {"q1", "q2", "q3", "AC", "BC"}.issubset(known_symbols):
            score += 14
        if score > 0:
            scored.append((score, formula))
    scored.sort(key=lambda item: (item[0], len(item[1].required_variables)), reverse=True)
    return [formula for _, formula in scored]


def choose_formula_with_model(
    parsed: ParsedProblem,
    candidates: list[Formula],
    model_complete: ModelCompleteFn | None,
) -> Formula | None:
    if model_complete is None:
        return None
    candidate_payload = [
        {
            "id": formula.id,
            "topic": formula.topic,
            "equation": formula.equation,
            "required_variables": formula.required_variables,
            "target_variables": formula.target_variables,
            "keywords": formula.keywords,
        }
        for formula in candidates[:5]
    ]
    prompt = json.dumps(
        {
            "question": parsed.question,
            "target": parsed.target,
            "knowns": sorted(parsed.knowns),
            "candidate_formulas": candidate_payload,
        },
        ensure_ascii=False,
    )
    try:
        raw = model_complete(PLANNER_SYSTEM_PROMPT, prompt)
        selected_id = parse_formula_id(raw)
    except Exception:
        return None
    allowed = {formula.id: formula for formula in candidates}
    return allowed.get(selected_id)


def parse_formula_id(raw: str) -> str:
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return ""
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("formula_id", "")).strip()


def requirements_available(formula: Formula, known_symbols: set[str]) -> bool:
    if formula.id == "series_resistance":
        return len([symbol for symbol in known_symbols if symbol.startswith("R") and symbol != "R"]) >= 2
    return set(formula.required_variables).issubset(known_symbols)


def build_steps_with_model(
    parsed: ParsedProblem,
    formula: Formula,
    model_complete: ModelCompleteFn | None,
) -> list[PlanStep] | None:
    if model_complete is None:
        return None
    payload = {
        "question": parsed.question,
        "known_symbols": sorted(parsed.knowns),
        "selected_formula": {
            "id": formula.id,
            "equation": formula.equation,
            "expression": formula.expression,
            "intermediates": formula.intermediates,
            "target_variables": formula.target_variables,
        },
    }
    try:
        raw = model_complete(
            CODE_PLAN_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False),
        )
        parsed_json = parse_json_object(raw)
    except Exception:
        return None
    raw_steps = parsed_json.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    steps: list[PlanStep] = []
    available = set(parsed.knowns)
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            return None
        target = str(raw_step.get("target", "")).strip()
        expression = str(raw_step.get("expression", "")).strip()
        if not target or not expression or not expression_is_allowed(expression, available):
            return None
        available.add(target)
        steps.append(
            PlanStep(
                formula_id=formula.id,
                formula=formula.equation,
                substitution=substitute_values(expression, parsed),
                target=target,
                expression=expression,
                source="slm",
            )
        )
    if steps[-1].target not in formula.target_variables:
        return None
    return steps


def build_formula_steps(parsed: ParsedProblem, formula: Formula) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for target, expression in formula.intermediates:
        steps.append(make_step(parsed, formula, target, expression, "formula_bank"))
    steps.append(
        make_step(
            parsed,
            formula,
            parsed.target if parsed.target in formula.target_variables else formula.target_variables[0],
            formula.expression,
            "formula_bank",
        )
    )
    return steps


def make_step(
    parsed: ParsedProblem,
    formula: Formula,
    target: str,
    expression: str,
    source: str,
) -> PlanStep:
    return PlanStep(
        formula_id=formula.id,
        formula=formula.equation,
        substitution=substitute_values(expression, parsed),
        target=target,
        expression=expression,
        source=source,
    )


def substitute_values(expression: str, parsed: ParsedProblem) -> str:
    values = {symbol: parsed.knowns[symbol].si_value for symbol in parsed.knowns}
    for symbol in sorted(values, key=len, reverse=True):
        expression = expression.replace(symbol, str(values[symbol]))
    return expression


def expression_is_allowed(expression: str, available_symbols: set[str]) -> bool:
    allowed_functions = {"sqrt", "sin", "cos", "tan", "Abs", "pi"}
    found = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
    if found - available_symbols - allowed_functions:
        return False
    return re.search(r"[^A-Za-z0-9_+\-*/().,\s]", expression) is None and "__" not in expression


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed: Any = json.loads(raw)
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
