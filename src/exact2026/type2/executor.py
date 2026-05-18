"""Execute Type 2 equation plans through restricted SymPy expressions."""

from __future__ import annotations

import math
import re

import sympy as sp

from .formula_bank import get_formula
from .schemas import EquationPlan, ExecutionResult, ParsedProblem


ALLOWED_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "Abs": sp.Abs,
    "pi": sp.pi,
}


def execute_plan(plan: EquationPlan, parsed: ParsedProblem) -> ExecutionResult:
    try:
        formula = get_formula(plan.formula_id)
        if formula is None:
            raise ValueError(f"Unknown formula id: {plan.formula_id}")
        return execute_steps(plan, parsed, formula.output_unit)
    except ZeroDivisionError as exc:
        return ExecutionResult(False, None, "", {}, [], f"Divide by zero: {exc}")
    except Exception as exc:
        return ExecutionResult(False, None, "", {}, [], str(exc))


def execute_steps(plan: EquationPlan, parsed: ParsedProblem, output_unit: str) -> ExecutionResult:
    values = {symbol: known.si_value for symbol, known in parsed.knowns.items()}
    intermediates: dict[str, float] = {}
    trace: list[str] = []

    context = dict(values)
    answer: float | None = None
    for step in plan.steps:
        answer = evaluate_expression(step.expression, context)
        context[step.target] = answer
        if step is not plan.steps[-1]:
            intermediates[step.target] = answer
        trace.append(f"{step.target} = {step.expression} = {answer:g}")

    if answer is None:
        raise ValueError("Equation plan has no executable steps.")
    if not math.isfinite(answer):
        raise ValueError("Expression produced a non-finite result.")

    return ExecutionResult(True, answer, output_unit, intermediates, trace)


def evaluate_expression(expression: str, values: dict[str, float]) -> float:
    assert_safe_expression(expression, values)
    locals_map = {name: sp.Float(value) for name, value in values.items()}
    locals_map.update(ALLOWED_FUNCTIONS)
    parsed = sp.sympify(expression, locals=locals_map)
    unresolved = parsed.free_symbols
    if unresolved:
        names = ", ".join(sorted(str(symbol) for symbol in unresolved))
        raise ValueError(f"Expression has unresolved symbols: {names}")
    result = float(parsed.evalf())
    if not math.isfinite(result):
        raise ValueError("Expression result is not finite.")
    return result


def assert_safe_expression(expression: str, values: dict[str, float]) -> None:
    if "__" in expression:
        raise ValueError("Unsafe expression token.")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
    allowed_names = set(values) | set(ALLOWED_FUNCTIONS)
    unknown = names - allowed_names
    if unknown:
        raise ValueError(f"Expression uses unknown names: {', '.join(sorted(unknown))}")
    if re.search(r"[^A-Za-z0-9_+\-*/().,\s]", expression):
        raise ValueError("Expression contains unsupported characters.")
