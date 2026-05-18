"""Formula-aware verification for Type 2 execution results."""

from __future__ import annotations

import math
import re
from typing import Any

import sympy as sp

from .executor import ALLOWED_FUNCTIONS, evaluate_expression
from .formula_bank import get_formula
from .schemas import EquationPlan, ExecutionResult, Formula, ParsedProblem, PlanStep, VerificationResult


REL_TOL = 1e-3
ABS_TOL = 1e-6


def backward_consistency_check(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
) -> VerificationResult:
    errors = basic_execution_errors(parsed, execution)
    formula = get_formula(plan.formula_id)
    if formula is None:
        errors.append(f"Unknown formula id: {plan.formula_id}.")
    elif execution.answer is not None and execution.success:
        errors.extend(verify_forward_calculation(parsed, formula, execution.answer))
        errors.extend(verify_steps_backward(parsed, plan, execution))

    return VerificationResult(not errors, 0.0, errors)


def verify_execution(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
) -> VerificationResult:
    return backward_consistency_check(parsed, plan, execution)


def basic_execution_errors(
    parsed: ParsedProblem,
    execution: ExecutionResult,
) -> list[str]:
    errors: list[str] = []
    if not execution.success:
        errors.append(execution.error or "Execution failed.")
    if execution.answer is None:
        errors.append("Answer is missing.")
    elif not math.isfinite(execution.answer):
        errors.append("Answer is not finite.")
    if not execution.unit:
        errors.append("Unit is missing.")
    if parsed.target_unit and execution.unit and parsed.target_unit != execution.unit:
        errors.append(f"Expected unit {parsed.target_unit}, got {execution.unit}.")
    if execution.answer is not None and abs(execution.answer) > 1e30:
        errors.append("Answer magnitude is implausibly large.")
    return errors


def verify_steps_backward(
    parsed: ParsedProblem,
    plan: EquationPlan,
    execution: ExecutionResult,
) -> list[str]:
    if execution.answer is None:
        return ["Cannot verify steps backward without an answer."]
    errors: list[str] = []
    context = {symbol: known.si_value for symbol, known in parsed.knowns.items()}
    context.update(execution.intermediates)
    if plan.steps:
        context[plan.steps[-1].target] = execution.answer

    for step in reversed(plan.steps):
        target_value = context.get(step.target)
        if target_value is None:
            errors.append(f"Backward review missing value for step target {step.target}.")
            continue
        errors.extend(reproduce_step_inputs(parsed, step, context, target_value))
    return errors


def reproduce_step_inputs(
    parsed: ParsedProblem,
    step: PlanStep,
    context: dict[str, float],
    target_value: float,
) -> list[str]:
    errors: list[str] = []
    symbol_names = extract_expression_names(step.expression) - set(ALLOWED_FUNCTIONS)
    for variable in symbol_names:
        if variable not in parsed.knowns:
            continue
        variable_symbol = sp.Symbol(variable)
        substituted = build_step_expression(step.expression, context, keep_symbol=variable)
        original = parsed.knowns[variable].si_value
        reproduced = reproduce_variable_numeric(
            substituted,
            variable_symbol,
            sp.Float(target_value),
            original,
        )
        if reproduced is None:
            errors.append(
                f"Backward review could not reproduce {variable} from step {step.target}."
            )
        elif not close_enough(reproduced, original):
            errors.append(
                f"Backward review mismatch for {variable}: expected {original}, got {reproduced}."
            )
    return errors


def build_step_expression(
    expression: str,
    context: dict[str, float],
    keep_symbol: str,
) -> sp.Expr:
    expression = normalize_namespace_prefixes(expression)
    names = extract_expression_names(expression)
    locals_map: dict[str, Any] = {
        name: sp.Symbol(name)
        for name in names
        if name not in ALLOWED_FUNCTIONS and (name not in context or name == keep_symbol)
    }
    locals_map.update(
        {
            name: sp.Float(value)
            for name, value in context.items()
            if name in names and name != keep_symbol
        }
    )
    locals_map.update(ALLOWED_FUNCTIONS)
    locals_map.update({"math": math, "sympy": sp, "sp": sp})
    return sp.sympify(expression, locals=locals_map)


def extract_expression_names(expression: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))


def normalize_namespace_prefixes(expression: str) -> str:
    return re.sub(r"\b(?:math|sympy|sp)\.", "", expression)


def verify_forward_calculation(
    parsed: ParsedProblem,
    formula: Formula,
    answer: float,
) -> list[str]:
    try:
        recomputed = recompute_formula(parsed, formula)
    except Exception as exc:
        return [f"Forward recomputation failed: {exc}."]
    if not close_enough(recomputed, answer):
        return [f"Forward recomputation mismatch: got {recomputed}, answer is {answer}."]
    return []


def recompute_formula(parsed: ParsedProblem, formula: Formula) -> float:
    context = {symbol: known.si_value for symbol, known in parsed.knowns.items()}
    for name, expression in formula.intermediates:
        context[name] = evaluate_expression(expression, context)
    return evaluate_expression(formula.expression, context)


def verify_inverse_reproduction(
    parsed: ParsedProblem,
    formula: Formula,
    answer: float,
) -> list[str]:
    errors: list[str] = []
    expression = build_symbolic_expression(formula)
    symbols = sorted(extract_symbol_names(formula), key=len, reverse=True)
    symbol_map = {name: sp.Symbol(name) for name in symbols}
    target = sp.Float(answer)

    for variable in formula.required_variables:
        if variable not in parsed.knowns:
            continue
        original = parsed.knowns[variable].si_value
        substituted = expression
        for name, symbol in symbol_map.items():
            if name == variable:
                continue
            if name in parsed.knowns:
                substituted = substituted.subs(symbol, sp.Float(parsed.knowns[name].si_value))

        reproduced = reproduce_variable_numeric(
            substituted,
            symbol_map[variable],
            target,
            original,
        )
        if reproduced is None:
            errors.append(f"Could not reproduce {variable}: no numeric solution near original value.")
            continue
        if not close_enough(reproduced, original):
            errors.append(
                f"Reproduced {variable} mismatch: expected {original}, "
                f"got {reproduced}."
            )
    return errors


def build_symbolic_expression(formula: Formula) -> sp.Expr:
    names = extract_symbol_names(formula)
    locals_map: dict[str, Any] = {name: sp.Symbol(name) for name in names}
    locals_map.update(ALLOWED_FUNCTIONS)
    intermediate_exprs: dict[str, sp.Expr] = {}
    for name, expression in formula.intermediates:
        intermediate_exprs[name] = sp.sympify(expression, locals=locals_map)
        locals_map[name] = intermediate_exprs[name]
    return sp.sympify(formula.expression, locals=locals_map)


def extract_symbol_names(formula: Formula) -> set[str]:
    text = " ".join(
        [formula.expression, *(expression for _, expression in formula.intermediates)]
    )
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return names - set(ALLOWED_FUNCTIONS)


def reproduce_variable_numeric(
    expression: sp.Expr,
    variable: sp.Symbol,
    target: sp.Float,
    original: float,
) -> float | None:
    residual = expression - target
    try:
        fn = sp.lambdify(variable, residual, "math")
        original_residual = float(fn(original))
    except Exception:
        return None
    if math.isfinite(original_residual) and abs(original_residual) <= max(ABS_TOL, abs(float(target)) * REL_TOL):
        return original

    for low, high in numeric_brackets(original):
        try:
            low_value = float(fn(low))
            high_value = float(fn(high))
        except Exception:
            continue
        if not all(math.isfinite(value) for value in (low_value, high_value)):
            continue
        if abs(low_value) <= ABS_TOL:
            return low
        if abs(high_value) <= ABS_TOL:
            return high
        if low_value * high_value > 0:
            continue
        solved = bisect_root(fn, low, high)
        if solved is not None:
            return solved
    return None


def numeric_brackets(original: float) -> list[tuple[float, float]]:
    if original == 0:
        return [(-1.0, 1.0), (-1e-6, 1e-6)]
    width = max(abs(original) * 0.5, 1e-12)
    return [
        (original - width, original + width),
        (original * 0.9, original * 1.1),
        (original * 0.5, original * 1.5),
    ]


def bisect_root(fn: Any, low: float, high: float) -> float | None:
    low_value = float(fn(low))
    high_value = float(fn(high))
    for _ in range(40):
        mid = (low + high) / 2
        mid_value = float(fn(mid))
        if abs(mid_value) <= ABS_TOL:
            return mid
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    candidate = (low + high) / 2
    return candidate if math.isfinite(candidate) else None


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def verify_llm_fallback(answer: str, unit: str) -> VerificationResult:
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
    confidence = 0.55 if not errors else 0.3
    return VerificationResult(not errors, confidence, errors)
