"""Tools used by the Type 2 LangGraph agents."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

import sympy as sp

from .formula_bank import FORMULAS
from .schemas import ExecutionResult, Formula, ParsedProblem


ALLOWED_IMPORTS = {"math", "sympy"}
ALLOWED_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
    "sum": sum,
    "str": str,
    "sqrt": math.sqrt,
    "Abs": abs,
}
SAFE_MATH_EXPORTS = {
    name: value
    for name, value in vars(math).items()
    if not name.startswith("_")
}
SAFE_SYMPY_EXPORTS = {
    name: value
    for name, value in vars(sp).items()
    if not name.startswith("_")
}
FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
}


def formula_bank_tool(parsed: ParsedProblem, limit: int = 8) -> list[Formula]:
    known_symbols = set(parsed.knowns)
    question_words = set(parsed.question.lower().replace("-", " ").split())
    scored: list[tuple[int, Formula]] = []
    for formula in FORMULAS:
        score = 0
        available_count = len(set(formula.required_variables).intersection(known_symbols))
        if set(formula.required_variables).issubset(known_symbols):
            score += 20
        score += available_count * 3
        if parsed.target and parsed.target in formula.target_variables:
            score += 8
        if parsed.topic and parsed.topic == formula.topic:
            score += 6
        if formula.id == "ac_real_power_vzr" and {"V", "Z", "R"}.issubset(known_symbols):
            score += 18
        if formula.id == "power_v2r" and "Z" in known_symbols:
            score -= 10
        if formula.id == "net_coulomb_force_right_triangle" and {"q1", "q2", "q3", "AC", "BC"}.issubset(known_symbols):
            score += 18
        score += len(question_words.intersection(formula.keywords))
        if score > 0:
            scored.append((score, formula))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [formula for _, formula in scored[:limit]]


def execute_generated_code_tool(
    code: str,
    parsed: ParsedProblem,
) -> ExecutionResult:
    try:
        validate_generated_code(code)
        env = build_execution_env(parsed)
        exec(compile(code, "<type2-generated-code>", "exec"), env, env)
        raw_answer = env.get("answer")
        if raw_answer is None:
            raise ValueError('Generated code must set "answer".')
        answer = float(raw_answer)
        unit = str(env.get("unit", "")).strip()
        steps = normalize_string_list(env.get("steps"))
        premises = normalize_string_list(env.get("premises"))
        trace = steps or [f"Generated code computed answer = {answer:g} {unit}".strip()]
        intermediates = {
            key: float(value)
            for key, value in env.items()
            if is_public_number(key, value) and key not in parsed.knowns and key != "answer"
        }
        if premises:
            trace.extend(f"Premise: {premise}" for premise in premises)
        return ExecutionResult(True, answer, unit, intermediates, trace)
    except Exception as exc:
        return ExecutionResult(False, None, "", {}, [], str(exc))


def validate_generated_code(code: str) -> None:
    if "__" in code:
        raise ValueError("Generated code contains forbidden dunder token.")
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            raise ValueError(f"Unsupported generated-code node: {type(node).__name__}.")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import not allowed: {alias.name}.")
        if isinstance(node, ast.Call):
            validate_call(node)
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name: {node.id}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Forbidden private attribute: {node.attr}.")


def validate_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name):
        if (
            node.func.id not in ALLOWED_BUILTINS
            and node.func.id not in SAFE_MATH_EXPORTS
            and node.func.id not in SAFE_SYMPY_EXPORTS
        ):
            raise ValueError(f"Call not allowed: {node.func.id}.")
        return
    if isinstance(node.func, ast.Attribute):
        root = node.func.value
        if isinstance(root, ast.Name) and root.id in {"steps", "premises"} and node.func.attr == "append":
            return
        if isinstance(root, ast.Name) and root.id in {"math", "sp", "sympy"}:
            return
    raise ValueError("Only allowlisted builtin/math/sympy calls are executable.")


def build_execution_env(parsed: ParsedProblem) -> dict[str, Any]:
    env: dict[str, Any] = {
        "__builtins__": {**ALLOWED_BUILTINS, "__import__": limited_import},
        "math": math,
        "sympy": sp,
        "sp": sp,
        **SAFE_MATH_EXPORTS,
        **SAFE_SYMPY_EXPORTS,
        "pi": math.pi,
        "sqrt": math.sqrt,
        "Abs": abs,
    }
    for symbol, known in parsed.knowns.items():
        env[symbol] = known.si_value
    return env


def limited_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    if name == "math":
        return math
    if name == "sympy":
        return sp
    raise ImportError(f"Import not allowed: {name}")


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def is_public_number(key: str, value: Any) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key)) and isinstance(value, (int, float))
