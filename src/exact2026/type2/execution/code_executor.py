"""Safe execution wrapper for structured Type 2 generated code."""

from __future__ import annotations

from typing import Any

from ..tools import ALLOWED_BUILTINS, SAFE_MATH_EXPORTS, SAFE_SYMPY_EXPORTS, limited_import, validate_generated_code


def execute_structured_code(
    code_payload: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    code = str(code_payload.get("code", "")).strip()
    if not code:
        return failed("Generated payload did not include code.", code)
    try:
        validate_generated_code(code)
        env = build_plan_execution_env(plan)
        exec(compile(code, "<type2-structured-code>", "exec"), env, env)
        trace = env.get("trace")
        final = env.get("final")
        if not isinstance(trace, list):
            raise ValueError('Generated code must set "trace" to a list.')
        if not isinstance(final, dict):
            raise ValueError('Generated code must set "final" to a dict.')
        return {
            "status": "SUCCESS",
            "code": code,
            "trace": trace,
            "final": final,
            "error": None,
        }
    except Exception as exc:
        return failed(str(exc), code)


def build_plan_execution_env(plan: dict[str, Any]) -> dict[str, Any]:
    import math
    import sympy as sp

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
    givens = plan.get("givens", {})
    if isinstance(givens, dict):
        for symbol, raw in givens.items():
            if isinstance(raw, dict) and isinstance(raw.get("si_value"), (int, float)):
                env[str(symbol)] = float(raw["si_value"])
    return env


def failed(error: str, code: str) -> dict[str, Any]:
    return {
        "status": "ERROR",
        "code": code,
        "trace": [],
        "final": {},
        "error": error,
    }
