"""Tools used by the Type 2 LangGraph agents."""

from __future__ import annotations

import ast
import math
from typing import Any

import sympy as sp

from .knowledge_search import formula_search_tool


ALLOWED_IMPORTS = {"math", "sympy"}
ALLOWED_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
    "sum": sum,
    "str": str,
    "float": float,
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
        if isinstance(root, ast.Name) and root.id in {"steps", "premises", "trace"} and node.func.attr == "append":
            return
        if isinstance(root, ast.Name) and root.id in {"math", "sp", "sympy"}:
            return
    raise ValueError("Only allowlisted builtin/math/sympy calls are executable.")


def limited_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    if name == "math":
        return math
    if name == "sympy":
        return sp
    raise ImportError(f"Import not allowed: {name}")
