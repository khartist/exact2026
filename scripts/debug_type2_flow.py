#!/usr/bin/env python3
"""Print Type 2 structured solver node inputs and outputs for one question."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exact2026.type2.agents.code_generator_agent import code_generator_agent
from exact2026.type2.agents.planner_agent import planner_agent
from exact2026.type2.graph import (
    answer_composer_node,
    code_executor_node,
    execution_validator_node,
    fallback_node,
    planner_validator_node,
    route_from_execution_validator,
    route_from_planner_validator,
)
from exact2026.type2.knowledge_search import search_physics_knowledge
from exact2026.type2.llm import build_type2_llm
from exact2026.type2.schemas import Type2SolverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug one Type 2 question by printing every graph node input/output."
    )
    parser.add_argument("--question", required=True, help="Physics question to inspect.")
    parser.add_argument("--use-ollama", action="store_true", help="Use the configured Ollama model.")
    parser.add_argument("--model", default="gemma4:e2b-it-q4_K_M", help="Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat", help="Ollama URL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm = (
        build_type2_llm(
            Type2SolverConfig(
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        )
        if args.use_ollama
        else None
    )
    state: dict[str, Any] = {
        "question": args.question,
        "planner_attempts": 0,
        "code_attempts": 0,
    }
    print_section("raw_question", args.question)
    print_section("search_query", args.question)
    print_section("formula_law_results", search_physics_knowledge(args.question, top_k=8))

    for _ in range(2):
        run_node("planner_agent", state, lambda current: planner_agent(current, llm))
        run_node("planner_validator", state, planner_validator_node)
        route = route_from_planner_validator(state)
        print_section("planner_route", route)
        if route == "code_generator_agent":
            break
        if route == "fallback":
            run_node("fallback", state, fallback_node)
            return 0

    for _ in range(2):
        run_node("code_generator_agent", state, lambda current: code_generator_agent(current, llm))
        run_node("code_executor", state, code_executor_node)
        run_node("execution_validator", state, execution_validator_node)
        route = route_from_execution_validator(state)
        print_section("execution_route", route)
        if route == "answer_composer":
            run_node("answer_composer", state, answer_composer_node)
            return 0
        if route == "fallback":
            run_node("fallback", state, fallback_node)
            return 0

    run_node("fallback", state, fallback_node)
    return 0


def run_node(name: str, state: dict[str, Any], fn: Any) -> None:
    print_section(f"{name}.input", state)
    output = fn(state)
    print_section(f"{name}.output", output)
    state.update(output)


def print_section(name: str, value: Any) -> None:
    print(f"\n===== {name} =====")
    print(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
