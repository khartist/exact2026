"""Planner LangChain agent for structured Type 2 calculation contracts."""

from __future__ import annotations

from typing import Any

from ..knowledge_search import build_planner_knowledge_context
from ..prompts import load_prompt
from ..state import Type2State
from .common import invoke_json_agent


def planner_agent(state: Type2State, llm: Any) -> dict[str, Any]:
    question = state["question"]
    search_query = question
    knowledge = build_planner_knowledge_context(search_query, state.get("knowledge_config"))
    attempts = state.get("planner_attempts", 0) + 1
    payload = {
        "question": question,
        "search_query": search_query,
        "physics_knowledge": knowledge,
        "previous_plan": state.get("plan"),
        "validator_errors": state.get("planner_validation").errors
        if state.get("planner_validation")
        else [],
        "attempt": attempts,
    }
    plan = invoke_json_agent(llm, load_prompt("planner.txt"), payload)
    return {
        "search_query": search_query,
        "knowledge": knowledge,
        "plan": plan,
        "planner_attempts": attempts,
    }
