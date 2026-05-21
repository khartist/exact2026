"""Structured LangGraph workflow for Type 2 physics solving."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .agents.code_generator_agent import code_generator_agent
from .agents.planner_agent import planner_agent
from .composer import compose_answer, compose_direct_answer
from .execution.code_executor import execute_structured_code
from .schemas import KnowledgeSearchConfig, PipelineResult
from .state import Type2State
from .validation.execution_validator import validate_structured_execution
from .validation.planner_validator import validate_calculation_plan

MAX_PLANNER_ATTEMPTS = 2
MAX_CODE_ATTEMPTS = 2


def solve_with_structured_graph(
    question: str,
    llm: Any = None,
    knowledge_config: KnowledgeSearchConfig | None = None,
) -> PipelineResult:
    graph = build_type2_graph(llm)
    final_state = graph.invoke(
        {
            "question": question,
            "planner_attempts": 0,
            "code_attempts": 0,
            "knowledge_config": knowledge_config,
        }
    )
    result = final_state.get("result")
    if isinstance(result, PipelineResult):
        return result
    return fallback_result(question, "The Type 2 graph ended without a result.")


def build_type2_graph(llm: Any = None):
    workflow = StateGraph(Type2State)
    workflow.add_node("planner_agent", lambda state: planner_agent(state, llm))
    workflow.add_node("planner_validator", planner_validator_node)
    workflow.add_node("code_generator_agent", lambda state: code_generator_agent(state, llm))
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("execution_validator", execution_validator_node)
    workflow.add_node("answer_composer", answer_composer_node)
    workflow.add_node("direct_answer_composer", direct_answer_composer_node)
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "planner_agent")
    workflow.add_edge("planner_agent", "planner_validator")
    workflow.add_conditional_edges(
        "planner_validator",
        route_from_planner_validator,
        {
            "code_generator_agent": "code_generator_agent",
            "direct_answer_composer": "direct_answer_composer",
            "planner_agent": "planner_agent",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("code_generator_agent", "code_executor")
    workflow.add_edge("code_executor", "execution_validator")
    workflow.add_conditional_edges(
        "execution_validator",
        route_from_execution_validator,
        {
            "answer_composer": "answer_composer",
            "code_generator_agent": "code_generator_agent",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("answer_composer", END)
    workflow.add_edge("direct_answer_composer", END)
    workflow.add_edge("fallback", END)
    return workflow.compile()


def planner_validator_node(state: Type2State) -> dict[str, Any]:
    validation = validate_calculation_plan(state.get("plan", {}))
    update: dict[str, Any] = {"planner_validation": validation}
    if not validation.ok and state.get("planner_attempts", 0) >= MAX_PLANNER_ATTEMPTS:
        update["fallback_reason"] = validation.repair_prompt or "Planner validation failed."
    return update


def code_executor_node(state: Type2State) -> dict[str, Any]:
    execution = execute_structured_code(state.get("code_payload", {}), state["plan"])
    return {"execution": execution}


def execution_validator_node(state: Type2State) -> dict[str, Any]:
    validation = validate_structured_execution(state["plan"], state.get("execution", {}))
    update: dict[str, Any] = {"execution_validation": validation}
    if not validation.ok and state.get("code_attempts", 0) >= MAX_CODE_ATTEMPTS:
        update["fallback_reason"] = validation.repair_prompt or "Execution validation failed."
    return update


def answer_composer_node(state: Type2State) -> dict[str, Any]:
    execution = state["execution"]
    trace = execution.get("trace", [])
    if trace and isinstance(trace[-1], dict):
        trace[-1]["is_final"] = True
    result = compose_answer(
        state["question"],
        state["plan"],
        execution,
        metadata={
            "formula_id": final_formula_id(state["plan"]),
            "formula_ids": formula_ids(state["plan"]),
            "planner_attempts": state.get("planner_attempts", 0),
            "code_attempts": state.get("code_attempts", 0),
            "planner_validation_errors": state.get("planner_validation").errors
            if state.get("planner_validation")
            else [],
            "execution_validation_errors": state.get("execution_validation").errors
            if state.get("execution_validation")
            else [],
        },
    )
    return {"result": result}


def direct_answer_composer_node(state: Type2State) -> dict[str, Any]:
    result = compose_direct_answer(
        state["question"],
        state["plan"],
        metadata={
            "planner_attempts": state.get("planner_attempts", 0),
            "planner_validation_errors": state.get("planner_validation").errors
            if state.get("planner_validation")
            else [],
        },
    )
    return {"result": result}


def fallback_node(state: Type2State) -> dict[str, Any]:
    return {
        "result": fallback_result(
            state["question"],
            state.get("fallback_reason", "Structured Type 2 solving failed."),
        )
    }


def route_from_planner_validator(
    state: Type2State,
) -> Literal["code_generator_agent", "direct_answer_composer", "planner_agent", "fallback"]:
    validation = state["planner_validation"]
    if validation.ok:
        if str(state.get("plan", {}).get("status", "")).strip() == "DIRECT_ANSWER":
            return "direct_answer_composer"
        return "code_generator_agent"
    if state.get("planner_attempts", 0) < MAX_PLANNER_ATTEMPTS:
        return "planner_agent"
    return "fallback"


def route_from_execution_validator(
    state: Type2State,
) -> Literal["answer_composer", "code_generator_agent", "fallback"]:
    validation = state["execution_validation"]
    if validation.ok:
        return "answer_composer"
    if state.get("code_attempts", 0) < MAX_CODE_ATTEMPTS:
        return "code_generator_agent"
    return "fallback"


def fallback_result(question: str, reason: str) -> PipelineResult:
    return PipelineResult(
        answer="",
        unit="",
        explanation=reason,
        cot=["Structured planner/code execution did not produce a validated answer."],
        premises=[],
        metadata={
            "agent_loop": "structured_langgraph",
            "verified": False,
            "fallback": "structured_failure",
            "question": question,
        },
    )


def formula_ids(plan: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        formula_id = str(step.get("formula_id", "")).strip()
        if formula_id and formula_id not in ids:
            ids.append(formula_id)
    return ids


def final_formula_id(plan: dict[str, Any]) -> str:
    ids = formula_ids(plan)
    return ids[-1] if ids else ""
