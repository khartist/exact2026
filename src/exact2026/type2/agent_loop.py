"""LangGraph agent loop for Type 2 physics solving."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .explanation import build_explanation
from .formula_bank import get_formula
from .parser import ModelCompleteFn, parse_question
from .pipeline import FallbackModelFn, format_number, solve_with_llm_fallback
from .planner import parse_json_object
from .schemas import (
    EquationPlan,
    ExecutionResult,
    Formula,
    ParsedProblem,
    PipelineResult,
    PlanStep,
    VerificationResult,
)
from .tools import execute_generated_code_tool, formula_bank_tool
from .verifier import backward_consistency_check

logger = logging.getLogger(__name__)

MAX_LOOPS = 3

PLANNER_AGENT_PROMPT = """You are the planner agent for a physics QA solver.
You receive the current question and the formula_bank tool output as candidate formulas.
Generate a problem-specific ordered sequence of steps for this exact question.
Do not use generic template steps.
Return strict JSON only:
{"action":"formula_bank|custom_code|fallback|EXIT","formula_id":"optional id","steps":["step 1","step 2"],"reason":"short reason"}
The steps list must contain several concrete, problem-specific steps when solving is possible.
Choose fallback only if solving is impossible with candidates or generated code."""

CODE_GENERATOR_PROMPT = """You are the code generator agent for a physics QA solver.
Write executable Python code that computes the answer from the provided SI variables.
Follow the planner_steps in order when writing the code.
Allowed imports: math, sympy. Set variables: answer, unit, steps, premises.
Do not read files, use network, define functions/classes, or use markdown.
Return strict JSON only: {"code":"...python code..."}"""

REVIEWER_AGENT_PROMPT = """You are the reviewer agent for a physics QA solver.
Review the question, planner steps, generated code trace, backward consistency issues, and answer.
Return strict JSON only:
{"passed":true/false,"confidence":0.0-1.0,"errors":["issue"],"feedback":"short feedback"}"""


class AgentState(TypedDict, total=False):
    question: str
    parsed: ParsedProblem
    candidates: list[Formula]
    formula_id: str
    planner_steps: list[str]
    code: str
    plan: EquationPlan
    execution: ExecutionResult
    review: VerificationResult
    review_source: str
    result: PipelineResult
    loop_count: int
    decision: str
    errors: list[str]
    feedback: str


def solve_with_agent_loop(
    question: str,
    model_complete: ModelCompleteFn | None = None,
    fallback_model: FallbackModelFn | None = None,
) -> PipelineResult:
    graph = build_type2_graph(model_complete, fallback_model)
    final_state = graph.invoke({"question": question, "loop_count": 0, "errors": []})
    result = final_state.get("result")
    if isinstance(result, PipelineResult):
        return result
    return PipelineResult(
        answer="",
        unit="",
        explanation="The Type 2 agent loop ended without a final result.",
        cot=["Planner did not produce an executable solution."],
        premises=[],
        confidence=0.2,
        metadata={"agent_loop": "no_result"},
    )


def build_type2_graph(
    model_complete: ModelCompleteFn | None,
    fallback_model: FallbackModelFn | None,
):
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", lambda state: planner_agent(state, model_complete))
    workflow.add_node(
        "code_generator",
        lambda state: code_generator_agent(state, model_complete, fallback_model),
    )
    workflow.add_node("reviewer", lambda state: reviewer_agent(state, model_complete))
    workflow.add_node("fallback", lambda state: fallback_agent(state, fallback_model))
    workflow.add_node("finalizer", lambda state: finalizer_agent(state, model_complete))
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "code_generator": "code_generator",
            "fallback": "fallback",
            "finalizer": "finalizer",
        },
    )
    workflow.add_conditional_edges(
        "code_generator",
        route_from_code_generator,
        {
            "reviewer": "reviewer",
            "planner": "planner",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("reviewer", "planner")
    workflow.add_edge("fallback", END)
    workflow.add_edge("finalizer", END)
    return workflow.compile()


def planner_agent(
    state: AgentState,
    model_complete: ModelCompleteFn | None,
) -> dict[str, Any]:
    question = state["question"]
    parsed = state.get("parsed") or parse_question(question, model_complete)
    candidates = formula_bank_tool(parsed)
    review = state.get("review")
    loop_count = state.get("loop_count", 0)

    if review and review.passed and state.get("execution"):
        return {"parsed": parsed, "candidates": candidates, "decision": "finalize"}
    if review and not state.get("execution") and state.get("loop_count", 0) > 0:
        return {"parsed": parsed, "candidates": candidates, "decision": "fallback"}
    if loop_count >= MAX_LOOPS:
        return {
            "parsed": parsed,
            "candidates": candidates,
            "decision": "finalize" if state.get("execution") else "fallback",
        }

    decision = planner_model_decision(parsed, candidates, state, model_complete)
    if decision["action"] == "fallback":
        return {"parsed": parsed, "candidates": candidates, "decision": "fallback"}
    if decision["action"] == "EXIT" and state.get("execution"):
        return {"parsed": parsed, "candidates": candidates, "decision": "finalize"}
    if decision["action"] == "custom_code":
        return {
            "parsed": parsed,
            "candidates": candidates,
            "formula_id": "",
            "planner_steps": decision.get("steps", []),
            "loop_count": loop_count + 1,
            "decision": "code_generator",
            "feedback": decision.get("reason", ""),
        }

    formula_id = decision.get("formula_id") or (candidates[0].id if candidates else "")
    if not formula_id:
        return {"parsed": parsed, "candidates": candidates, "decision": "fallback"}
    return {
        "parsed": parsed,
        "candidates": candidates,
        "formula_id": formula_id,
        "planner_steps": decision.get("steps", []),
        "loop_count": loop_count + 1,
        "decision": "code_generator",
        "feedback": decision.get("reason", ""),
    }


def planner_model_decision(
    parsed: ParsedProblem,
    candidates: list[Formula],
    state: AgentState,
    model_complete: ModelCompleteFn | None,
) -> dict[str, Any]:
    if model_complete is None:
        formula = candidates[0] if strong_candidate_exists(parsed, candidates) else None
        return {
            "action": "formula_bank" if formula else "custom_code",
            "formula_id": formula.id if formula else "",
            "steps": [],
            "reason": "Model unavailable; planner step list left for code generation.",
        }
    payload = {
        "question": parsed.question,
        "knowns": {
            symbol: {"value": known.si_value, "unit": known.si_unit}
            for symbol, known in parsed.knowns.items()
        },
        "target": parsed.target,
        "target_unit": parsed.target_unit,
        "formula_bank_candidates": [
            {
                "id": item.id,
                "equation": item.equation,
                "required_variables": item.required_variables,
                "target_variables": item.target_variables,
                "description": item.description,
            }
            for item in candidates
        ],
        "previous_review_errors": state.get("review").errors
        if state.get("review")
        else [],
        "loop_count": state.get("loop_count", 0),
    }
    try:
        raw = model_complete(
            PLANNER_AGENT_PROMPT, json.dumps(payload, ensure_ascii=False)
        )
        parsed_json = parse_json_object(raw)
    except Exception:
        parsed_json = {}
    action = str(parsed_json.get("action", "")).strip()
    formula_id = str(parsed_json.get("formula_id", "")).strip()
    planner_steps = normalize_planner_steps(parsed_json.get("steps"))
    allowed_ids = {item.id for item in candidates}
    if action == "formula_bank" and formula_id in allowed_ids:
        formula = next(item for item in candidates if item.id == formula_id)
        if not planner_steps:
            planner_steps = request_planner_steps_retry(
                parsed, candidates, model_complete, action, formula_id
            )
        return {
            "action": action,
            "formula_id": formula_id,
            "steps": planner_steps,
            "reason": str(parsed_json.get("reason", "")),
        }
    if action == "custom_code":
        if not planner_steps:
            planner_steps = request_planner_steps_retry(
                parsed, candidates, model_complete, action, formula_id
            )
        return {
            "action": "custom_code",
            "steps": planner_steps,
            "reason": str(parsed_json.get("reason", "")),
        }
    if action == "EXIT":
        return {
            "action": "EXIT",
            "steps": planner_steps,
            "reason": str(parsed_json.get("reason", "")),
        }
    if strong_candidate_exists(parsed, candidates):
        return {
            "action": "formula_bank",
            "formula_id": candidates[0].id,
            "steps": planner_steps
            or request_planner_steps_retry(
                parsed,
                candidates,
                model_complete,
                "formula_bank",
                candidates[0].id,
            ),
        }
    return {
        "action": "custom_code",
        "steps": planner_steps,
        "reason": str(parsed_json.get("reason", "")),
    }


def normalize_planner_steps(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def strong_candidate_exists(parsed: ParsedProblem, candidates: list[Formula]) -> bool:
    if not candidates:
        return False
    best = candidates[0]
    known_symbols = set(parsed.knowns)
    return set(best.required_variables).issubset(known_symbols)


def request_planner_steps_retry(
    parsed: ParsedProblem,
    candidates: list[Formula],
    model_complete: ModelCompleteFn | None,
    action: str,
    formula_id: str,
) -> list[str]:
    if model_complete is None:
        return []
    retry_payload = {
        "question": parsed.question,
        "target": parsed.target,
        "knowns": {
            symbol: {"value": known.si_value, "unit": known.si_unit}
            for symbol, known in parsed.knowns.items()
        },
        "formula_bank_candidates": [
            {
                "id": item.id,
                "equation": item.equation,
                "required_variables": item.required_variables,
                "target_variables": item.target_variables,
                "description": item.description,
            }
            for item in candidates
        ],
        "selected_action": action,
        "selected_formula_id": formula_id,
        "instruction": (
            "Return 3 to 10 problem-specific ordered steps for this exact question. "
            "Do not use generic templates. Do not omit the step list."
        ),
    }
    try:
        raw = model_complete(
            PLANNER_AGENT_PROMPT
            + "\nYou previously returned no usable step list. Return a detailed step list now.",
            json.dumps(retry_payload, ensure_ascii=False),
        )
        parsed_json = parse_json_object(raw)
    except Exception:
        return []
    return normalize_planner_steps(parsed_json.get("steps"))


def code_generator_agent(
    state: AgentState,
    model_complete: ModelCompleteFn | None,
    fallback_model: FallbackModelFn | None,
) -> dict[str, Any]:
    parsed = state["parsed"]
    formula = get_formula(state.get("formula_id", ""))
    code = generate_code_with_model(parsed, formula, state, model_complete)
    if not code and formula is not None:
        code = deterministic_code_from_formula(parsed, formula)
    if not code and model_complete is not None:
        code = generate_code_with_fallback_prompt(parsed, state, model_complete)
    if not code:
        if fallback_model is not None:
            return {"decision": "fallback"}
        return {
            "execution": ExecutionResult(
                False, None, "", {}, [], "No generated code available."
            ),
            "code": "",
            "review": VerificationResult(False, 0.0, ["No generated code available."]),
            "feedback": "No generated code available.",
        }
    execution = execute_generated_code_tool(code, parsed)
    plan = plan_from_code(parsed, formula, code)
    logger.info("type2.code_generator code=%s execution=%s", code, execution)
    return {"code": code, "plan": plan, "execution": execution}


def generate_code_with_model(
    parsed: ParsedProblem,
    formula: Formula | None,
    state: AgentState,
    model_complete: ModelCompleteFn | None,
) -> str:
    if model_complete is None:
        return ""
    payload = {
        "question": parsed.question,
        "si_variables": {
            symbol: known.si_value for symbol, known in parsed.knowns.items()
        },
        "target": parsed.target,
        "target_unit": parsed.target_unit,
        "formula_candidate": formula_to_payload(formula),
        "planner_steps": state.get("planner_steps", []),
        "review_feedback": state.get("feedback", ""),
        "review_errors": state.get("review").errors if state.get("review") else [],
    }
    try:
        raw = model_complete(
            CODE_GENERATOR_PROMPT, json.dumps(payload, ensure_ascii=False)
        )
        parsed_json = parse_json_object(raw)
    except Exception:
        return ""
    code = str(parsed_json.get("code", "")).strip()
    if code:
        return code
    raw_code = extract_code_block(raw) or raw.strip()
    return raw_code if looks_like_code(raw_code) else ""


def generate_code_with_fallback_prompt(
    parsed: ParsedProblem,
    state: AgentState,
    model_complete: ModelCompleteFn,
) -> str:
    payload = {
        "question": parsed.question,
        "si_variables": {
            symbol: known.si_value for symbol, known in parsed.knowns.items()
        },
        "target": parsed.target,
        "target_unit": parsed.target_unit,
        "planner_steps": state.get("planner_steps", []),
        "feedback": state.get("feedback", ""),
        "instruction": "Write executable Python directly, no JSON required.",
    }
    try:
        raw = model_complete(
            "You are a physics code generator. Write executable Python only.",
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception:
        return ""
    raw_code = extract_code_block(raw) or raw.strip()
    return raw_code if looks_like_code(raw_code) else ""


def deterministic_code_from_formula(parsed: ParsedProblem, formula: Formula) -> str:
    lines = ["import math", "import sympy as sp", "steps = []", "premises = []"]
    for target, expression in formula.intermediates:
        lines.append(f"{target} = {expression}")
        lines.append(f"steps.append('{target} = {expression} = ' + str({target}))")
    final_target = (
        parsed.target
        if parsed.target in formula.target_variables
        else formula.target_variables[0]
    )
    lines.append(f"answer = {formula.expression}")
    lines.append(f"unit = {formula.output_unit!r}")
    lines.append(
        f"steps.append('{final_target} = {formula.expression} = ' + str(answer))"
    )
    lines.append(f"premises.append({formula.description!r})")
    return "\n".join(lines)


def plan_from_code(
    parsed: ParsedProblem, formula: Formula | None, code: str
) -> EquationPlan:
    steps = code_to_plan_steps(parsed, code)
    return EquationPlan(
        formula_id=formula.id if formula is not None else "custom_generated_code",
        steps=steps,
        confidence=0.8 if formula is not None else 0.6,
        source="generated_code",
    )


def code_to_plan_steps(parsed: ParsedProblem, code: str) -> list[PlanStep]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return [
            PlanStep(
                formula_id="custom_generated_code",
                formula="custom generated code",
                substitution=code,
                target=parsed.target or "answer",
                expression="answer",
                source="generated_code",
            )
        ]

    steps: list[PlanStep] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        target_name = target.id
        if target_name in {"steps", "premises", "unit"}:
            continue
        expression = ast.unparse(node.value)
        if target_name == "answer":
            target_name = parsed.target or "answer"
        steps.append(
            PlanStep(
                formula_id="generated_code",
                formula="generated code",
                substitution=expression,
                target=target_name,
                expression=expression,
                source="generated_code",
            )
        )
    if not steps:
        steps.append(
            PlanStep(
                formula_id="custom_generated_code",
                formula="custom generated code",
                substitution=code,
                target=parsed.target or "answer",
                expression="answer",
                source="generated_code",
            )
        )
    return steps


def reviewer_agent(
    state: AgentState,
    model_complete: ModelCompleteFn | None,
) -> dict[str, Any]:
    parsed = state["parsed"]
    plan = state["plan"]
    execution = state["execution"]
    consistency = backward_consistency_check(parsed, plan, execution)
    model_review = reviewer_model_decision(state, consistency, model_complete)
    review = model_review or execution_only_review(consistency)
    logger.info("type2.reviewer review=%s", review)
    return {
        "review": review,
        "review_source": "slm" if model_review else "execution_only",
        "feedback": "; ".join(review.errors),
        "errors": state.get("errors", []) + consistency.errors,
    }


def execution_only_review(consistency: VerificationResult) -> VerificationResult:
    if consistency.passed:
        return VerificationResult(
            True, 0.5, ["No SLM reviewer available; accepted execution-only result."]
        )
    return VerificationResult(False, 0.25, consistency.errors)


def reviewer_model_decision(
    state: AgentState,
    consistency: VerificationResult,
    model_complete: ModelCompleteFn | None,
) -> VerificationResult | None:
    if model_complete is None:
        return None
    execution = state["execution"]
    payload = {
        "question": state["question"],
        "parsed_knowns": {
            symbol: known.raw for symbol, known in state["parsed"].knowns.items()
        },
        "code": state.get("code", ""),
        "execution_trace": execution.trace,
        "answer": execution.answer,
        "unit": execution.unit,
        "backward_consistency": {
            "passed": consistency.passed,
            "issues": consistency.errors,
        },
        "planner_steps": state.get("planner_steps", []),
    }
    try:
        raw = model_complete(
            REVIEWER_AGENT_PROMPT, json.dumps(payload, ensure_ascii=False)
        )
        parsed_json = parse_json_object(raw)
    except Exception:
        return None
    passed = parsed_json.get("passed")
    confidence = parsed_json.get("confidence")
    errors = parsed_json.get("errors", [])
    if (
        not isinstance(passed, bool)
        or not isinstance(confidence, (int, float))
        or not isinstance(errors, list)
    ):
        return None
    combined_errors = [str(item) for item in errors]
    if consistency.errors:
        combined_errors.extend(
            f"Consistency issue: {item}" for item in consistency.errors
        )
    return VerificationResult(
        passed and consistency.passed,
        max(0.0, min(float(confidence), 1.0)),
        combined_errors,
    )


def fallback_agent(
    state: AgentState,
    fallback_model: FallbackModelFn | None,
) -> dict[str, Any]:
    result = solve_with_llm_fallback(state["question"], fallback_model)
    result.metadata["agent_loop"] = "fallback"
    return {"result": result}


def finalizer_agent(
    state: AgentState,
    model_complete: ModelCompleteFn | None,
) -> dict[str, Any]:
    parsed = state["parsed"]
    plan = state.get("plan")
    execution = state.get("execution")
    review = state.get("review")
    if plan is None or execution is None or execution.answer is None:
        result = PipelineResult(
            answer="",
            unit="",
            explanation="No executable code produced a final answer.",
            cot=["Planner/code generator did not produce a usable execution."],
            premises=[],
            confidence=0.2,
            metadata={"agent_loop": "finalizer_no_execution"},
        )
        return {"result": result}

    if review is None:
        review = VerificationResult(False, 0.35, ["Reviewer did not run."])
    if not review.passed and state.get("loop_count", 0) >= MAX_LOOPS:
        review = VerificationResult(
            False,
            min(review.confidence, 0.4),
            review.errors + ["Returned nearest result after max planner loops."],
        )
    answer_text = format_number(execution.answer)
    explanation, cot, premises = build_explanation(
        parsed,
        plan,
        execution,
        review,
        answer_text,
        model_complete,
    )
    result = PipelineResult(
        answer=answer_text,
        unit=execution.unit,
        explanation=explanation,
        cot=cot,
        premises=premises,
        confidence=0.5
        if state.get("review_source") == "execution_only"
        else review.confidence,
        metadata={
            "agent_loop": "langgraph",
            "formula_id": plan.formula_id,
            "planner_steps": state.get("planner_steps", []),
            "plan_source": plan.source,
            "verified": review.passed,
            "review_errors": review.errors,
            "review_source": state.get("review_source", "execution_only"),
            "loops": state.get("loop_count", 0),
        },
    )
    return {"result": result}


def formula_to_payload(formula: Formula | None) -> dict[str, Any] | None:
    if formula is None:
        return None
    return {
        "id": formula.id,
        "equation": formula.equation,
        "expression": formula.expression,
        "intermediates": formula.intermediates,
        "output_unit": formula.output_unit,
        "description": formula.description,
    }


def route_from_planner(
    state: AgentState,
) -> Literal["code_generator", "fallback", "finalizer"]:
    decision = state.get("decision")
    if decision == "code_generator":
        return "code_generator"
    if decision == "fallback":
        return "fallback"
    return "finalizer"


def route_from_code_generator(
    state: AgentState,
) -> Literal["reviewer", "planner", "fallback"]:
    if state.get("decision") == "fallback":
        return "fallback"
    if state.get("plan") and state.get("execution"):
        return "reviewer"
    return "planner"


def extract_code_block(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return ""


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    return any(token in text for token in ("=", "answer", "unit", "steps", "premises"))
