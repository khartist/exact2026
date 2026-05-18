"""Orchestration for the Type 2 physics MVP pipeline."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from .parser import ModelCompleteFn, extract_json_object, parse_question
from .schemas import PipelineResult
from .verifier import verify_llm_fallback

logger = logging.getLogger(__name__)

FallbackModelFn = Callable[[str], str]

FALLBACK_SYSTEM_PROMPT = """You solve physics problems.
Use only the stated problem information and standard physics formulas.
Return strict JSON only with keys: answer, unit, formula, explanation, cot, premises, confidence.
The answer must be numeric without the unit. Do not use markdown."""


def solve_physics_question(
    question: str,
    model_complete: ModelCompleteFn | None = None,
    fallback_model: FallbackModelFn | None = None,
) -> PipelineResult:
    from .agent_loop import solve_with_agent_loop

    logger.info("type2.raw_question=%s", question)
    result = solve_with_agent_loop(
        question,
        model_complete=model_complete,
        fallback_model=fallback_model or fallback_from_model_complete(model_complete),
    )
    logger.info("type2.final_response=%s", result)
    return result


def fallback_from_model_complete(model_complete: ModelCompleteFn | None) -> FallbackModelFn | None:
    if model_complete is None:
        return None

    def call(prompt: str) -> str:
        return model_complete(FALLBACK_SYSTEM_PROMPT, prompt)

    return call


def solve_with_llm_fallback(
    question: str,
    fallback_model: FallbackModelFn | None,
) -> PipelineResult:
    if fallback_model is None:
        return PipelineResult(
            answer="",
            unit="",
            explanation=(
                "The problem could not be solved by the current formula bank, and no "
                "fallback model was available to produce an answer."
            ),
            cot=[
                "Parsed the question.",
                "Checked the formula bank for a matching trusted formula.",
                "No matching formula or fallback model was available.",
            ],
            premises=[],
            confidence=0.2,
            metadata={"fallback": "unavailable"},
        )

    prompt = "\n".join(["Physics problem:", question, "", "Return strict JSON only."])
    try:
        raw = fallback_model(prompt)
    except Exception as exc:
        return PipelineResult(
            answer="",
            unit="",
            explanation=f"The fallback model failed: {exc}",
            cot=["Formula bank did not match.", "Fallback model call failed."],
            premises=[],
            confidence=0.2,
            raw_response=str(exc),
            metadata={"fallback": "error"},
        )
    parsed = extract_json_object(raw) or {}
    answer = clean_scalar(parsed.get("answer"))
    unit = clean_scalar(parsed.get("unit"))
    verification = verify_llm_fallback(answer, unit)
    cot = normalize_string_list(parsed.get("cot")) or [
        "Formula bank did not match.",
        "Used fallback model physics knowledge.",
    ]
    premises = normalize_string_list(parsed.get("premises"))
    if not premises and parsed.get("formula"):
        premises = [clean_scalar(parsed.get("formula"))]
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = verification.confidence
    confidence = max(0.2, min(float(confidence), verification.confidence if not verification.passed else 0.7))
    return PipelineResult(
        answer=answer,
        unit=unit,
        explanation=clean_scalar(parsed.get("explanation")) or "Solved using fallback model physics knowledge.",
        cot=cot,
        premises=premises,
        confidence=confidence,
        raw_response=raw,
        metadata={"fallback": "llm", "verified": verification.passed},
    )


def format_number(value: float) -> str:
    rounded = round(value, 2)
    if abs(value - rounded) <= max(1e-9, abs(value) * 1e-4):
        text = f"{rounded:.2f}"
    else:
        text = f"{value:.6g}"
    return re.sub(r"\.?0+$", "", text) if "." in text else text


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := clean_scalar(item))]


def dumps_response(result: PipelineResult) -> str:
    return json.dumps(result.to_api_dict(), ensure_ascii=False)
