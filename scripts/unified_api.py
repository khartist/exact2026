#!/usr/bin/env python3
"""Unified EXACT 2026 API runner that routes to task-specific solvers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import task1_baseline
import task2_baseline


QueryType = Literal["type1", "type2"]
ModelFn = Callable[[str, QueryType], str]

DEFAULT_MODEL = "gemma4:e2b-it-q4_K_M"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass(frozen=True)
class UnifiedQuery:
    """Normalized official unified API input."""

    query_type: QueryType
    question: str
    premises: list[str]
    raw_sample: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official EXACT 2026 unified API-format samples."
    )
    parser.add_argument("--input", required=True, help="Unified input JSON file.")
    parser.add_argument("--output", required=True, help="Unified response JSON file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama chat API URL.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run N samples.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based sample offset to start from.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout per sample in seconds.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_unified_samples(path: Path) -> list[dict[str, Any]]:
    """Load a unified stream from common wrapper keys or a single sample."""

    data = load_json(path)
    if isinstance(data, list):
        return validate_sample_list(data)
    if not isinstance(data, dict):
        raise TypeError(f"Input JSON must be an object or list, got {type(data).__name__}")

    for key in ("queries", "samples", "inputs", "data"):
        value = data.get(key)
        if value is not None:
            return validate_sample_list(value)

    if "question" in data:
        return [data]

    raise ValueError(
        'Input JSON object must contain "queries", "samples", "inputs", "data", '
        'or a single sample with "question".'
    )


def validate_sample_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"Unified stream must be a list, got {type(value).__name__}")

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(value):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample at index {index} must be an object.")
        samples.append(sample)
    return samples


def normalize_query(sample: dict[str, Any]) -> UnifiedQuery:
    """Detect query type and normalize fields without requiring optional data."""

    raw_premises = sample.get("premises-NL")
    query_type: QueryType = "type1" if has_non_empty_premises_field(raw_premises) else "type2"
    return UnifiedQuery(
        query_type=query_type,
        question=normalize_string(sample.get("question", "")),
        premises=normalize_premises(raw_premises),
        raw_sample=sample,
    )


def has_non_empty_premises_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def normalize_premises(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text for item in value if (text := normalize_string(item))]
    if value:
        return [normalize_string(value)]
    return []


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def solve_unified_query(query: UnifiedQuery, model_fn: ModelFn) -> dict[str, Any]:
    """Route to the task-specific solver while sharing the official input API."""

    if query.query_type == "type1":
        return solve_type1_query(query, model_fn)
    return solve_type2_query(query, model_fn)


def solve_type1_query(query: UnifiedQuery, model_fn: ModelFn) -> dict[str, Any]:
    record = {
        "premises-NL": query.premises,
        "premises-FOL": normalize_premises(query.raw_sample.get("premises-FOL")),
    }
    prompt = task1_baseline.build_user_prompt(record, query.question, include_fol=False)
    parsed = task1_baseline.parse_model_response(model_fn(prompt, "type1"))
    response = normalize_api_response(parsed)
    if query.premises:
        response["premises"] = query.premises
    fol = normalize_fol(query.raw_sample.get("premises-FOL"))
    if fol:
        response["fol"] = fol
    return response


def solve_type2_query(query: UnifiedQuery, model_fn: ModelFn) -> dict[str, Any]:
    prompt = task2_baseline.build_user_prompt({"question": query.question})
    parsed = task2_baseline.parse_model_response(model_fn(prompt, "type2"))
    if parsed.get("unit"):
        parsed = dict(parsed)
        parsed["answer"] = join_answer_and_unit(parsed.get("answer"), parsed.get("unit"))
    return normalize_api_response(parsed)


def normalize_fol(value: Any) -> str | None:
    if isinstance(value, list):
        lines = [normalize_string(item) for item in value if normalize_string(item)]
        return "\n".join(lines) if lines else None
    text = normalize_string(value)
    return text or None


def join_answer_and_unit(answer: Any, unit: Any) -> str:
    answer_text = normalize_string(answer)
    unit_text = normalize_string(unit)
    if not unit_text:
        return answer_text
    if unit_text in answer_text:
        return answer_text
    return f"{answer_text} {unit_text}".strip()


def normalize_api_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Return valid JSON-serializable API data with mandatory fields."""

    response: dict[str, Any] = {
        "answer": normalize_string(parsed.get("answer", "")),
        "explanation": normalize_explanation(parsed.get("explanation", "")),
    }

    optional_normalizers: dict[str, Callable[[Any], Any | None]] = {
        "fol": normalize_optional_string,
        "cot": normalize_optional_string_list,
        "premises": normalize_optional_string_list,
        "confidence": normalize_optional_confidence,
    }
    for field, normalizer in optional_normalizers.items():
        if field not in parsed:
            continue
        value = normalizer(parsed[field])
        if value is not None:
            response[field] = value
    return response


def normalize_explanation(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(normalize_string(item) for item in value if normalize_string(item))
    return normalize_string(value)


def normalize_optional_string(value: Any) -> str | None:
    text = normalize_string(value)
    return text if text else None


def normalize_optional_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [normalize_string(item) for item in value if normalize_string(item)]
    return items if items else None


def normalize_optional_confidence(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return value
    return None


def validate_api_response(response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(response.get("answer"), str):
        errors.append('"answer" must be a string.')
    if not isinstance(response.get("explanation"), str):
        errors.append('"explanation" must be a string.')
    if "fol" in response and not isinstance(response["fol"], str):
        errors.append('"fol" must be a string when present.')
    if "cot" in response and not is_string_list(response["cot"]):
        errors.append('"cot" must be a list of strings when present.')
    if "premises" in response and not is_string_list(response["premises"]):
        errors.append('"premises" must be a list of strings when present.')
    if "confidence" in response:
        confidence = response["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append('"confidence" must be a number when present.')
        elif not 0 <= confidence <= 1:
            errors.append('"confidence" must be between 0 and 1 when present.')
    return errors


def is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    samples = load_unified_samples(input_path)
    end = None if args.limit is None else args.start + args.limit
    selected = list(enumerate(samples))[args.start:end]

    def ollama_model_fn(prompt: str, query_type: QueryType) -> str:
        if query_type == "type1":
            return task1_baseline.call_ollama(
                args.ollama_url,
                args.model,
                prompt,
                args.temperature,
                args.timeout,
            )
        return task2_baseline.call_ollama(
            args.ollama_url,
            args.model,
            prompt,
            args.temperature,
            args.timeout,
        )

    responses: list[dict[str, Any]] = []
    for sample_index, sample in selected:
        query = normalize_query(sample)
        response = solve_unified_query(query, ollama_model_fn)
        validation_errors = validate_api_response(response)
        if validation_errors:
            response["validation_errors"] = validation_errors
        response["sample_index"] = sample_index
        response["query_type"] = query.query_type
        responses.append(response)
        print(f"[{len(responses)}/{len(selected)}] sample={sample_index} type={query.query_type}")

    save_json(
        output_path,
        {
            "input": str(input_path),
            "model": args.model,
            "responses": responses,
        },
    )
    print(f"Wrote {len(responses)} unified responses to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
