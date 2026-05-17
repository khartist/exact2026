#!/usr/bin/env python3
"""Evaluation framework for pre-run EXACT 2026 model outputs."""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


TaskName = Literal["task1", "task2"]


TASK_ALIASES: dict[str, TaskName] = {
    "task1": "task1",
    "type1": "task1",
    "logic_based_educational_queries_text_only": "task1",
    "task2": "task2",
    "type2": "task2",
    "physics_problems_text_only": "task2",
}


@dataclass(frozen=True)
class FieldComparison:
    """One expected-vs-predicted field comparison."""

    field: str
    predicted: str | None
    expected: str | None
    correct: bool


@dataclass(frozen=True)
class SampleEvaluation:
    """Per-sample metric result."""

    sample_index: int
    task: str | None
    sample_id: str | int | None
    correct: bool
    comparisons: list[FieldComparison]
    errors: list[str]


class TaskExactMatchEvaluator(ABC):
    """Task-specific P1 exact-match logic."""

    task_name: TaskName
    required_fields: tuple[str, ...]

    def evaluate(self, sample: dict[str, Any], sample_index: int) -> SampleEvaluation:
        errors = missing_field_errors(sample, self.required_fields)
        comparisons = [] if errors else self.compare(sample)
        correct = bool(comparisons) and all(item.correct for item in comparisons)
        return SampleEvaluation(
            sample_index=sample_index,
            task=self.task_name,
            sample_id=sample.get("id", sample.get("record_index", sample.get("row_index"))),
            correct=correct,
            comparisons=comparisons,
            errors=errors,
        )

    @abstractmethod
    def compare(self, sample: dict[str, Any]) -> list[FieldComparison]:
        """Return field comparisons for this task."""


class Task1ExactMatchEvaluator(TaskExactMatchEvaluator):
    task_name: TaskName = "task1"
    required_fields = ("answer", "correct_answer")

    def compare(self, sample: dict[str, Any]) -> list[FieldComparison]:
        return [
            compare_field(
                field="answer",
                predicted=sample.get("answer"),
                expected=sample.get("correct_answer"),
            )
        ]


class Task2ExactMatchEvaluator(TaskExactMatchEvaluator):
    task_name: TaskName = "task2"
    required_fields = ("answer", "unit", "correct_ans", "correct_unit")

    def compare(self, sample: dict[str, Any]) -> list[FieldComparison]:
        return [
            compare_field(
                field="answer",
                predicted=sample.get("answer"),
                expected=sample.get("correct_ans"),
            ),
            compare_field(
                field="unit",
                predicted=sample.get("unit"),
                expected=sample.get("correct_unit"),
            ),
        ]


class Metric(ABC):
    """Base interface for metrics."""

    name: str

    @abstractmethod
    def evaluate_samples(
        self,
        samples: list[dict[str, Any]],
        task_override: TaskName | None,
        default_task: TaskName | None,
    ) -> dict[str, Any]:
        """Evaluate all samples and return a serializable result."""


class P1ExactMatchMetric(Metric):
    """P1 exact-match correctness with task-specific field logic."""

    name = "p1_exact_match"

    def __init__(self) -> None:
        self.evaluators: dict[TaskName, TaskExactMatchEvaluator] = {
            "task1": Task1ExactMatchEvaluator(),
            "task2": Task2ExactMatchEvaluator(),
        }

    def evaluate_samples(
        self,
        samples: list[dict[str, Any]],
        task_override: TaskName | None,
        default_task: TaskName | None,
    ) -> dict[str, Any]:
        per_sample = [
            self.evaluate_one(sample, sample_index, task_override, default_task)
            for sample_index, sample in enumerate(samples)
        ]
        total_samples = len(per_sample)
        correct_samples = sum(item.correct for item in per_sample)
        accuracy = correct_samples / total_samples if total_samples else 0.0
        return {
            "metric": self.name,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": accuracy,
            "samples": [serialize_sample_result(item) for item in per_sample],
        }

    def evaluate_one(
        self,
        sample: dict[str, Any],
        sample_index: int,
        task_override: TaskName | None,
        default_task: TaskName | None,
    ) -> SampleEvaluation:
        task = task_override or infer_task(sample) or default_task
        if task is None:
            return SampleEvaluation(
                sample_index=sample_index,
                task=None,
                sample_id=sample.get("id", sample.get("record_index", sample.get("row_index"))),
                correct=False,
                comparisons=[],
                errors=[
                    "Unable to infer task. Provide --task task1 or --task task2, "
                    "or include the required task-specific fields."
                ],
            )
        return self.evaluators[task].evaluate(sample, sample_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-run EXACT 2026 outputs with P1 exact match."
    )
    parser.add_argument("--input", required=True, help="Input JSON output file.")
    parser.add_argument("--output", required=True, help="Evaluation result JSON file.")
    parser.add_argument(
        "--task",
        choices=("auto", "task1", "task2"),
        default="auto",
        help="Task override. Default: infer per sample.",
    )
    parser.add_argument(
        "--metric",
        choices=("p1_exact_match",),
        default="p1_exact_match",
        help="Metric to run.",
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


def extract_samples(data: Any) -> tuple[list[dict[str, Any]], TaskName | None]:
    """Read samples from either a prediction object or a plain list."""

    if isinstance(data, list):
        return validate_samples(data), None

    if not isinstance(data, dict):
        raise TypeError(
            f"Input JSON must be an object or list, got {type(data).__name__}"
        )

    raw_samples = data.get("predictions", data.get("samples"))
    if raw_samples is None:
        raise ValueError('Input JSON object must contain "predictions" or "samples".')

    task_hint = normalize_task_name(data.get("task"))
    return validate_samples(raw_samples), task_hint


def validate_samples(raw_samples: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_samples, list):
        raise TypeError(
            f'Expected "predictions"/"samples" to be a list, got {type(raw_samples).__name__}'
        )

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample at index {index} must be an object.")
        samples.append(sample)
    return samples


def normalize_task_name(value: Any) -> TaskName | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return TASK_ALIASES.get(normalized)


def infer_task(sample: dict[str, Any]) -> TaskName | None:
    sample_task = normalize_task_name(sample.get("task"))
    if sample_task:
        return sample_task

    has_task2 = all(field in sample for field in Task2ExactMatchEvaluator.required_fields)
    if has_task2:
        return "task2"

    has_task1 = all(field in sample for field in Task1ExactMatchEvaluator.required_fields)
    if has_task1:
        return "task1"

    return None


def missing_field_errors(sample: dict[str, Any], required_fields: Iterable[str]) -> list[str]:
    return [
        f'Missing required field "{field}".'
        for field in required_fields
        if field not in sample
    ]


def compare_field(field: str, predicted: Any, expected: Any) -> FieldComparison:
    predicted_text = normalize_value(predicted)
    expected_text = normalize_value(expected)
    return FieldComparison(
        field=field,
        predicted=predicted_text,
        expected=expected_text,
        correct=predicted_text == expected_text,
    )


def normalize_value(value: Any) -> str | None:
    """Normalize values for exact match without changing case or numeric format."""

    if value is None:
        return None
    return str(value).strip()


def serialize_sample_result(result: SampleEvaluation) -> dict[str, Any]:
    return {
        "sample_index": result.sample_index,
        "task": result.task,
        "id": result.sample_id,
        "correct": result.correct,
        "comparisons": [
            {
                "field": item.field,
                "predicted": item.predicted,
                "expected": item.expected,
                "correct": item.correct,
            }
            for item in result.comparisons
        ],
        "errors": result.errors,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_json(input_path)
    samples, file_task_hint = extract_samples(data)
    task_override = None if args.task == "auto" else args.task

    metric = P1ExactMatchMetric()
    result = metric.evaluate_samples(samples, task_override, file_task_hint)
    result["input"] = str(input_path)
    result["task"] = task_override or "auto"
    if file_task_hint:
        result["file_task_hint"] = file_task_hint

    save_json(output_path, result)
    print(
        f"Wrote {result['metric']} evaluation for {result['total_samples']} samples "
        f"to {output_path}. Accuracy: {result['accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
