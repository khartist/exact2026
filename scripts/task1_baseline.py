#!/usr/bin/env python3
"""Baseline runner for EXACT 2026 Task 1 using local Ollama."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_INPUT = (
    "EXACT2026_dataset_2026-05-15/"
    "Logic_Based_Educational_Queries_Text_Only/"
    "Logic_Based_Educational_Queries.json"
)
DEFAULT_OUTPUT = "outputs/task1_gemma4_e2b_baseline.json"
VALID_ANSWERS = ("Yes", "No", "Unknown", "A", "B", "C", "D")


SYSTEM_PROMPT = """<|think|>
You are a careful logic reasoner for educational logic queries.
Use only the stated premises. Do not use outside facts.
For each question, decide whether the queried statement follows from the premises.
Make sure to think carefully before answering.
Utilize the provided premises to make your reasoning.
Return exactly one JSON object with these keys:
- "answer": one of "Yes", "No", or "Unknown" for Yes No question and "A", "B", "C", "D" for multiple choice questions
- "explanation": a concise logical explanation for the answer

Answer rules:
- "Yes" means the statement is entailed by the premises.
- "No" means the negation is entailed by the premises or the statement contradicts the premises.
- "Unknown" means neither the statement nor its negation is entailed by the premises.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Task 1 baseline with a local Ollama model."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Task 1 input JSON path."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Prediction JSON path."
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b-it-q4_K_M",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat API URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run this many records. Use --limit 1 or --limit 10 for testing.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based record offset to start from.",
    )
    parser.add_argument(
        "--include-fol",
        action="store_true",
        help="Include premises-FOL in the prompt in addition to premises-NL.",
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
        help="HTTP timeout per question in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between questions in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing predictions if the output file already exists.",
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


def build_user_prompt(record: dict[str, Any], question: str, include_fol: bool) -> str:
    nl_lines = "\n".join(
        f"{i}. {premise}" for i, premise in enumerate(record["premises-NL"], start=1)
    )
    parts = [
        "Premises in natural language:",
        nl_lines,
    ]

    if include_fol:
        fol_lines = "\n".join(
            f"{i}. {premise}"
            for i, premise in enumerate(record["premises-FOL"], start=1)
        )
        parts.extend(["", "Premises in first-order logic:", fol_lines])

    parts.extend(
        [
            "",
            "Question:",
            question,
            "",
            'Return JSON only, for example: {"answer":"Yes","explanation":"..."}',
        ]
    )
    return "\n".join(parts)


def call_ollama(
    url: str,
    model: str,
    user_prompt: str,
    temperature: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to call Ollama at {url}: HTTP {exc.code} {exc.reason}. "
            f"Body: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call Ollama at {url}: {exc}") from exc

    try:
        return body["message"]["content"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {body}") from exc


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidates = [stripped]
    if fenced:
        candidates.insert(0, fenced.group(1))

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(stripped[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_answer(value: Any, raw_text: str) -> str:
    text = str(value or "").strip()
    for answer in VALID_ANSWERS:
        if text.lower() == answer.lower():
            return answer

    match = re.search(r"\b(yes|no|unknown)\b", raw_text, re.IGNORECASE)
    if match:
        word = match.group(1).lower()
        return {"yes": "Yes", "no": "No", "unknown": "Unknown"}[word]
    return "Unknown"


def parse_model_response(raw_text: str) -> dict[str, str]:
    parsed = extract_json_object(raw_text)
    if parsed is None:
        return {
            "answer": normalize_answer(None, raw_text),
            "explanation": raw_text.strip(),
        }

    explanation = parsed.get("explanation", "")
    if isinstance(explanation, list):
        explanation = " ".join(str(item) for item in explanation)
    return {
        "answer": normalize_answer(parsed.get("answer"), raw_text),
        "explanation": str(explanation).strip() or raw_text.strip(),
    }


def existing_keys(output_path: Path) -> set[tuple[int, int]]:
    if not output_path.exists():
        return set()
    data = load_json(output_path)
    return {
        (int(item["record_index"]), int(item["question_index"]))
        for item in data.get("predictions", [])
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_json(input_path)
    if not isinstance(records, list):
        raise TypeError(
            f"Expected input JSON to be a list, got {type(records).__name__}"
        )

    end = None if args.limit is None else args.start + args.limit
    selected = list(enumerate(records))[args.start : end]
    seen = existing_keys(output_path) if args.resume else set()

    predictions: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        predictions.extend(load_json(output_path).get("predictions", []))

    total_questions = sum(len(record.get("questions", [])) for _, record in selected)
    done = 0

    for record_index, record in selected:
        for question_index, question in enumerate(record.get("questions", [])):
            key = (record_index, question_index)
            if key in seen:
                continue

            done += 1
            print(
                f"[{done}/{total_questions}] record={record_index} question={question_index}",
                flush=True,
            )
            user_prompt = build_user_prompt(record, question, args.include_fol)
            raw_response = call_ollama(
                args.ollama_url,
                args.model,
                user_prompt,
                args.temperature,
                args.timeout,
            )
            parsed = parse_model_response(raw_response)
            predictions.append(
                {
                    "record_index": record_index,
                    "question_index": question_index,
                    "question": question,
                    "answer": parsed["answer"],
                    "explanation": parsed["explanation"],
                    "raw_response": raw_response,
                }
            )
            save_json(
                output_path,
                {
                    "task": "Logic_Based_Educational_Queries_Text_Only",
                    "model": args.model,
                    "input": str(input_path),
                    "valid_answers": list(VALID_ANSWERS),
                    "predictions": predictions,
                },
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
