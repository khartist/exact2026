#!/usr/bin/env python3
"""Baseline runner for EXACT 2026 Task 2 using local Ollama."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exact2026.type2.pipeline import dumps_response, solve_physics_question

DEFAULT_INPUT = (
    "EXACT2026_dataset_2026-05-15/"
    "Physics_Problems_Text_Only/"
    "Physics_Problems_Text_Only.csv"
)
DEFAULT_OUTPUT = "outputs/task2_gemma4_e2b_baseline.json"


SYSTEM_PROMPT = """<|think|>
You are a careful physics problem solver.
Use only the information stated in the problem.
Make sure to think carefully before answering.
Solve step by step, track units, and return exactly one JSON object with these keys:
- "answer": the final numeric or symbolic answer only
- "unit": the final unit only, or an empty string if dimensionless
- "explanation": a concise derivation explaining the formula, substitutions, and unit conversion
Use plain text math in JSON string values. Do not use LaTeX commands or backslashes.

Do not include markdown. Do not include extra keys.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Task 2 physics baseline with a local Ollama model."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Task 2 CSV path.")
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
        help="Only run this many rows. Use --limit 1 or --limit 10 for testing.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based row offset to start from.",
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
        help="HTTP timeout per row in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between rows in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing predictions if the output file already exists.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_user_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "Physics problem:",
            row["question"].strip(),
            "",
            "Return JSON only, for example:",
            '{"answer":"0.045","unit":"J","explanation":"..."}',
        ]
    )


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
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except requests.HTTPError as exc:
        response = exc.response
        status = (
            f"HTTP {response.status_code} {response.reason}"
            if response is not None
            else "HTTP error"
        )
        error_body = response.text if response is not None else ""
        raise RuntimeError(
            f"Failed to call Ollama at {url}: {status}. Body: {error_body}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned non-JSON response: {response.text}"
        ) from exc

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

    for candidate in candidates:
        parsed = extract_json_like_fields(candidate, ("answer", "unit", "explanation"))
        if parsed:
            return parsed
    return None


def extract_json_like_fields(text: str, keys: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key in keys:
        value = extract_json_like_string_field(text, key)
        if value is not None:
            parsed[key] = value
    return parsed


def extract_json_like_string_field(text: str, key: str) -> str | None:
    quoted = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    if quoted:
        return decode_json_like_string(quoted.group(1))

    unquoted = re.search(
        rf'"{re.escape(key)}"\s*:\s*([^,\n}}]+)',
        text,
        re.DOTALL,
    )
    if unquoted:
        return unquoted.group(1).strip().strip('"')
    return None


def decode_json_like_string(text: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char != "\\" or i + 1 >= len(text):
            chars.append(char)
            i += 1
            continue

        next_char = text[i + 1]
        if next_char in {'"', "\\", "/"}:
            chars.append(next_char)
        else:
            chars.append(char)
            chars.append(next_char)
        i += 2
    return "".join(chars)


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_model_response(raw_text: str) -> dict[str, str]:
    parsed = extract_json_object(raw_text)
    if parsed is None:
        return {
            "answer": "",
            "unit": "",
            "explanation": raw_text.strip(),
        }

    explanation = parsed.get("explanation", "")
    if isinstance(explanation, list):
        explanation = " ".join(str(item) for item in explanation)
    return {
        "answer": clean_scalar(parsed.get("answer")),
        "unit": clean_scalar(parsed.get("unit")),
        "explanation": clean_scalar(explanation) or raw_text.strip(),
    }


def existing_keys(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    data = load_json(output_path)
    return {int(item["row_index"]) for item in data.get("predictions", [])}


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)
    end = None if args.limit is None else args.start + args.limit
    selected = list(enumerate(rows))[args.start : end]
    seen = existing_keys(output_path) if args.resume else set()

    predictions: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        predictions.extend(load_json(output_path).get("predictions", []))

    total_rows = len(selected)
    done = 0

    for row_index, row in selected:
        if row_index in seen:
            continue

        done += 1
        print(
            f"[{done}/{total_rows}] row={row_index} id={row.get('id', '')}", flush=True
        )
        result = solve_physics_question(
            row.get("question", ""),
            model_complete=lambda system_prompt, user_prompt: call_ollama(
                args.ollama_url,
                args.model,
                "\n\n".join([system_prompt, user_prompt]),
                args.temperature,
                args.timeout,
            ),
            fallback_model=lambda prompt: call_ollama(
                args.ollama_url,
                args.model,
                prompt,
                args.temperature,
                args.timeout,
            ),
        )
        parsed = result.to_api_dict()
        raw_response = result.raw_response or dumps_response(result)

        correct_ans = row.get("answer", [])
        correct_cot = row.get("cot", [])
        correct_unit = row.get("unit", [])

        predictions.append(
            {
                "row_index": row_index,
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answer": parsed["answer"],
                "unit": parsed["unit"],
                "explanation": parsed["explanation"],
                "cot": parsed.get("cot", []),
                "premises": parsed.get("premises", []),
                "confidence": parsed.get("confidence", 0),
                "raw_response": raw_response,
                "correct_ans": correct_ans,
                "correct_cot": correct_cot,
                "correct_unit": correct_unit,
            }
        )
        save_json(
            output_path,
            {
                "task": "Physics_Problems_Text_Only",
                "model": args.model,
                "input": str(input_path),
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
