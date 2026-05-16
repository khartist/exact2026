#!/usr/bin/env python3
"""Baseline runner for EXACT 2026 Task 2 using local Ollama."""

from __future__ import annotations

import argparse
import csv
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
        user_prompt = build_user_prompt(row)
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
                "row_index": row_index,
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answer": parsed["answer"],
                "unit": parsed["unit"],
                "explanation": parsed["explanation"],
                "raw_response": raw_response,
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
