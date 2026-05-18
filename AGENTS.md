# AGENTS.md

Guidance for coding agents working in this repository.

## Project Purpose

This repo contains baseline runners and evaluation utilities for EXACT 2026
text-only tasks using a local Ollama model.

Supported tasks:

- Task 1: `Logic_Based_Educational_Queries_Text_Only`
- Task 2: `Physics_Problems_Text_Only`

The default local model is `gemma4:e2b-it-q4_K_M`, called through Ollama at
`http://localhost:11434/api/chat`.

## Important Files

- `scripts/task1_baseline.py`: Task 1 baseline runner.
- `scripts/task2_baseline.py`: Task 2 baseline runner.
- `scripts/evaluate.py`: Modular evaluation CLI, currently with P1 exact match.
- `scripts/unified_api.py`: Official unified Type 1/Type 2 API router.
- `scripts/run_baseline.sh`: Convenience wrapper that does not require `just`.
- `README.md`: User-facing setup, run, and evaluation instructions.
- `justfile`: Optional shortcuts for users with `just`.
- `pyproject.toml`, `requirements.txt`, `uv.lock`: Python dependency metadata.

Local-only paths:

- `EXACT2026_dataset_2026-05-15/`: dataset; ignored by Git.
- `outputs/`: generated model outputs; ignored by Git.
- `eval/`: generated evaluation reports; ignored by Git.
- `.venv/`: local Python environment; ignored by Git.

Do not commit local datasets, outputs, evaluation artifacts, virtualenvs, or
`.DS_Store`.

## Development Commands

Primary check:

```bash
./scripts/run_baseline.sh check
```

Setup without optional tools:

```bash
./scripts/run_baseline.sh setup
```

Smoke runs:

```bash
./scripts/run_baseline.sh task1-test
./scripts/run_baseline.sh task2-test
```

Limited runs:

```bash
./scripts/run_baseline.sh task1 10
./scripts/run_baseline.sh task2 10
```

Full runs:

```bash
./scripts/run_baseline.sh task1
./scripts/run_baseline.sh task2
```

Evaluation:

```bash
./scripts/run_baseline.sh eval outputs/task1_smoke_limit1.json eval/task1_smoke_p1.json
./scripts/run_baseline.sh eval outputs/task2_smoke_limit1.json eval/task2_smoke_p1.json
```

Unified official API-format run:

```bash
./scripts/run_baseline.sh unified unified_input.json outputs/unified_responses.json
```

Optional `uv`/`just` commands exist in `README.md`, but do not require those
tools for normal changes unless the user asks.

## Baseline Behavior

Task 1:

- Reads the Task 1 JSON file.
- Sends each question with natural-language premises to Ollama.
- Optional `--include-fol` includes `premises-FOL`.
- Writes `answer`, `correct_answer`, `explanation`,
  `correct_explanation`, and `raw_response`.
- Valid answer labels are `Yes`, `No`, `Unknown`, `A`, `B`, `C`, `D`.

Task 2:

- Reads the Task 2 CSV file.
- Prompts with the `question` field only. Do not leak source `cot`,
  `answer`, or `unit` into generation prompts.
- Writes `answer`, `unit`, `explanation`, `raw_response`, plus source fields
  `correct_ans`, `correct_cot`, and `correct_unit`.

Both baselines use `requests.post(..., json=payload)` for Ollama calls.

## Unified API Behavior

`scripts/unified_api.py` handles the official merged input stream but keeps
Type 1 and Type 2 solving separate.

- Type 1 is detected when `premises-NL` exists and is non-empty.
- Missing or empty `premises-NL` is routed to Type 2.
- Type 1 routes to the existing Task 1 prompt/parser from `task1_baseline.py`.
- Type 2 routes to the existing Task 2 prompt/parser from `task2_baseline.py`.
- The loader accepts a plain list, a single sample object, or an object with
  `queries`, `samples`, `inputs`, or `data`.
- Each response must include JSON-serializable `answer` and `explanation`.
- Optional response fields are kept only when valid:
  `fol` string, `cot` list of strings, `premises` list of strings, and
  `confidence` number between 0 and 1.

## Output Parsing Gotcha

Ollama sometimes returns JSON-shaped text that is not strict JSON because
explanations can include unescaped LaTeX-style backslashes such as `\mu`,
`\times`, or `\implies`. The scripts intentionally:

1. Try strict `json.loads`.
2. Fall back to extracting JSON-like string fields.

Keep this fallback behavior unless replacing it with a more robust parser.
Prompts also ask the model to avoid LaTeX commands and backslashes.

## Evaluation Design

`scripts/evaluate.py` is meant to be extensible.

Current metric:

- `p1_exact_match`

Current task logic:

- Task 1: `answer == correct_answer`
- Task 2: `answer == correct_ans` and `unit == correct_unit`

The evaluator accepts either:

- a top-level object with `predictions` or `samples`
- a plain list of sample objects

Task detection order:

1. Explicit CLI `--task task1` or `--task task2`
2. Per-sample `task` field or task-specific required fields
3. Top-level file task hint

Missing required fields should produce per-sample `errors` and count as
incorrect, not crash the whole run.

When adding new metrics, follow the existing `Metric` abstraction and keep
task-specific behavior separated from aggregate reporting.

## Coding Conventions

- Use Python 3.11+ type hints.
- Prefer small functions with explicit inputs and JSON-serializable outputs.
- Keep CLI scripts dependency-light.
- Use `requests` for HTTP calls.
- Preserve existing JSON output shapes unless the user asks to change them.
- Keep generated artifacts out of Git.
- Run `./scripts/run_baseline.sh check` before finalizing code changes.
- Add or update tests under `tests/` for unified API behavior.

## Git Notes

Use conventional commit messages when committing, for example:

- `feat(eval): add metric`
- `fix(parser): handle malformed json output`
- `refactor(task1): simplify prompt construction`

The remote is expected to be `origin` on branch `main`.
