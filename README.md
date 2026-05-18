# EXACT 2026 Baselines

Baseline runners for the EXACT 2026 text-only tasks using a local Ollama model.

The scripts currently support:

- Task 1: `Logic_Based_Educational_Queries_Text_Only`
- Task 2: `Physics_Problems_Text_Only`

Both runners call the local Ollama chat API at `http://localhost:11434/api/chat`
and default to the model tag `gemma4:e2b-it-q4_K_M`.

Task 2 now uses a LangGraph agent loop:

- planner agent
- code generator agent
- reviewer agent
- up to 3 loops before returning the nearest executable result

The planner first queries the formula bank. If it finds a useful formula, it can
choose a formula-bank path or continue with a multi-step executable plan. If no
formula-bank entry fits, the planner can still generate internal solution steps
and send them to the code generator. The reviewer combines SLM judgment with
backward consistency checks on the executed steps.

Task 1 reads the JSON dataset and writes one model `answer`, source
`correct_answer`, model `explanation`, source `correct_explanation`, and
`raw_response` per question. Task 2 reads the physics CSV and writes one
numeric/symbolic `answer`, `unit`, `explanation`, and `raw_response` per row;
it prompts with the `question` field only.

## Requirements

Install these if they are not already available on your machine:

- Ollama: <https://ollama.com/download>
- Python 3.11 or newer: <https://www.python.org/downloads/>

Make sure Ollama is running and the model is available:

```bash
ollama list
```

## Setup

Use the helper script:

```bash
./scripts/run_baseline.sh setup
```

This creates `.venv/` and installs the Python dependencies from
`requirements.txt`.

## Quick Tests

Run Task 1 on the first record:

```bash
./scripts/run_baseline.sh task1-test
```

Run Task 2 on the first row:

```bash
./scripts/run_baseline.sh task2-test
```

Run the first 10 records/rows:

```bash
./scripts/run_baseline.sh task1 10
./scripts/run_baseline.sh task2 10
```

## Full Runs

Task 1:

```bash
./scripts/run_baseline.sh task1
```

Task 2:

```bash
./scripts/run_baseline.sh task2
```

## Output

Default outputs are written to:

```text
outputs/task1_gemma4_e2b_baseline.json
outputs/task2_gemma4_e2b_baseline.json
```

Smoke test outputs are written to:

```text
outputs/task1_smoke_limit1.json
outputs/task2_smoke_limit1.json
```

The `outputs/` directory is ignored by Git.

## Unified API Format

The official test set can mix Type 1 and Type 2 samples in one JSON stream.
Use `scripts/unified_api.py` for that format. The unified API only shares input
loading and routing; Type 1 and Type 2 still use their task-specific solver
prompts and response parsers.

Type 1 samples contain non-empty `premises-NL` and `question`:

```json
{
  "premises-NL": ["If A then B.", "A is true."],
  "question": "Does B follow?"
}
```

Type 2 samples only need `question`:

```json
{
  "question": "Calculate the energy stored in capacitor C."
}
```

The router detects Type 1 when `premises-NL` exists and is non-empty. Missing
or empty `premises-NL` is routed to Type 2. The loader accepts a plain list of
samples, a single sample object, or an object containing `queries`, `samples`,
`inputs`, or `data`.

Run unified samples:

```bash
./scripts/run_baseline.sh unified unified_input.json outputs/unified_responses.json
```

With `just`:

```bash
just unified unified_input.json outputs/unified_responses.json
```

Each response is JSON-serializable and always contains at least:

```json
{
  "answer": "...",
  "explanation": "..."
}
```

Optional fields such as `fol`, `cot`, `premises`, and `confidence` are included
when available and valid.

## FastAPI

Run the unified solver as an API server:

```bash
./scripts/run_baseline.sh serve
```

With `just`:

```bash
just serve
```

With `uv` directly:

```bash
uv run uvicorn exact2026.app:app --app-dir src --host 0.0.0.0 --port 8000
```

The server exposes:

- `GET /health`
- `POST /solve`

`POST /solve` accepts the same unified sample shapes as `scripts/unified_api.py`
and returns the same response objects.

## Evaluation

Evaluate a pre-run output JSON with P1 exact match:

```bash
./scripts/run_baseline.sh eval outputs/task1_smoke_limit1.json eval/task1_smoke_p1.json
./scripts/run_baseline.sh eval outputs/task2_smoke_limit1.json eval/task2_smoke_p1.json
```

With `just` for the smoke outputs:

```bash
just eval-task1
just eval-task2
```

The evaluator accepts either a top-level `{"predictions": [...]}` object or a
plain list of sample objects. By default it infers the task per sample from the
available fields. You can override task detection:

```bash
.venv/bin/python scripts/evaluate.py \
  --input outputs/task1_smoke_limit1.json \
  --output eval/task1_smoke_p1.json \
  --task task1
```

P1 exact match currently uses:

- Task 1: `answer == correct_answer`
- Task 2: `answer == correct_ans` and `unit == correct_unit`

Each evaluation JSON contains per-sample correctness, field comparisons,
missing-field errors, total sample count, correct sample count, and accuracy.

## Direct Commands

Task 1:

```bash
.venv/bin/python scripts/task1_baseline.py --limit 1
.venv/bin/python scripts/task1_baseline.py --limit 10
.venv/bin/python scripts/task1_baseline.py
```

The default Task 1 output path is:

```text
outputs/task1_gemma4_e2b_baseline.json
```

Task 2:

```bash
.venv/bin/python scripts/task2_baseline.py --limit 1
.venv/bin/python scripts/task2_baseline.py --limit 10
.venv/bin/python scripts/task2_baseline.py
```

The default Task 2 output path is:

```text
outputs/task2_gemma4_e2b_baseline.json
```

Useful shared options:

- `--model MODEL`: override the local Ollama model tag.
- `--start N`: start from zero-based record/row index `N`.
- `--limit N`: run only `N` records/rows.
- `--resume`: continue an existing output file and skip completed items.
- `--output PATH`: write predictions to a custom path.

Task 1 also supports:

- `--include-fol`: include `premises-FOL` alongside `premises-NL`.

## Optional uv And just Commands

The baseline does not require `uv` or `just`. If you prefer them, install from:

- uv: <https://docs.astral.sh/uv/getting-started/installation/>
- just: <https://github.com/casey/just>

Optional `uv` commands:

```bash
uv sync
uv run python scripts/task1_baseline.py --limit 1
uv run python scripts/task2_baseline.py --limit 1
```

Optional `just` shortcuts:

```bash
just setup
just check
just task1-test
just task2-test
just task1-10
just task2-10
just task1
just task2
just unified unified_input.json outputs/unified_responses.json
just eval-task1
just eval-task2
```
