# EXACT 2026 Baselines

Baseline runners for the EXACT 2026 text-only tasks using a local Ollama model.

The scripts currently support:

- Task 1: `Logic_Based_Educational_Queries_Text_Only`
- Task 2: `Physics_Problems_Text_Only`

Both runners call the local Ollama chat API at `http://localhost:11434/api/chat`
and default to the model tag `gemma4:e2b-it-q4_K_M`.

Task 2 uses a structured LangGraph pipeline:

- planner agent
- code generator agent
- deterministic planner validator
- sandboxed code executor
- deterministic execution validator
- answer composer

Before planning, the pipeline runs a lightweight text search over the local
formula/law bank. The planner receives the raw question plus the top search
results and is responsible for extracting givens, target, units, SI
conversions, and calculation steps. It preserves the stated value and unit in
each given, such as `mC`, `mm`, `cm`, or `μF`; `si_value` and `si_unit` exist
for executor consistency and only need conversion when the calculation formula
requires consistent units. There is no regex/parser fallback that extracts
known variables or builds a deterministic physics plan. If the planner does not
return a valid plan, the graph returns a structured fallback and the baseline
can then use the normal direct model fallback response path.

Task 1 reads the JSON dataset and writes one model `answer`, source
`correct_answer`, model `explanation`, source `correct_explanation`, and
`raw_response` per question. Task 2 reads the physics CSV and writes one
numeric/symbolic `answer`, `unit`, `explanation`, and `raw_response` per row;
it prompts with the `question` field only.

## Type 2 Structured Solver

The Type 2 solver lives under `src/exact2026/type2/`.

Important pieces:

- `knowledge_search.py`: token-overlap text search over formulas and laws.
- `formula_bank.py`: local trusted formula/law entries.
- `agents/planner_agent.py`: asks the model for a machine-readable calculation
  plan from the raw question and search context.
- `validation/planner_validator.py`: checks plan structure and internal
  consistency without inferring physics facts.
- `agents/code_generator_agent.py`: generates code from the validated plan.
- `execution/code_executor.py`: executes generated code in a restricted
  environment.
- `validation/execution_validator.py`: checks trace/final result consistency.

The planner output shape is:

```json
{
  "target": {"symbol": "X_C", "description": "capacitive reactance", "unit": "Ω"},
  "givens": {
    "C": {"value": 75, "unit": "μF", "si_value": 0.000075, "si_unit": "F"},
    "f": {"value": 60, "unit": "Hz", "si_value": 60, "si_unit": "Hz"}
  },
  "steps": [
    {
      "id": "s1",
      "goal": "Compute capacitive reactance.",
      "output": "X_C",
      "formula_id": "capacitive_reactance",
      "formula": "X_C = 1/(2*pi*f*C)",
      "inputs": ["f", "C"],
      "unit": "Ω",
      "premise": "Capacitive reactance: X_C = 1/(2πfC)"
    }
  ],
  "final_step": "s1",
  "missing_information": [],
  "status": "READY"
}
```

`READY` plans must include declared givens with `value`, `unit`, `si_value`,
and `si_unit`; step inputs must come from givens or previous outputs; formulas
must reference only declared inputs and allowed math names. Constants such as
`k` must be explicitly included in `givens` if used.
The `value`/`unit` pair should match the problem statement when practical;
converted values belong in `si_value`/`si_unit` and are optional unless the
formula requires consistent units.

Each step `premise` should state the formula or physics law used for that step.
The planner should prefer formulas and laws from the search context, but it may
use standard physics knowledge when the local formula bank does not contain the
needed item.

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

With `uv`:

```bash
uv sync
```

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

The `just task2-10` shortcut differs from the helper script: it randomly
samples 10 Task 2 rows.

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

Optional fields such as `fol`, `cot`, and `premises` are included when
available and valid. Type 2 responses do not include a confidence score.

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

## Type 2 Debugging

Print the Type 2 graph inputs and outputs for one question:

```bash
just type2-debug "Find the capacitive reactance when C = 75 μF and f = 60 Hz."
```

Use Ollama for the planner/code-generation calls:

```bash
just type2-debug-ollama "Find the capacitive reactance when C = 75 μF and f = 60 Hz."
```

The debug output includes the raw question, search query, top formula/law
results, planner output, planner validation, code generation, execution,
execution validation, and final composed output or fallback.

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

Task 2 also supports:

- `--random-sample`: randomly choose rows from the selected range.
- `--seed N`: make `--random-sample` reproducible.

## Optional uv And just Commands

The baseline does not require `uv` or `just`. If you prefer them, install from:

- uv: <https://docs.astral.sh/uv/getting-started/installation/>
- just: <https://github.com/casey/just>

Optional `uv` commands:

```bash
uv sync
uv run python -m unittest discover -s tests
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
just task2-10  # random 10 Task 2 rows
just task1
just task2
just unified unified_input.json outputs/unified_responses.json
just type2-debug "Find the impedance when U = 100 V and I = 2 A."
just type2-debug-ollama "Find the impedance when U = 100 V and I = 2 A."
just eval-task1
just eval-task2
```
