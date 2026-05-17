# EXACT 2026 Baselines

Baseline runners for the EXACT 2026 text-only tasks using a local Ollama model.

The scripts currently support:

- Task 1: `Logic_Based_Educational_Queries_Text_Only`
- Task 2: `Physics_Problems_Text_Only`

Both runners call the local Ollama chat API at `http://localhost:11434/api/chat`
and default to the model tag `gemma4:e2b-it-q4_K_M`.

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

## Direct Commands

Task 1:

```bash
.venv/bin/python scripts/task1_baseline.py --limit 1
.venv/bin/python scripts/task1_baseline.py --limit 10
.venv/bin/python scripts/task1_baseline.py
```

Task 2:

```bash
.venv/bin/python scripts/task2_baseline.py --limit 1
.venv/bin/python scripts/task2_baseline.py --limit 10
.venv/bin/python scripts/task2_baseline.py
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
```
