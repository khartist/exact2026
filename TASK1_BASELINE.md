# Task 1 Baseline

This baseline runs the Task 1 JSON through a local Ollama chat model and writes
one `Yes` / `No` / `Unknown` answer plus a logic explanation per question.

## Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

No Python packages are required beyond the standard library.

## Quick tests

Run only the first record:

```bash
uv run python scripts/task1_baseline.py --limit 1
```

Run the first 10 records:

```bash
uv run python scripts/task1_baseline.py --limit 10
```

## Full run

```bash
uv run python scripts/task1_baseline.py
```

The default output path is:

```text
outputs/task1_gemma4_e2b_baseline.json
```

Useful options:

- `--model gemma4:e2b-it-q4_K_M`: local Ollama model name.
- `--start N`: start from zero-based record index `N`.
- `--limit N`: run only `N` records.
- `--resume`: continue an existing output file and skip completed questions.
- `--include-fol`: include `premises-FOL` alongside `premises-NL` in the prompt.
- `--output PATH`: choose a different output path for test runs.
