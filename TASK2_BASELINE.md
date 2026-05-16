# Task 2 Baseline

This baseline runs the Task 2 physics CSV through a local Ollama chat model and
writes one numeric/symbolic `answer`, `unit`, and physics `explanation` per row.
It prompts with the `question` field only.

## Quick tests

Run only the first row:

```bash
uv run python scripts/task2_baseline.py --limit 1
```

Run the first 10 rows:

```bash
uv run python scripts/task2_baseline.py --limit 10
```

## Full run

```bash
uv run python scripts/task2_baseline.py
```

The default output path is:

```text
outputs/task2_gemma4_e2b_baseline.json
```

Useful options:

- `--model gemma4:e2b-it-q4_K_M`: local Ollama model name.
- `--start N`: start from zero-based row index `N`.
- `--limit N`: run only `N` rows.
- `--resume`: continue an existing output file and skip completed rows.
- `--output PATH`: choose a different output path for test runs.
