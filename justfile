set shell := ["bash", "-cu"]

setup:
    uv sync

check:
    uv run python -m py_compile scripts/task1_baseline.py scripts/task2_baseline.py scripts/evaluate.py scripts/unified_api.py src/exact2026/app.py
    uv run python -m unittest discover -s tests

task1-test:
    uv run python scripts/task1_baseline.py --limit 1 --output outputs/task1_smoke_limit1.json

task2-test:
    uv run python scripts/task2_baseline.py --limit 1 --output outputs/task2_smoke_limit1.json

task1-10:
    uv run python scripts/task1_baseline.py --limit 10

task2-10:
    uv run python scripts/task2_baseline.py --limit 10 --random-sample

task1:
    uv run python scripts/task1_baseline.py --resume

task2:
    uv run python scripts/task2_baseline.py --resume

unified input output:
    uv run python scripts/unified_api.py --input {{input}} --output {{output}}

type2-debug question:
    uv run python scripts/debug_type2_flow.py --question "{{question}}"

type2-debug-ollama question:
    uv run python scripts/debug_type2_flow.py --use-ollama --question "{{question}}"

serve:
    uv run uvicorn exact2026.app:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}

eval-task1:
    uv run python scripts/evaluate.py --input outputs/task1_smoke_limit1.json --output eval/task1_smoke_p1.json

eval-task2:
    uv run python scripts/evaluate.py --input outputs/task2_smoke_limit1.json --output eval/task2_smoke_p1.json
