set shell := ["bash", "-cu"]

setup:
    uv sync

check:
    uv run python -m py_compile scripts/task1_baseline.py scripts/task2_baseline.py scripts/evaluate.py

task1-test:
    uv run python scripts/task1_baseline.py --limit 1 --output outputs/task1_smoke_limit1.json

task2-test:
    uv run python scripts/task2_baseline.py --limit 1 --output outputs/task2_smoke_limit1.json

task1-10:
    uv run python scripts/task1_baseline.py --limit 10

task2-10:
    uv run python scripts/task2_baseline.py --limit 10

task1:
    uv run python scripts/task1_baseline.py --resume

task2:
    uv run python scripts/task2_baseline.py --resume

eval-task1:
    uv run python scripts/evaluate.py --input outputs/task1_smoke_limit1.json --output eval/task1_smoke_p1.json

eval-task2:
    uv run python scripts/evaluate.py --input outputs/task2_smoke_limit1.json --output eval/task2_smoke_p1.json
