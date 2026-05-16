set shell := ["bash", "-cu"]

setup:
    uv venv --python 3.11

check:
    uv run python -m py_compile scripts/task1_baseline.py scripts/task2_baseline.py

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

