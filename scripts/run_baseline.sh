#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_baseline.sh setup
  ./scripts/run_baseline.sh check
  ./scripts/run_baseline.sh task1-test
  ./scripts/run_baseline.sh task2-test
  ./scripts/run_baseline.sh task1 [limit]
  ./scripts/run_baseline.sh task2 [limit]
  ./scripts/run_baseline.sh unified INPUT_JSON OUTPUT_JSON [limit]
  ./scripts/run_baseline.sh eval INPUT_JSON OUTPUT_JSON [task1|task2|auto]

Examples:
  ./scripts/run_baseline.sh setup
  ./scripts/run_baseline.sh task1 10
  ./scripts/run_baseline.sh task2
  ./scripts/run_baseline.sh unified unified_input.json outputs/unified.json 10
  ./scripts/run_baseline.sh eval outputs/task1_smoke_limit1.json eval/task1_p1.json
EOF
}

cmd="${1:-}"
limit="${2:-}"
python_bin="${PYTHON_BIN:-}"

if [[ -z "$python_bin" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

case "$cmd" in
  setup)
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ;;
  check)
    "$python_bin" -m py_compile scripts/task1_baseline.py scripts/task2_baseline.py scripts/evaluate.py scripts/unified_api.py
    "$python_bin" -m unittest discover -s tests
    ;;
  task1-test)
    "$python_bin" scripts/task1_baseline.py --limit 1 --output outputs/task1_smoke_limit1.json
    ;;
  task2-test)
    "$python_bin" scripts/task2_baseline.py --limit 1 --output outputs/task2_smoke_limit1.json
    ;;
  task1)
    if [[ -n "$limit" ]]; then
      "$python_bin" scripts/task1_baseline.py --limit "$limit"
    else
      "$python_bin" scripts/task1_baseline.py --resume
    fi
    ;;
  task2)
    if [[ -n "$limit" ]]; then
      "$python_bin" scripts/task2_baseline.py --limit "$limit"
    else
      "$python_bin" scripts/task2_baseline.py --resume
    fi
    ;;
  unified)
    input="${2:-}"
    output="${3:-}"
    limit="${4:-}"
    if [[ -z "$input" || -z "$output" ]]; then
      usage
      exit 1
    fi
    if [[ -n "$limit" ]]; then
      "$python_bin" scripts/unified_api.py --input "$input" --output "$output" --limit "$limit"
    else
      "$python_bin" scripts/unified_api.py --input "$input" --output "$output"
    fi
    ;;
  eval)
    input="${2:-}"
    output="${3:-}"
    task="${4:-auto}"
    if [[ -z "$input" || -z "$output" ]]; then
      usage
      exit 1
    fi
    "$python_bin" scripts/evaluate.py --input "$input" --output "$output" --task "$task"
    ;;
  *)
    usage
    exit 1
    ;;
esac
