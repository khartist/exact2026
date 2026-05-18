"""FastAPI app for the EXACT 2026 unified solver."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import task1_baseline  # noqa: E402
import task2_baseline  # noqa: E402
import unified_api  # noqa: E402


DEFAULT_MODEL = os.getenv("EXACT_MODEL", unified_api.DEFAULT_MODEL)
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", unified_api.DEFAULT_OLLAMA_URL)
DEFAULT_TIMEOUT = float(os.getenv("EXACT_TIMEOUT", "180"))
DEFAULT_TEMPERATURE = float(os.getenv("EXACT_TEMPERATURE", "0"))

app = FastAPI(title="EXACT 2026 MVP Solver", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "ollama_url": DEFAULT_OLLAMA_URL,
    }


@app.post("/solve")
def solve(payload: Any = Body(...)) -> dict[str, Any]:
    samples = normalize_payload(payload)
    responses: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        query = unified_api.normalize_query(sample)
        response = unified_api.solve_unified_query(query, api_model_fn)
        validation_errors = unified_api.validate_api_response(response)
        if validation_errors:
            response["validation_errors"] = validation_errors
        response["sample_index"] = sample_index
        response["query_type"] = query.query_type
        responses.append(response)
    return {"responses": responses}


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return unified_api.validate_sample_list(payload)
    if isinstance(payload, dict):
        for key in ("queries", "samples", "inputs", "data"):
            value = payload.get(key)
            if value is not None:
                return unified_api.validate_sample_list(value)
        if "question" in payload:
            return [payload]
    raise ValueError("Payload must be a sample, list of samples, or unified wrapper object.")


def api_model_fn(prompt: str, query_type: unified_api.QueryType) -> str:
    if query_type == "type1":
        return task1_baseline.call_ollama(
            DEFAULT_OLLAMA_URL,
            DEFAULT_MODEL,
            prompt,
            DEFAULT_TEMPERATURE,
            DEFAULT_TIMEOUT,
        )
    return task2_baseline.call_ollama(
        DEFAULT_OLLAMA_URL,
        DEFAULT_MODEL,
        prompt,
        DEFAULT_TEMPERATURE,
        DEFAULT_TIMEOUT,
    )
