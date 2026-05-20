from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient  # noqa: E402

from exact2026.app import app  # noqa: E402


class FastApiAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_solve_single_type2_sample(self) -> None:
        with (
            patch("exact2026.type2.pipeline.build_type2_llm", return_value=None),
            patch(
                "task2_baseline.call_ollama",
                return_value='{"answer":"35.37","unit":"Ω","explanation":"Fallback physics solve."}',
            ),
        ):
            response = self.client.post(
                "/solve",
                json={"question": "Find capacitive reactance when C = 75 μF and f = 60 Hz."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["responses"][0]["answer"], "35.37")
        self.assertEqual(body["responses"][0]["unit"], "Ω")

    def test_solve_mixed_batch(self) -> None:
        with (
            patch("task1_baseline.call_ollama", return_value='{"answer":"Yes","explanation":"By rule."}'),
            patch(
                "task2_baseline.call_ollama",
                return_value='{"answer":"50","unit":"Ω","explanation":"Fallback physics solve."}',
            ),
            patch("exact2026.type2.pipeline.build_type2_llm", return_value=None),
        ):
            response = self.client.post(
                "/solve",
                json={
                    "queries": [
                        {
                            "premises-NL": ["If A then B.", "A is true."],
                            "question": "Does B follow?",
                        },
                        {
                            "question": "Find impedance when U = 100 V and I = 2 A.",
                        },
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["responses"][0]["query_type"], "type1")
        self.assertEqual(body["responses"][0]["answer"], "Yes")
        self.assertEqual(body["responses"][1]["query_type"], "type2")
        self.assertEqual(body["responses"][1]["answer"], "50")
        self.assertEqual(body["responses"][1]["unit"], "Ω")


if __name__ == "__main__":
    unittest.main()
