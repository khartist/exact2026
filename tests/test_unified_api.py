from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.unified_api import (  # noqa: E402
    load_unified_samples,
    normalize_api_response,
    normalize_query,
    solve_unified_query,
    validate_api_response,
)


class UnifiedApiTests(unittest.TestCase):
    def test_type1_input_with_premises_and_question(self) -> None:
        query = normalize_query(
            {
                "premises-NL": ["If A then B.", "A is true."],
                "question": "Does B follow?",
            }
        )

        self.assertEqual(query.query_type, "type1")
        self.assertEqual(query.premises, ["If A then B.", "A is true."])
        self.assertEqual(query.question, "Does B follow?")

    def test_type2_input_with_only_question(self) -> None:
        query = normalize_query({"question": "Calculate the energy stored."})

        self.assertEqual(query.query_type, "type2")
        self.assertEqual(query.premises, [])
        self.assertEqual(query.question, "Calculate the energy stored.")

    def test_missing_premises_nl_is_type2(self) -> None:
        query = normalize_query({"question": "What is the answer?"})

        self.assertEqual(query.query_type, "type2")

    def test_empty_premises_nl_is_type2(self) -> None:
        query = normalize_query({"premises-NL": [], "question": "What is the answer?"})

        self.assertEqual(query.query_type, "type2")

    def test_type1_routes_to_logic_solver_prompt(self) -> None:
        query = normalize_query(
            {
                "premises-NL": ["If A then B.", "A is true."],
                "question": "Does B follow?",
            }
        )
        calls: list[tuple[str, str]] = []

        response = solve_unified_query(query, self.fake_model(calls))

        self.assertEqual(calls[0][1], "type1")
        self.assertIn("Premises in natural language:", calls[0][0])
        self.assertIn("If A then B.", calls[0][0])
        self.assertEqual(response["answer"], "Yes")
        self.assertEqual(response["premises"], ["If A then B.", "A is true."])

    def test_type2_routes_to_physics_solver_prompt(self) -> None:
        query = normalize_query({"question": "Calculate the energy stored."})
        calls: list[tuple[str, str]] = []

        response = solve_unified_query(query, self.fake_model(calls))

        self.assertEqual(calls[0][1], "type2")
        self.assertIn("Physics problem:", calls[0][0])
        self.assertNotIn("Premises in natural language:", calls[0][0])
        self.assertEqual(response["answer"], "0.045 J")

    def test_output_contains_answer_and_explanation(self) -> None:
        query = normalize_query(
            {
                "premises-NL": ["If A then B.", "A is true."],
                "question": "Does B follow?",
            }
        )

        response = solve_unified_query(query, self.fake_model([]))

        self.assertIn("answer", response)
        self.assertIn("explanation", response)
        self.assertEqual(validate_api_response(response), [])

    def test_optional_fields_have_valid_types_when_present(self) -> None:
        response = normalize_api_response(
            {
                "answer": "Yes",
                "explanation": "A concise explanation.",
                "fol": "forall x P(x)",
                "cot": ["Step 1: Read the premises.", "Step 2: Apply the rule."],
                "premises": ["If A then B.", "A is true."],
                "confidence": 0.92,
            }
        )

        self.assertIsInstance(response["fol"], str)
        self.assertTrue(all(isinstance(item, str) for item in response["cot"]))
        self.assertTrue(all(isinstance(item, str) for item in response["premises"]))
        self.assertIsInstance(response["confidence"], (int, float))
        self.assertGreaterEqual(response["confidence"], 0)
        self.assertLessEqual(response["confidence"], 1)
        self.assertEqual(validate_api_response(response), [])

    def test_loader_accepts_single_sample_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            path.write_text(json.dumps({"question": "Only one?"}), encoding="utf-8")

            samples = load_unified_samples(path)

        self.assertEqual(samples, [{"question": "Only one?"}])

    @staticmethod
    def fake_model(calls: list[tuple[str, str]]):
        def model(prompt: str, query_type: str) -> str:
            calls.append((prompt, query_type))
            if query_type == "type1":
                return json.dumps(
                    {
                        "answer": "Yes",
                        "explanation": "B follows by modus ponens.",
                        "cot": ["Step 1: Use the implication.", "Step 2: Apply A."],
                        "confidence": 0.9,
                    }
                )
            return json.dumps(
                {
                    "answer": "0.045",
                    "unit": "J",
                    "explanation": "Use E = 0.5 C U^2.",
                    "cot": ["Step 1: Convert C.", "Step 2: Substitute values."],
                    "confidence": 0.8,
                }
            )

        return model


if __name__ == "__main__":
    unittest.main()
