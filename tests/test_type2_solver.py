from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exact2026.type2.pipeline import solve_physics_question  # noqa: E402


class Type2SolverTests(unittest.TestCase):
    def assert_answer_close(self, question: str, expected: float, unit: str) -> None:
        result = solve_physics_question(question)

        self.assertAlmostEqual(float(result.answer), expected, delta=max(0.02, abs(expected) * 0.002))
        self.assertEqual(result.unit, unit)
        self.assertTrue(result.explanation)
        self.assertTrue(result.cot)
        self.assertTrue(result.premises)
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertEqual(result.metadata["review_source"], "execution_only")

    def test_inductive_reactance(self) -> None:
        self.assert_answer_close(
            "Find the inductive reactance when L = 0.25 H and f = 60 Hz.",
            94.25,
            "Ω",
        )

    def test_capacitive_reactance(self) -> None:
        self.assert_answer_close(
            "Find the capacitive reactance when C = 75 μF and f = 60 Hz.",
            35.37,
            "Ω",
        )

    def test_ac_power(self) -> None:
        self.assert_answer_close(
            "Find the AC real power when Z = 50 Ω, R = 30 Ω, and V = 150 V.",
            270,
            "W",
        )

    def test_rlc_impedance(self) -> None:
        self.assert_answer_close(
            "Find the impedance of a series RLC circuit with R = 12 Ω, L = 0.2 H, C = 50 μF, and f = 50 Hz.",
            12.03,
            "Ω",
        )

    def test_impedance_from_voltage_current(self) -> None:
        self.assert_answer_close(
            "Find the impedance when U = 100 V and I = 2 A.",
            50,
            "Ω",
        )

    def test_power_dissipated(self) -> None:
        self.assert_answer_close(
            "Find the power dissipated when R = 18 Ω, Z = 36 Ω, and V = 108 V.",
            162,
            "W",
        )

    def test_missing_formula_fallback_uses_model(self) -> None:
        def fallback(prompt: str) -> str:
            return json.dumps(
                {
                    "answer": "3.14",
                    "unit": "rad/s",
                    "formula": "omega = theta / t",
                    "explanation": "Use angular speed formula.",
                    "cot": ["Identify theta and t.", "Apply omega = theta / t."],
                    "premises": ["omega = theta / t"],
                    "confidence": 0.65,
                }
            )

        result = solve_physics_question(
            "A rotating object covers theta = 6.28 rad in t = 2 s. Find angular speed.",
            fallback_model=fallback,
        )

        self.assertEqual(result.answer, "3.14")
        self.assertEqual(result.unit, "rad/s")
        self.assertEqual(result.premises, ["omega = theta / t"])
        self.assertLess(result.confidence, 0.71)

    def test_model_can_generate_equation_steps(self) -> None:
        calls: list[str] = []

        def model_complete(system_prompt: str, user_prompt: str) -> str:
            calls.append(system_prompt)
            if "select physics formulas" in system_prompt:
                return json.dumps({"formula_id": "ac_real_power_vzr"})
            if "code generator agent" in system_prompt:
                return json.dumps(
                    {
                        "code": "\n".join(
                            [
                                "steps = []",
                                "premises = []",
                                "I = V/Z",
                                "steps.append('Compute I = V/Z.')",
                                "answer = I**2*R",
                                "unit = 'W'",
                                "steps.append('Compute P = I^2 R.')",
                                "premises.append('I = V/Z')",
                                "premises.append('P = I^2 R')",
                            ]
                        )
                    }
                )
            if "write concise physics solution steps" in system_prompt:
                return json.dumps(
                    {
                        "explanation": "Use I = V/Z, then P = I^2 R.",
                        "cot": ["Compute I = V/Z.", "Compute P = I^2 R."],
                        "premises": ["I = V/Z", "P = I^2 R"],
                    }
                )
            if "reviewer agent" in system_prompt:
                return json.dumps(
                    {
                        "passed": True,
                        "confidence": 0.82,
                        "errors": [],
                        "feedback": "The code matches the requested AC power calculation.",
                    }
                )
            return "{}"

        result = solve_physics_question(
            "Find the AC real power when Z = 50 Ω, R = 30 Ω, and V = 150 V.",
            model_complete=model_complete,
        )

        self.assertEqual(result.answer, "270")
        self.assertEqual(result.metadata["formula_id"], "ac_real_power_vzr")
        self.assertEqual(result.metadata["review_source"], "slm")
        self.assertIn("code generator agent", " ".join(calls))
        self.assertEqual(result.cot, ["Compute I = V/Z.", "Compute P = I^2 R."])

    def test_ld003_net_coulomb_force(self) -> None:
        result = solve_physics_question(
            "Points A and B are separated by 20 cm in air. Charges q1 = -3 × 10^-6 C "
            "and q2 = 8 × 10^-6 C are placed at A and B, respectively. A test charge "
            "q3 = 2 × 10^-6 C is placed at point C such that AC = 12 cm and BC = 16 cm. "
            "Calculate the magnitude of the electric force acting on q3."
        )

        self.assertAlmostEqual(float(result.answer), 6.76, delta=0.02)
        self.assertEqual(result.unit, "N")
        self.assertEqual(result.metadata["formula_id"], "net_coulomb_force_right_triangle")
        self.assertTrue(result.metadata["verified"])

    def test_planner_produces_series_of_steps(self) -> None:
        result = solve_physics_question(
            "Find the impedance of a series RLC circuit with R = 12 Ω, L = 0.2 H, "
            "C = 50 μF, and f = 50 Hz."
        )

        planner_steps = result.metadata["planner_steps"]
        self.assertIsInstance(planner_steps, list)
        self.assertGreaterEqual(len(planner_steps), 4)
        self.assertTrue(any("intermediate" in step.lower() for step in planner_steps))
        self.assertTrue(result.metadata["verified"])


if __name__ == "__main__":
    unittest.main()
