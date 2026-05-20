from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exact2026.type2.pipeline import solve_physics_question  # noqa: E402
from exact2026.type2.knowledge_search import search_physics_knowledge  # noqa: E402
from exact2026.type2.validation.execution_validator import validate_structured_execution  # noqa: E402
from exact2026.type2.validation.planner_validator import validate_calculation_plan  # noqa: E402


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, handler):
        self.handler = handler

    def invoke(self, messages):
        text = "\n".join(str(message.get("content", message)) for message in messages)
        return FakeMessage(self.handler(text))


class Type2SolverTests(unittest.TestCase):
    def solve_with_plan(self, question: str, plan: dict[str, Any]):
        prompts: list[str] = []

        def handler(prompt: str) -> str:
            prompts.append(prompt)
            if "planner agent" in prompt:
                return json.dumps(plan)
            return "{}"

        return solve_physics_question(question, llm=FakeLLM(handler)), prompts

    def assert_plan_answer_close(
        self,
        question: str,
        plan: dict[str, Any],
        expected: float,
        unit: str,
    ) -> None:
        result, _ = self.solve_with_plan(question, plan)

        self.assertAlmostEqual(float(result.answer), expected, delta=max(0.02, abs(expected) * 0.002))
        self.assertEqual(result.unit, unit)
        self.assertTrue(result.explanation)
        self.assertTrue(result.cot)
        self.assertTrue(result.premises)
        self.assertEqual(result.metadata["agent_loop"], "structured_langgraph")
        self.assertTrue(result.metadata["verified"])

    def test_inductive_reactance(self) -> None:
        self.assert_plan_answer_close(
            "Find the inductive reactance when L = 0.25 H and f = 60 Hz.",
            {
                "target": {"symbol": "X_L", "description": "inductive reactance", "unit": "Ω"},
                "givens": {
                    "L": {"value": 0.25, "unit": "H", "si_value": 0.25, "si_unit": "H"},
                    "f": {"value": 60, "unit": "Hz", "si_value": 60, "si_unit": "Hz"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute inductive reactance.", "output": "X_L", "formula_id": "inductive_reactance", "formula": "X_L = 2*pi*f*L", "inputs": ["f", "L"], "unit": "Ω", "premise": "Inductive reactance: X_L = 2πfL"}
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            },
            94.25,
            "Ω",
        )

    def test_capacitive_reactance(self) -> None:
        self.assert_plan_answer_close(
            "Find the capacitive reactance when C = 75 μF and f = 60 Hz.",
            {
                "target": {"symbol": "X_C", "description": "capacitive reactance", "unit": "Ω"},
                "givens": {
                    "C": {"value": 75, "unit": "μF", "si_value": 75e-6, "si_unit": "F"},
                    "f": {"value": 60, "unit": "Hz", "si_value": 60, "si_unit": "Hz"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute capacitive reactance.", "output": "X_C", "formula_id": "capacitive_reactance", "formula": "X_C = 1/(2*pi*f*C)", "inputs": ["f", "C"], "unit": "Ω", "premise": "Capacitive reactance: X_C = 1/(2πfC)"}
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            },
            35.37,
            "Ω",
        )

    def test_ac_power(self) -> None:
        self.assert_plan_answer_close(
            "Find the AC real power when Z = 50 Ω, R = 30 Ω, and V = 150 V.",
            {
                "target": {"symbol": "P", "description": "real power", "unit": "W"},
                "givens": {
                    "Z": {"value": 50, "unit": "Ω", "si_value": 50, "si_unit": "Ω"},
                    "R": {"value": 30, "unit": "Ω", "si_value": 30, "si_unit": "Ω"},
                    "V": {"value": 150, "unit": "V", "si_value": 150, "si_unit": "V"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute current.", "output": "I", "formula_id": "ac_current", "formula": "I = V / Z", "inputs": ["V", "Z"], "unit": "A", "premise": "AC current: I = V/Z"},
                    {"id": "s2", "goal": "Compute real power.", "output": "P", "formula_id": "ac_real_power_vzr", "formula": "P = I**2 * R", "inputs": ["I", "R"], "unit": "W", "premise": "AC real power: P = I^2 R"},
                ],
                "final_step": "s2",
                "missing_information": [],
                "status": "READY",
            },
            270,
            "W",
        )

    def test_rlc_impedance(self) -> None:
        self.assert_plan_answer_close(
            "Find the impedance of a series RLC circuit with R = 12 Ω, L = 0.2 H, C = 50 μF, and f = 50 Hz.",
            rlc_plan(),
            12.03,
            "Ω",
        )

    def test_impedance_from_voltage_current(self) -> None:
        self.assert_plan_answer_close(
            "Find the impedance when U = 100 V and I = 2 A.",
            {
                "target": {"symbol": "Z", "description": "impedance", "unit": "Ω"},
                "givens": {
                    "V": {"value": 100, "unit": "V", "si_value": 100, "si_unit": "V"},
                    "I": {"value": 2, "unit": "A", "si_value": 2, "si_unit": "A"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute impedance.", "output": "Z", "formula_id": "ohm_resistance", "formula": "Z = V / I", "inputs": ["V", "I"], "unit": "Ω", "premise": "Impedance follows Ohm law Z = V/I."}
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            },
            50,
            "Ω",
        )

    def test_power_dissipated(self) -> None:
        self.assert_plan_answer_close(
            "Find the power dissipated when R = 18 Ω, Z = 36 Ω, and V = 108 V.",
            {
                "target": {"symbol": "P", "description": "power dissipated", "unit": "W"},
                "givens": {
                    "R": {"value": 18, "unit": "Ω", "si_value": 18, "si_unit": "Ω"},
                    "Z": {"value": 36, "unit": "Ω", "si_value": 36, "si_unit": "Ω"},
                    "V": {"value": 108, "unit": "V", "si_value": 108, "si_unit": "V"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute real power.", "output": "P", "formula_id": "ac_real_power_vzr", "formula": "P = (V/Z)**2 * R", "inputs": ["V", "Z", "R"], "unit": "W", "premise": "AC real power: P = (V/Z)^2 R"}
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            },
            162,
            "W",
        )

    def test_text_search_finds_capacitor_energy(self) -> None:
        results = search_physics_knowledge("energy stored in a capacitor with capacitance and voltage")

        self.assertEqual(results[0]["id"], "capacitor_energy")

    def test_text_search_finds_coulomb_vector_force(self) -> None:
        results = search_physics_knowledge(
            "charges at triangle points calculate magnitude of net electric force vector"
        )
        ids = {item["id"] for item in results}

        self.assertIn("coulomb_force", ids)
        self.assertTrue(
            {"net_coulomb_force_right_triangle", "net_coulomb_force_equilateral", "coulomb_superposition"}.intersection(ids)
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
                }
            )

        result = solve_physics_question(
            "A rotating object covers theta = 6.28 rad in t = 2 s. Find angular speed.",
            fallback_model=fallback,
        )

        self.assertEqual(result.answer, "3.14")
        self.assertEqual(result.unit, "rad/s")
        self.assertEqual(result.premises, ["omega = theta / t"])
        self.assertNotIn("confidence", result.to_api_dict())

    def test_malformed_fallback_json_still_extracts_answer_unit_and_cot(self) -> None:
        def fallback(prompt: str) -> str:
            return (
                "{\n"
                '"answer": 1.556e-26\n'
                '"unit": "N"\n'
                '"explanation": "1. Identify charges.\\n2. Apply Coulomb law.\\n'
                '3. Combine vectors using $F_{net}=\\\\sqrt{3}F$."\n'
                "}"
            )

        result = solve_physics_question(
            "Unsupported fallback-only question with no parseable knowns.",
            fallback_model=fallback,
        )

        self.assertEqual(result.answer, "1.556e-26")
        self.assertEqual(result.unit, "N")
        self.assertIn("Coulomb", result.explanation)
        self.assertGreaterEqual(len(result.cot), 3)

    def test_fake_planner_llm_produces_valid_plan_without_parser_help(self) -> None:
        plan = {
            "target": {"symbol": "P", "description": "real power", "unit": "W"},
            "givens": {
                "Z": {"value": 50, "unit": "Ω", "si_value": 50, "si_unit": "Ω"},
                "R": {"value": 30, "unit": "Ω", "si_value": 30, "si_unit": "Ω"},
                "V": {"value": 150, "unit": "V", "si_value": 150, "si_unit": "V"},
            },
            "steps": [
                {"id": "s1", "goal": "Compute current.", "output": "I", "formula_id": "ac_current", "formula": "I = V / Z", "inputs": ["V", "Z"], "unit": "A", "premise": "AC current: I = V/Z"},
                {"id": "s2", "goal": "Compute real power.", "output": "P", "formula_id": "ac_real_power_vzr", "formula": "P = I**2 * R", "inputs": ["I", "R"], "unit": "W", "premise": "Power: P = I^2 R"},
            ],
            "final_step": "s2",
            "missing_information": [],
            "status": "READY",
        }
        result, prompts = self.solve_with_plan(
            "Find the AC real power when Z = 50 Ω, R = 30 Ω, and V = 150 V.",
            plan,
        )

        self.assertEqual(result.answer, "270")
        self.assertIn("physics_knowledge", " ".join(prompts))
        self.assertNotIn("parsed_context", " ".join(prompts))
        self.assertEqual(result.unit, "W")
        self.assertEqual(result.premises, ["AC current: I = V/Z", "Power: P = I^2 R"])

    def test_ld003_net_coulomb_force(self) -> None:
        result, _ = self.solve_with_plan(
            (
                "Points A and B are separated by 20 cm in air. Charges q1 = -3 × 10^-6 C "
                "and q2 = 8 × 10^-6 C are placed at A and B, respectively. A test charge "
                "q3 = 2 × 10^-6 C is placed at point C such that AC = 12 cm and BC = 16 cm. "
                "Calculate the magnitude of the electric force acting on q3."
            ),
            {
                "target": {"symbol": "F", "description": "net force magnitude", "unit": "N"},
                "givens": {
                    "k": {"value": 8.9875517923e9, "unit": "N m^2/C^2", "si_value": 8.9875517923e9, "si_unit": "N m^2/C^2"},
                    "q1": {"value": -3e-6, "unit": "C", "si_value": -3e-6, "si_unit": "C"},
                    "q2": {"value": 8e-6, "unit": "C", "si_value": 8e-6, "si_unit": "C"},
                    "q3": {"value": 2e-6, "unit": "C", "si_value": 2e-6, "si_unit": "C"},
                    "AC": {"value": 12, "unit": "cm", "si_value": 0.12, "si_unit": "m"},
                    "BC": {"value": 16, "unit": "cm", "si_value": 0.16, "si_unit": "m"},
                },
                "steps": [
                    {"id": "s1", "goal": "Compute force from q1.", "output": "F13", "formula_id": "net_coulomb_force_right_triangle", "formula": "F13 = k*Abs(q1*q3)/AC**2", "inputs": ["k", "q1", "q3", "AC"], "unit": "N", "premise": "Coulomb force magnitude F = k |q_i q_j| / r^2."},
                    {"id": "s2", "goal": "Compute force from q2.", "output": "F23", "formula_id": "net_coulomb_force_right_triangle", "formula": "F23 = k*Abs(q2*q3)/BC**2", "inputs": ["k", "q2", "q3", "BC"], "unit": "N", "premise": "Coulomb force magnitude F = k |q_i q_j| / r^2."},
                    {"id": "s3", "goal": "Combine perpendicular forces.", "output": "F", "formula_id": "net_coulomb_force_right_triangle", "formula": "F = sqrt(F13**2 + F23**2)", "inputs": ["F13", "F23"], "unit": "N", "premise": "Perpendicular electric force components combine by vector magnitude."},
                ],
                "final_step": "s3",
                "missing_information": [],
                "status": "READY",
            },
        )

        self.assertAlmostEqual(float(result.answer), 6.76, delta=0.02)
        self.assertEqual(result.unit, "N")
        self.assertEqual(result.metadata["formula_id"], "net_coulomb_force_right_triangle")
        self.assertTrue(result.metadata["verified"])

    def test_planner_produces_series_of_steps(self) -> None:
        result, _ = self.solve_with_plan(
            "Find the impedance of a series RLC circuit with R = 12 Ω, L = 0.2 H, "
            "C = 50 μF, and f = 50 Hz.",
            rlc_plan(),
        )

        self.assertIn("X_L", result.explanation)
        self.assertIn("X_C", result.explanation)
        self.assertTrue(result.metadata["verified"])

    def test_planner_validator_rejects_undefined_input(self) -> None:
        validation = validate_calculation_plan(
            {
                "target": {"symbol": "V", "description": "voltage", "unit": "V"},
                "givens": {"I": {"value": 2, "unit": "A", "si_value": 2, "si_unit": "A"}},
                "steps": [
                    {
                        "id": "s1",
                        "goal": "Compute voltage.",
                        "output": "V",
                        "formula_id": "ohm_voltage",
                        "formula": "V = I * R",
                        "inputs": ["I", "R"],
                        "unit": "V",
                        "premise": "Ohm's law.",
                    }
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            }
        )

        self.assertFalse(validation.ok)
        self.assertTrue(any("undefined" in error for error in validation.errors))

    def test_planner_validator_rejects_undeclared_constant(self) -> None:
        validation = validate_calculation_plan(
            {
                "target": {"symbol": "F", "description": "force", "unit": "N"},
                "givens": {
                    "q1": {"value": 1e-6, "unit": "C", "si_value": 1e-6, "si_unit": "C"},
                    "q2": {"value": 2e-6, "unit": "C", "si_value": 2e-6, "si_unit": "C"},
                    "r": {"value": 0.5, "unit": "m", "si_value": 0.5, "si_unit": "m"},
                },
                "steps": [
                    {
                        "id": "s1",
                        "goal": "Compute Coulomb force.",
                        "output": "F",
                        "formula_id": "coulomb_force",
                        "formula": "F = k*q1*q2/r**2",
                        "inputs": ["q1", "q2", "r"],
                        "unit": "N",
                        "premise": "Coulomb law.",
                    }
                ],
                "final_step": "s1",
                "missing_information": [],
                "status": "READY",
            }
        )

        self.assertFalse(validation.ok)
        self.assertTrue(any("undeclared names: k" in error for error in validation.errors))

    def test_execution_validator_rejects_unit_mismatch(self) -> None:
        plan = {
            "target": {"symbol": "V", "description": "voltage", "unit": "V"},
            "givens": {"I": {"value": 2, "unit": "A", "si_value": 2, "si_unit": "A"}},
            "steps": [
                {
                    "id": "s1",
                    "goal": "Compute voltage.",
                    "output": "V",
                    "formula_id": "ohm_voltage",
                    "formula": "V = I * R",
                    "inputs": ["I", "R"],
                    "unit": "V",
                    "premise": "Ohm's law.",
                }
            ],
            "final_step": "s1",
            "missing_information": [],
            "status": "READY",
        }
        execution = {
            "status": "SUCCESS",
            "trace": [
                {
                    "step_id": "s1",
                    "formula": "V = I * R",
                    "substitution": "V = 2 * 10",
                    "output": "V",
                    "value": 20,
                    "unit": "A",
                }
            ],
            "final": {"symbol": "V", "value": 20, "unit": "A"},
        }

        validation = validate_structured_execution(plan, execution)

        self.assertFalse(validation.ok)
        self.assertTrue(any("unit" in error for error in validation.errors))

def rlc_plan() -> dict[str, Any]:
    return {
        "target": {"symbol": "Z", "description": "series RLC impedance", "unit": "Ω"},
        "givens": {
            "R": {"value": 12, "unit": "Ω", "si_value": 12, "si_unit": "Ω"},
            "L": {"value": 0.2, "unit": "H", "si_value": 0.2, "si_unit": "H"},
            "C": {"value": 50, "unit": "μF", "si_value": 50e-6, "si_unit": "F"},
            "f": {"value": 50, "unit": "Hz", "si_value": 50, "si_unit": "Hz"},
        },
        "steps": [
            {"id": "s1", "goal": "Compute inductive reactance.", "output": "X_L", "formula_id": "inductive_reactance", "formula": "X_L = 2*pi*f*L", "inputs": ["f", "L"], "unit": "Ω", "premise": "X_L = 2πfL"},
            {"id": "s2", "goal": "Compute capacitive reactance.", "output": "X_C", "formula_id": "capacitive_reactance", "formula": "X_C = 1/(2*pi*f*C)", "inputs": ["f", "C"], "unit": "Ω", "premise": "X_C = 1/(2πfC)"},
            {"id": "s3", "goal": "Compute impedance.", "output": "Z", "formula_id": "series_rlc_impedance", "formula": "Z = sqrt(R**2 + (X_L - X_C)**2)", "inputs": ["R", "X_L", "X_C"], "unit": "Ω", "premise": "Z = sqrt(R^2 + (X_L - X_C)^2)"},
        ],
        "final_step": "s3",
        "missing_information": [],
        "status": "READY",
    }


if __name__ == "__main__":
    unittest.main()
