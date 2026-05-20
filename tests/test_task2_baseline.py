from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exact2026.type2.json_utils import extract_json_object  # noqa: E402
from task2_baseline import parse_model_response, select_rows  # noqa: E402


class Task2BaselineSelectionTests(unittest.TestCase):
    def test_select_rows_limit_is_contiguous_by_default(self) -> None:
        rows = [{"id": str(index)} for index in range(5)]

        selected = select_rows(rows, start=1, limit=2)

        self.assertEqual([index for index, _ in selected], [1, 2])

    def test_select_rows_random_sample_is_seeded(self) -> None:
        rows = [{"id": str(index)} for index in range(20)]

        first = select_rows(rows, limit=10, random_sample=True, seed=123)
        second = select_rows(rows, limit=10, random_sample=True, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertNotEqual([index for index, _ in first], list(range(10)))

    def test_parse_model_response_handles_plain_key_value_text(self) -> None:
        raw = (
            "answer:125\n"
            "unit:μJ\n"
            "explanation:Solved using the capacitor discharge energy formula."
        )

        parsed = parse_model_response(raw)

        self.assertEqual(parsed["answer"], "125")
        self.assertEqual(parsed["unit"], "μJ")
        self.assertIn("capacitor discharge energy", parsed["explanation"])

    def test_extract_json_object_repairs_bare_latex_backslashes(self) -> None:
        raw = (
            '{"answer":"125","unit":"μJ",'
            '"explanation":"Use $F_{net}=\\sqrt{3}F$ and $\\mu$ carefully."}'
        )

        parsed = extract_json_object(raw)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["unit"], "μJ")
        self.assertIn("\\sqrt{3}", parsed["explanation"])


if __name__ == "__main__":
    unittest.main()
