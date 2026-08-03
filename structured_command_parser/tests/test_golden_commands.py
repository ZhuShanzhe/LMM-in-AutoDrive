from __future__ import annotations

import json
import unittest
from pathlib import Path

from structured_command_parser import HybridCommandParser
from structured_command_parser.scripts.evaluate_parser import (
    matches_expected,
    summarize_result,
)
from structured_command_parser.src.schema_tools import validate_document


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_commands.jsonl"


class GoldenCommandTest(unittest.TestCase):
    def test_rule_samples(self) -> None:
        parser = HybridCommandParser()
        samples = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(samples), 30)
        rule_samples = [sample for sample in samples if sample["parser_path"] == "RULE"]
        self.assertEqual(len(rule_samples), 15)
        for sample in rule_samples:
            with self.subTest(sample=sample["sample_id"]):
                result = parser.parse(sample["text"], request_id=sample["sample_id"])
                validate_document(result)
                self.assertTrue(
                    matches_expected(summarize_result(result), sample["expected"])
                )

    def test_all_competition_samples_use_fast_path_within_budget(self) -> None:
        parser = HybridCommandParser()
        samples = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for sample in samples:
            with self.subTest(sample=sample["sample_id"]):
                result = parser.parse(sample["text"], request_id=sample["sample_id"])
                validate_document(result)
                self.assertTrue(
                    matches_expected(summarize_result(result), sample["expected"])
                )
                self.assertLess(result["parse_result"]["latency_ms"], 50.0)


if __name__ == "__main__":
    unittest.main()
