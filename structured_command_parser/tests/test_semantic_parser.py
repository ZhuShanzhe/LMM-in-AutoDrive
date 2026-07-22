from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from structured_command_parser.src.schema_tools import validate_document
from structured_command_parser.src.semantic_parser import SemanticIntentParser, _label_key


class SemanticIntentParserTests(unittest.TestCase):
    def make_parser(self) -> SemanticIntentParser:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as file:
            file.write(
                '{"text":"向左变道","expected":{"status":"VALID",'
                '"actions":["CHANGE_LANE"],"directions":["LEFT"]}}\n'
            )
            path = Path(file.name)
        self.addCleanup(path.unlink)
        return SemanticIntentParser("fake-model", prototypes_path=path)

    def test_semantic_document_extracts_live_speed_and_direction(self) -> None:
        parser = self.make_parser()
        result = parser._make_intent(
            "向右变道后减速到36km/h",
            "向右变道后减速到36 km/h",
            "VOICE",
            "semantic-1",
            {
                "status": "VALID",
                "actions": ["CHANGE_LANE", "SET_SPEED"],
                "directions": ["RIGHT"],
            },
            0.88,
            2.0,
        )
        validate_document(result)
        self.assertEqual(
            [step["action"] for step in result["intent"]["steps"]],
            ["CHANGE_LANE", "SET_SPEED"],
        )
        self.assertEqual(
            result["intent"]["steps"][1]["parameters"]["target_speed_mps"],
            10.0,
        )

    def test_missing_direction_fails_safe(self) -> None:
        parser = self.make_parser()
        result = parser._make_intent(
            "换到另一条车道",
            "换到另一条车道",
            "VOICE",
            "semantic-2",
            {"status": "VALID", "actions": ["CHANGE_LANE"]},
            0.8,
            2.0,
        )
        validate_document(result)
        self.assertEqual(result["parse_result"]["status"], "NEEDS_CLARIFICATION")

    def test_explicit_direction_overrides_retrieved_prototype(self) -> None:
        parser = self.make_parser()
        result = parser._make_intent(
            "请往右边并线",
            "请往右边并线",
            "VOICE",
            "semantic-3",
            {
                "status": "VALID",
                "actions": ["CHANGE_LANE"],
                "directions": ["LEFT"],
            },
            0.85,
            2.0,
        )
        self.assertEqual(
            result["intent"]["steps"][0]["parameters"]["direction"],
            "RIGHT",
        )

    def test_retrieval_label_does_not_split_left_and_right(self) -> None:
        left = {"status": "VALID", "actions": ["TURN"], "directions": ["LEFT"]}
        right = {"status": "VALID", "actions": ["TURN"], "directions": ["RIGHT"]}
        self.assertEqual(_label_key(left), _label_key(right))


if __name__ == "__main__":
    unittest.main()
