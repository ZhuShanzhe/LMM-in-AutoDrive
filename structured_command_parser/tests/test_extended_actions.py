from __future__ import annotations

import unittest

from structured_command_parser.src.english_parser import QwenEnglishIntentParser
from structured_command_parser.src.factory import make_document
from structured_command_parser.src.llm_parser import QwenIntentParser
from structured_command_parser.src.schema_tools import (
    load_schema,
    semantic_errors,
    validate_document,
)


class ExtendedActionContractTest(unittest.TestCase):
    def test_schema_exposes_complete_action_inventory(self) -> None:
        actions = set(
            load_schema()["$defs"]["step"]["properties"]["action"]["enum"]
        )
        self.assertEqual(
            actions,
            {
                "KEEP_LANE",
                "SET_SPEED",
                "ADJUST_SPEED",
                "STOP",
                "WAIT",
                "FOLLOW",
                "APPROACH",
                "NAVIGATE_TO",
                "CHANGE_LANE",
                "MERGE",
                "TURN",
                "U_TURN",
                "PROCEED",
                "YIELD",
                "PULL_OVER",
                "PARK",
                "OVERTAKE",
                "PASS_BY",
                "AVOID",
                "REVERSE",
                "ENTER_AREA",
                "EXIT_AREA",
                "EMERGENCY_BRAKE",
                "RESUME",
                "CANCEL",
            },
        )

    def test_follow_wait_merge_and_park_expand_to_valid_steps(self) -> None:
        commands = [
            {
                "action": "FOLLOW",
                "target_type": "VEHICLE",
                "target_relation": "AHEAD",
                "duration_s": 300,
            },
            {
                "action": "WAIT",
                "target_type": "PEDESTRIAN",
                "target_relation": "AHEAD_CROSSING",
                "condition": "pedestrian clears the crossing",
            },
            {
                "action": "MERGE",
                "lane_index": 2,
                "lane_reference": "RIGHT_EDGE",
            },
            {
                "action": "PARK",
                "purpose": "PICK_UP",
                "target_type": "PARKING_SPACE",
                "target_relation": "RIGHT",
                "parking_maneuver": "REVERSE",
            },
        ]
        steps = QwenIntentParser._expand_commands(commands)
        self.assertEqual(
            [step["action"] for step in steps],
            ["FOLLOW", "WAIT", "MERGE", "PARK"],
        )
        self.assertEqual(steps[0]["completion"]["type"], "DURATION_ELAPSED")
        self.assertEqual(steps[1]["trigger"]["type"], "CONDITION")
        self.assertIn("TARGET_LANE_SAFE", steps[2]["preconditions"])
        self.assertEqual(steps[3]["completion"]["type"], "PARKING_COMPLETED")

        document = make_document(
            raw_text="follow, wait, merge, and park",
            normalized_text="follow, wait, merge, and park",
            modality="TEXT",
            language="en-US",
            category="NAVIGATION",
            urgency="NORMAL",
            steps=steps,
            status="VALID",
            method="HYBRID",
            model="test",
            confidence=1.0,
            latency_ms=1.0,
        )
        self.assertEqual(document["schema_version"], "1.2.0")
        validate_document(document)

    def test_target_required_extended_actions_are_checked(self) -> None:
        for action in (
            "FOLLOW",
            "APPROACH",
            "NAVIGATE_TO",
            "PASS_BY",
            "ENTER_AREA",
            "EXIT_AREA",
        ):
            steps = QwenIntentParser._expand_commands([{"action": action}])
            document = {
                "intent": {"urgency": "NORMAL", "steps": steps},
                "parse_result": {"status": "VALID"},
            }
            self.assertTrue(
                any(f"{action} requires a target" in error for error in semantic_errors(document)),
                action,
            )

    def test_wait_requires_duration_or_condition(self) -> None:
        steps = QwenIntentParser._expand_commands([{"action": "WAIT"}])
        document = {
            "intent": {"urgency": "NORMAL", "steps": steps},
            "parse_result": {"status": "VALID"},
        }
        self.assertIn("step_1 WAIT requires duration_s or CONDITION trigger", semantic_errors(document))

    def test_explicit_coordinate_target_is_structured_and_validated(self) -> None:
        steps = QwenIntentParser._expand_commands(
            [
                {
                    "action": "NAVIGATE_TO",
                    "target_type": "COORDINATE",
                    "target_relation": "UNSPECIFIED",
                    "target_coordinates": {"x_m": 10.0, "y_m": 3.5, "frame": "MAP"},
                }
            ]
        )
        document = make_document(
            raw_text="navigate to map coordinate 10, 3.5",
            normalized_text="navigate to map coordinate 10, 3.5",
            modality="TEXT",
            language="en-US",
            category="NAVIGATION",
            urgency="NORMAL",
            steps=steps,
            status="VALID",
            method="RULE",
            model=None,
            confidence=1.0,
            latency_ms=1.0,
        )
        self.assertEqual(
            document["intent"]["steps"][0]["target"]["coordinates"]["frame"],
            "MAP",
        )

        missing_coordinates = QwenIntentParser._expand_commands(
            [
                {
                    "action": "NAVIGATE_TO",
                    "target_type": "COORDINATE",
                    "target_relation": "UNSPECIFIED",
                }
            ]
        )
        errors = semantic_errors(
            {
                "intent": {"urgency": "NORMAL", "steps": missing_coordinates},
                "parse_result": {"status": "VALID"},
            }
        )
        self.assertIn("step_1 COORDINATE target requires coordinates", errors)

    def test_english_normalizer_recovers_high_coverage_actions(self) -> None:
        payload = {
            "category": "NAVIGATION",
            "urgency": "NORMAL",
            "commands": [],
            "driving_style": "NORMAL",
        }
        normalized = QwenEnglishIntentParser._normalize_payload(
            payload,
            "Follow the blue car for five minutes, then turn around and back into a parking space.",
        )
        actions = [command["action"] for command in normalized["commands"]]
        self.assertEqual(actions, ["FOLLOW", "U_TURN", "PARK"])
        self.assertEqual(normalized["commands"][0]["duration_s"], 300)
        self.assertEqual(normalized["commands"][-1]["parking_maneuver"], "REVERSE")


if __name__ == "__main__":
    unittest.main()
