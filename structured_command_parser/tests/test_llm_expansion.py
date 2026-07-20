from __future__ import annotations

import unittest

from structured_command_parser.src.llm_parser import QwenIntentParser


class LlmExpansionTest(unittest.TestCase):
    def test_compact_commands_expand_to_schema_steps(self) -> None:
        commands = [
            {
                "action": "ADJUST_SPEED",
                "change": "DECREASE",
                "target_type": "PEDESTRIAN",
                "target_relation": "AHEAD_CROSSING",
            },
            {
                "action": "CHANGE_LANE",
                "direction": "LEFT",
                "lane_count": 1,
            },
            {
                "action": "OVERTAKE",
                "target_type": "SLOW_VEHICLE",
                "target_relation": "AHEAD",
            },
        ]
        steps = QwenIntentParser._expand_commands(commands)
        self.assertEqual(
            [step["action"] for step in steps],
            ["ADJUST_SPEED", "CHANGE_LANE", "OVERTAKE"],
        )
        self.assertEqual(steps[1]["depends_on"], ["step_1"])
        self.assertEqual(steps[2]["trigger"]["step_id"], "step_2")
        self.assertIn("LEFT_LANE_SAFE", steps[1]["preconditions"])

    def test_overtake_purpose_is_expanded_when_model_compacts_it(self) -> None:
        commands = [
            {
                "action": "CHANGE_LANE",
                "direction": "LEFT",
                "purpose": "OVERTAKE",
                "target_type": "SLOW_VEHICLE",
                "target_relation": "AHEAD",
            }
        ]
        steps = QwenIntentParser._expand_commands(commands)
        self.assertEqual(
            [step["action"] for step in steps], ["CHANGE_LANE", "OVERTAKE"]
        )
        self.assertEqual(steps[1]["depends_on"], ["step_1"])

    def test_yield_speed_change_waits_for_target(self) -> None:
        commands = [
            {
                "action": "ADJUST_SPEED",
                "change": "DECREASE",
                "purpose": "YIELD",
                "target_type": "PEDESTRIAN",
                "target_relation": "AHEAD_CROSSING",
            }
        ]
        step = QwenIntentParser._expand_commands(commands)[0]
        self.assertEqual(step["trigger"]["type"], "OBJECT_PRESENT")
        self.assertEqual(step["completion"]["type"], "TARGET_CLEARED")

    def test_model_aliases_are_normalized(self) -> None:
        payload = {
            "commands": [
                {"action": "RETURN_TO_LANE", "target_type": "TRAFFIC_CONSTRUCTION"}
            ]
        }
        normalized = QwenIntentParser._normalize_payload(
            payload, "避让施工锥桶并回归原车道"
        )
        self.assertEqual(
            [command["action"] for command in normalized["commands"]],
            ["AVOID", "RESUME"],
        )
        self.assertEqual(
            normalized["commands"][1]["target_type"], "CONSTRUCTION_ZONE"
        )

    def test_explicit_overtake_is_not_dropped(self) -> None:
        payload = {
            "commands": [{"action": "ADJUST_SPEED", "change": "INCREASE"}]
        }
        normalized = QwenIntentParser._normalize_payload(
            payload, "绕开前方慢车后提速"
        )
        self.assertEqual(normalized["commands"][0]["action"], "OVERTAKE")
        self.assertNotIn(
            "AVOID", [command["action"] for command in normalized["commands"]]
        )

    def test_yielding_speed_change_does_not_gain_duplicate_avoid(self) -> None:
        payload = {
            "commands": [
                {
                    "action": "ADJUST_SPEED",
                    "change": "DECREASE",
                    "purpose": "YIELD",
                    "target_type": "PEDESTRIAN",
                },
                {
                    "action": "CHANGE_LANE",
                    "direction": "LEFT",
                    "purpose": "OVERTAKE",
                },
            ]
        }
        normalized = QwenIntentParser._normalize_payload(
            payload, "看到前方横穿马路的行人，减速避让后向左变道超越慢车"
        )
        self.assertEqual(
            [command["action"] for command in normalized["commands"]],
            ["ADJUST_SPEED", "CHANGE_LANE"],
        )

    def test_explicit_kmh_is_converted(self) -> None:
        payload = {
            "commands": [{"action": "SET_SPEED", "target_speed_mps": 30}]
        }
        normalized = QwenIntentParser._normalize_payload(payload, "减速至30 km/h")
        self.assertEqual(normalized["commands"][0]["target_speed_mps"], 8.333)

    def test_invented_speed_is_removed(self) -> None:
        payload = {
            "category": "BASIC_CONTROL",
            "commands": [{"action": "SET_SPEED", "target_speed_mps": 36}],
        }
        normalized = QwenIntentParser._normalize_payload(
            payload, "前方路况危险,保持安全车速"
        )
        self.assertEqual(normalized["commands"][0]["action"], "ADJUST_SPEED")
        self.assertNotIn("target_speed_mps", normalized["commands"][0])
        self.assertEqual(normalized["category"], "EMERGENCY_RESPONSE")

    def test_missing_payload_tail_gets_safe_defaults(self) -> None:
        payload = {
            "category": "COMPLEX_OBSTACLE_AVOIDANCE",
            "urgency": "NORMAL",
            "commands": [{"action": "CHANGE_LANE", "direction": "LEFT"}],
        }
        normalized = QwenIntentParser._normalize_payload(payload, "向左变道")
        self.assertEqual(normalized["status"], "VALID")
        self.assertEqual(normalized["missing_slots"], [])

    def test_construction_merge_is_emergency_response(self) -> None:
        payload = {
            "category": "COMPLEX_OBSTACLE_AVOIDANCE",
            "urgency": "NORMAL",
            "commands": [{"action": "CHANGE_LANE", "direction": "LEFT"}],
        }
        normalized = QwenIntentParser._normalize_payload(
            payload, "施工路段，减速并道至左侧车道"
        )
        self.assertEqual(normalized["category"], "EMERGENCY_RESPONSE")
        self.assertEqual(normalized["driving_style"], "CONSERVATIVE")
        self.assertEqual(
            [command["action"] for command in normalized["commands"]],
            ["ADJUST_SPEED", "CHANGE_LANE"],
        )


if __name__ == "__main__":
    unittest.main()
