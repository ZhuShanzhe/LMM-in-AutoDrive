from __future__ import annotations

import unittest

from structured_command_parser.src.rule_parser import RuleIntentParser
from structured_command_parser.src.schema_tools import validate_document


class RuleParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RuleIntentParser()

    def parse(self, text: str) -> dict:
        result = self.parser.parse(text, request_id="test-command")
        self.assertIsNotNone(result)
        validate_document(result)
        return result

    def test_keep_lane_and_speed(self) -> None:
        result = self.parse("保持当前车道，提速至60km/h")
        self.assertEqual(
            [step["action"] for step in result["intent"]["steps"]],
            ["KEEP_LANE", "SET_SPEED"],
        )
        self.assertAlmostEqual(
            result["intent"]["steps"][1]["parameters"]["target_speed_mps"],
            16.667,
            places=3,
        )

    def test_turn_at_distance(self) -> None:
        result = self.parse("前方三百米路口右转")
        step = result["intent"]["steps"][0]
        self.assertEqual(step["action"], "TURN")
        self.assertEqual(step["parameters"]["direction"], "RIGHT")
        self.assertEqual(step["trigger"]["distance_m"], 300.0)

    def test_lane_change(self) -> None:
        result = self.parse("向左变道")
        step = result["intent"]["steps"][0]
        self.assertEqual(step["action"], "CHANGE_LANE")
        self.assertIn("LEFT_LANE_SAFE", step["preconditions"])

    def test_emergency_brake(self) -> None:
        result = self.parse("立即紧急刹车")
        self.assertEqual(result["intent"]["urgency"], "EMERGENCY")
        self.assertEqual(result["intent"]["steps"][0]["action"], "EMERGENCY_BRAKE")

    def test_complex_command_is_deferred_to_llm(self) -> None:
        result = self.parser.parse("看到前方行人后减速避让，然后向左变道")
        self.assertIsNone(result)

    def test_ambiguous_lane_change_requests_clarification(self) -> None:
        result = self.parse("换个道")
        self.assertEqual(result["parse_result"]["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(result["intent"]["steps"], [])

    def test_illegal_speed_request_is_unsupported(self) -> None:
        result = self.parse("不管限速开到最快")
        self.assertEqual(result["parse_result"]["status"], "UNSUPPORTED")

    def test_collision_request_is_unsupported(self) -> None:
        result = self.parse("直接撞向前方车辆")
        self.assertEqual(result["parse_result"]["status"], "UNSUPPORTED")
        self.assertEqual(result["intent"]["steps"], [])

    def test_brake_phrase_is_a_speed_decrease(self) -> None:
        result = self.parse("轻踩刹车")
        self.assertEqual(result["intent"]["steps"][0]["action"], "ADJUST_SPEED")

    def test_reversed_unit_target_speed(self) -> None:
        result = self.parse("以每秒23.9米的速度行驶")
        step = result["intent"]["steps"][0]
        self.assertEqual(step["action"], "SET_SPEED")
        self.assertEqual(step["parameters"]["target_speed_mps"], 23.9)

    def test_parking_lot_does_not_mean_stop(self) -> None:
        result = self.parse("在停车场向左变道")
        self.assertEqual(
            [step["action"] for step in result["intent"]["steps"]],
            ["CHANGE_LANE"],
        )


if __name__ == "__main__":
    unittest.main()
