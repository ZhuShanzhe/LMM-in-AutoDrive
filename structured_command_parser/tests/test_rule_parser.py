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

    def test_colloquial_kilometer_speed_keeps_numeric_target(self) -> None:
        cases = (
            "保持40公里速度行驶",
            "以40公里的速度行驶",
            "车速保持在40公里",
            "时速保持在四十公里",
        )
        for text in cases:
            with self.subTest(text=text):
                result = self.parse(text)
                step = result["intent"]["steps"][0]
                self.assertEqual(step["action"], "SET_SPEED")
                self.assertAlmostEqual(
                    step["parameters"]["target_speed_mps"],
                    11.111,
                    places=3,
                )
                self.assertEqual(step["parameters"]["source_value"], 40.0)
                self.assertEqual(step["parameters"]["source_unit"], "km/h")

    def test_distance_phrases_are_not_inferred_as_target_speed(self) -> None:
        for text in ("行驶40公里", "前方40公里后右转", "保持40米距离"):
            with self.subTest(text=text):
                result = self.parser.parse(text, request_id="distance-command")
                if result is None:
                    continue
                self.assertNotIn(
                    "SET_SPEED",
                    [
                        step["action"]
                        for step in result["intent"]["steps"]
                    ],
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

    def test_urgent_stop_without_danger_is_not_emergency_brake(self) -> None:
        for text in ("马上刹停", "立即停止车辆", "请尽快停车"):
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(result["intent"]["urgency"], "URGENT")
                self.assertEqual(result["intent"]["steps"][0]["action"], "STOP")

    def test_danger_context_can_make_urgent_stop_emergency(self) -> None:
        result = self.parse("前方突然冲出行人，立即停车")
        self.assertEqual(result["intent"]["urgency"], "EMERGENCY")
        self.assertEqual(result["intent"]["steps"][0]["action"], "EMERGENCY_BRAKE")

    def test_ordinary_stop_and_brake_remain_normal(self) -> None:
        stop = self.parse("把车停下来")
        self.assertEqual(stop["intent"]["urgency"], "NORMAL")
        self.assertEqual(stop["intent"]["steps"][0]["action"], "STOP")

        brake = self.parse("轻踩刹车")
        self.assertEqual(brake["intent"]["urgency"], "NORMAL")
        self.assertEqual(brake["intent"]["steps"][0]["action"], "ADJUST_SPEED")

    def test_complex_command_uses_fast_path(self) -> None:
        result = self.parse("看到前方行人后减速避让，然后向左变道")
        self.assertEqual(
            [step["action"] for step in result["intent"]["steps"]],
            ["ADJUST_SPEED", "CHANGE_LANE"],
        )
        self.assertEqual(result["parse_result"]["method"], "RULE")

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

    def test_diverse_chinese_aliases_use_fast_path(self) -> None:
        cases = {
            "把速度提上去": ["ADJUST_SPEED"],
            "往左侧车道并线": ["CHANGE_LANE"],
            "到前面的路口向右拐": ["TURN"],
            "礼让正在过街的行人": ["YIELD"],
            "躲开道路中央的锥桶": ["AVOID"],
            "马上刹停": ["STOP"],
            "降低车速后并入右侧车道": ["ADJUST_SPEED", "CHANGE_LANE"],
        }
        for text, expected_actions in cases.items():
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(
                    [step["action"] for step in result["intent"]["steps"]],
                    expected_actions,
                )
                self.assertLess(result["parse_result"]["latency_ms"], 50.0)

    def test_traffic_rule_violations_are_rejected_fast(self) -> None:
        for text in ("无视红灯继续开", "逆行绕过堵车路段", "从人行道上超过前车"):
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertEqual(result["parse_result"]["status"], "UNSUPPORTED")
                self.assertLess(result["parse_result"]["latency_ms"], 50.0)


if __name__ == "__main__":
    unittest.main()
