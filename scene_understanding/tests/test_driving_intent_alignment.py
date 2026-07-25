import copy
import json
import unittest
from pathlib import Path

from scene_understanding.src.driving_intent_alignment import (
    align_driving_intent,
    target_to_reference,
    validate_alignment_result,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "scene_understanding" / "schemas" / "examples" / "world_state.example.json"
PARSER_BASIC_EXAMPLE = ROOT / "structured_command_parser" / "examples" / "basic_set_speed.json"


def complex_intent() -> dict:
    return {
        "schema_version": "1.0.0",
        "request_id": "complex-0001",
        "input": {
            "modality": "VOICE",
            "language": "zh-CN",
            "raw_text": "看到前方横穿马路的行人，减速避让后向左变道超越慢车",
            "normalized_text": "看到前方横穿马路的行人，减速避让后向左变道超越慢车",
        },
        "intent": {
            "category": "COMPLEX_OBSTACLE_AVOIDANCE",
            "urgency": "NORMAL",
            "steps": [
                {
                    "step_id": "step_1",
                    "action": "ADJUST_SPEED",
                    "target": {
                        "type": "PEDESTRIAN",
                        "relation": "AHEAD_CROSSING",
                    },
                    "parameters": {"change": "DECREASE"},
                },
                {
                    "step_id": "step_2",
                    "action": "CHANGE_LANE",
                    "target": {"type": "SLOW_VEHICLE", "relation": "AHEAD"},
                    "parameters": {"direction": "LEFT", "lane_count": 1},
                },
                {
                    "step_id": "step_3",
                    "action": "OVERTAKE",
                    "target": {"type": "SLOW_VEHICLE", "relation": "AHEAD"},
                    "parameters": {},
                },
            ],
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "CONSERVATIVE",
            },
        },
        "parse_result": {
            "status": "VALID",
            "method": "LLM",
            "model": "Qwen2.5-1.5B-Instruct",
            "confidence": 0.96,
            "missing_slots": [],
            "warnings": [],
            "latency_ms": 0.0,
        },
    }


class DrivingIntentAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(WORLD_STATE_EXAMPLE.read_text(encoding="utf-8"))

    def test_maps_structured_targets_without_reparsing_chinese(self):
        pedestrian = target_to_reference(
            {"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
            action="ADJUST_SPEED",
        )
        self.assertEqual(pedestrian["target_type"], "pedestrian")
        self.assertEqual(pedestrian["position_hint"], "front")
        self.assertEqual(pedestrian["lane_hint"], "crossing_ego_path")

        slow_vehicle = target_to_reference(
            {"type": "SLOW_VEHICLE", "relation": "AHEAD"},
            action="OVERTAKE",
        )
        self.assertEqual(slow_vehicle["target_type"], "slow_vehicle")
        self.assertEqual(slow_vehicle["lane_hint"], "ego_lane")

    def test_reports_partial_alignment_without_inventing_pedestrian(self):
        result = align_driving_intent(complex_intent(), self.world_state)
        self.assertEqual(validate_alignment_result(result), [])
        self.assertEqual(result["alignment_status"], "PARTIAL")
        self.assertEqual(result["target_count"], 3)
        self.assertEqual(result["matched_target_count"], 2)

        pedestrian = result["step_alignments"][0]
        self.assertFalse(pedestrian["alignment_success"])
        self.assertIsNone(pedestrian["matched_entity"])

        for alignment in result["step_alignments"][1:]:
            self.assertTrue(alignment["alignment_success"])
            self.assertEqual(
                alignment["matched_entity"]["entity_id"], "carla_actor_42"
            )

    def test_skips_non_valid_parser_result(self):
        intent = complex_intent()
        intent["parse_result"]["status"] = "NEEDS_CLARIFICATION"
        intent["intent"]["steps"] = []
        result = align_driving_intent(intent, self.world_state)
        self.assertEqual(result["alignment_status"], "SKIPPED")
        self.assertEqual(result["step_alignments"], [])

    def test_rejects_duplicate_step_ids(self):
        intent = complex_intent()
        intent["intent"]["steps"][1]["step_id"] = "step_1"
        with self.assertRaisesRegex(ValueError, "duplicate step_id"):
            align_driving_intent(intent, self.world_state)

    def test_maps_lane_direction_from_parameters(self):
        reference = target_to_reference(
            {"type": "LANE", "relation": "UNSPECIFIED"},
            action="CHANGE_LANE",
            parameters={"direction": "LEFT"},
        )
        self.assertEqual(reference["target_type"], "left_lane")
        self.assertEqual(reference["lane_hint"], "left_adjacent_lane")

    def test_accepts_current_parser_driving_intent_contract(self):
        intent = json.loads(PARSER_BASIC_EXAMPLE.read_text(encoding="utf-8"))
        result = align_driving_intent(intent, self.world_state)

        self.assertEqual(result["request_id"], intent["request_id"])
        self.assertEqual(result["alignment_status"], "NOT_REQUIRED")
        self.assertEqual(validate_alignment_result(result), [])


if __name__ == "__main__":
    unittest.main()
