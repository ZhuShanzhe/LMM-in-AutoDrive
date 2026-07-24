import copy
import json
import unittest
from pathlib import Path

from scene_understanding.src.driving_intent_alignment import (
    RELATION_HINTS,
    TARGET_TYPE_MAP,
    WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET_TYPES,
    align_driving_intent,
    target_to_reference,
    validate_alignment_result,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "scene_understanding" / "schemas" / "examples" / "world_state.example.json"
DRIVING_INTENT_EXAMPLE = (
    ROOT
    / "structured_command_parser"
    / "examples"
    / "complex_obstacle_avoidance.json"
)
DRIVING_INTENT_SCHEMA = (
    ROOT
    / "structured_command_parser"
    / "schemas"
    / "driving_intent.schema.json"
)
DRIVING_INTENT_ALIGNMENT_SCHEMA = (
    ROOT
    / "scene_understanding"
    / "schemas"
    / "driving_intent_alignment.schema.json"
)
DRIVING_INTENT_ALIGNMENT_EXAMPLE = (
    ROOT
    / "scene_understanding"
    / "schemas"
    / "examples"
    / "driving_intent_alignment.example.json"
)


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

    def test_accepts_current_driving_intent_schema_version(self):
        driving_intent = json.loads(
            DRIVING_INTENT_EXAMPLE.read_text(encoding="utf-8")
        )

        result = align_driving_intent(driving_intent, self.world_state)

        self.assertEqual(result["request_id"], "complex-0001")
        self.assertEqual(result["parse_status"], "VALID")
        self.assertEqual(validate_alignment_result(result), [])

    def test_checked_in_batch_alignment_contract_is_synchronized(self):
        parser_schema = json.loads(
            DRIVING_INTENT_SCHEMA.read_text(encoding="utf-8")
        )
        alignment_schema = json.loads(
            DRIVING_INTENT_ALIGNMENT_SCHEMA.read_text(
                encoding="utf-8"
            )
        )
        example = json.loads(
            DRIVING_INTENT_ALIGNMENT_EXAMPLE.read_text(
                encoding="utf-8"
            )
        )
        driving_intent = json.loads(
            DRIVING_INTENT_EXAMPLE.read_text(encoding="utf-8")
        )
        expected = align_driving_intent(
            driving_intent,
            self.world_state,
        )

        self.assertEqual(example, expected)
        self.assertEqual(validate_alignment_result(example), [])
        self.assertEqual(
            set(alignment_schema["required"]),
            set(example),
        )
        self.assertEqual(
            alignment_schema["properties"]["schema_version"][
                "const"
            ],
            example["schema_version"],
        )
        self.assertEqual(
            alignment_schema["$defs"]["target"],
            parser_schema["$defs"]["target"],
        )
        self.assertEqual(
            alignment_schema["$defs"]["coordinates"],
            parser_schema["$defs"]["coordinates"],
        )

        required_step_fields = set(
            alignment_schema["$defs"]["stepAlignment"][
                "required"
            ]
        )
        for alignment in example["step_alignments"]:
            self.assertEqual(
                required_step_fields,
                set(alignment),
            )

    def test_covers_all_current_relation_values(self):
        schema = json.loads(
            DRIVING_INTENT_SCHEMA.read_text(encoding="utf-8")
        )
        relation_values = set(
            schema["$defs"]["target"]["properties"]["relation"]["enum"]
        )

        self.assertEqual(
            relation_values - set(RELATION_HINTS),
            set(),
        )

    def test_classifies_all_current_target_types(self):
        schema = json.loads(
            DRIVING_INTENT_SCHEMA.read_text(encoding="utf-8")
        )
        schema_types = set(
            schema["$defs"]["target"]["properties"]["type"]["enum"]
        )
        classified_types = (
            set(TARGET_TYPE_MAP)
            | set(
                WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET_TYPES
            )
            | {"LANE", "UNKNOWN"}
        )

        self.assertEqual(
            schema_types,
            classified_types,
        )

    def test_maps_extended_spatial_relations(self):
        expected_positions = {
            "FRONT_LEFT": "front_left",
            "FRONT_RIGHT": "front_right",
            "REAR_LEFT": "rear_left",
            "REAR_RIGHT": "rear_right",
            "IN_FRONT_OF": "front",
            "PAST": "rear",
            "NEXT_TO": "unknown",
            "NEAR": "unknown",
            "INSIDE": "unknown",
        }

        for relation, expected_position in expected_positions.items():
            with self.subTest(relation=relation):
                reference = target_to_reference(
                    {
                        "type": "VEHICLE",
                        "relation": relation,
                    },
                    action="FOLLOW",
                )
                self.assertEqual(
                    reference["position_hint"],
                    expected_position,
                )

    def test_maps_extended_actor_target_types(self):
        expected_types = {
            "CYCLIST": "cyclist",
            "TRAFFIC_CONE": "traffic_cone",
            "OBSTACLE": "obstacle",
            "ROAD_HAZARD": "road_hazard",
        }

        for target_type, expected_type in expected_types.items():
            with self.subTest(target_type=target_type):
                reference = target_to_reference(
                    {
                        "type": target_type,
                        "relation": "AHEAD",
                    },
                    action="AVOID",
                )
                self.assertEqual(
                    reference["target_type"],
                    expected_type,
                )
                self.assertEqual(
                    reference["position_hint"],
                    "front",
                )

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


    def test_reports_world_state_capability_unavailable_for_map_targets(self):
        target_types = {
            "AREA",
            "CONSTRUCTION_ZONE",
            "COORDINATE",
            "CROSSWALK",
            "CURB",
            "DESTINATION",
            "DROPOFF_POINT",
            "LANDMARK",
            "PARKING_AREA",
            "PARKING_SPACE",
            "PICKUP_POINT",
            "ROAD",
            "STOP_LINE",
        }

        for target_type in sorted(target_types):
            with self.subTest(target_type=target_type):
                intent = complex_intent()
                intent["intent"]["steps"] = [
                    intent["intent"]["steps"][0]
                ]
                intent["intent"]["steps"][0]["target"] = {
                    "type": target_type,
                    "relation": "AHEAD",
                }

                result = align_driving_intent(
                    intent,
                    self.world_state,
                )
                alignment = result["step_alignments"][0]

                self.assertFalse(
                    alignment["alignment_success"]
                )
                self.assertIsNone(
                    alignment["matched_entity"]
                )
                self.assertEqual(
                    alignment["reason_code"],
                    "world_state_capability_unavailable",
                )

    def test_keeps_unknown_target_type_explicitly_unsupported(self):
        intent = complex_intent()
        intent["intent"]["steps"] = [
            intent["intent"]["steps"][0]
        ]
        intent["intent"]["steps"][0]["target"] = {
            "type": "UNKNOWN",
            "relation": "UNSPECIFIED",
        }

        result = align_driving_intent(
            intent,
            self.world_state,
        )
        alignment = result["step_alignments"][0]

        self.assertFalse(alignment["alignment_success"])
        self.assertIsNone(alignment["matched_entity"])
        self.assertEqual(
            alignment["reason_code"],
            "unsupported_target_type",
        )


if __name__ == "__main__":
    unittest.main()
