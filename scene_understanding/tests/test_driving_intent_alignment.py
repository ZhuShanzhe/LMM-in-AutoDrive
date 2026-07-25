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

    def _compositional_stop_intent(self) -> dict:
        return {
            "schema_version": "1.2.0",
            "request_id": "grounding-0001",
            "input": {
                "modality": "TEXT",
                "language": "en-US",
                "raw_text": "Stop before the red truck.",
                "normalized_text": "Stop before the red truck.",
            },
            "normalization": {
                "edits": [],
                "unresolved_references": [],
            },
            "intent": {
                "category": "BASIC_CONTROL",
                "urgency": "NORMAL",
                "entities": [
                    {
                        "entity_id": "target_1",
                        "type": "VEHICLE",
                        "relation": "AHEAD",
                        "description": "the red truck",
                        "canonical_attributes": {
                            "color": "RED",
                            "vehicle_subtype": "TRUCK",
                        },
                        "open_descriptors": [],
                        "source_span": "the red truck",
                    }
                ],
                "suppressed_intents": [],
                "steps": [
                    {
                        "step_id": "step_1",
                        "action": "STOP",
                        "target_ref": "target_1",
                        "parameters": {},
                        "trigger": {"type": "IMMEDIATE"},
                        "depends_on": [],
                        "preconditions": ["TARGET_VISIBLE"],
                        "on_blocked": "SAFE_STOP",
                        "goal_conditions": [
                            {
                                "predicate": "BEFORE",
                                "subject": "EGO",
                                "object": "target_1",
                                "source_span": "before",
                            }
                        ],
                        "completion": {"type": "STOPPED_BEFORE_TARGET"},
                    }
                ],
                "constraints": {
                    "safety_first": True,
                    "obey_traffic_rules": True,
                    "driving_style": "NORMAL",
                },
            },
            "parse_result": {
                "status": "VALID",
                "method": "HYBRID",
                "model": "test",
                "confidence": 0.99,
                "missing_slots": [],
                "warnings": [],
                "latency_ms": 1.0,
            },
        }

    def test_structured_attributes_select_the_red_truck(self):
        state = copy.deepcopy(self.world_state)
        base = state["objects"][0]
        red_truck = copy.deepcopy(base)
        red_truck["object_id"] = "red_truck"
        red_truck["subtype"] = "vehicle.test.truck"
        red_truck["semantic_matches"] = [
            {
                "camera_name": "front",
                "visual_object_id": "v-red",
                "bbox_2d": [0.1, 0.1, 0.3, 0.4],
                "description": "vehicle; truck; red; front",
                "confidence": 0.98,
            }
        ]
        white_car = copy.deepcopy(base)
        white_car["object_id"] = "white_car"
        white_car["distance_m"] = red_truck["distance_m"] - 2
        white_car["semantic_matches"] = [
            {
                "camera_name": "front",
                "visual_object_id": "v-white",
                "bbox_2d": [0.5, 0.1, 0.7, 0.4],
                "description": "vehicle; sedan; white; front",
                "confidence": 0.98,
            }
        ]
        state["objects"] = [white_car, red_truck]
        result = align_driving_intent(
            self._compositional_stop_intent(), state
        )
        alignment = result["step_alignments"][0]
        self.assertTrue(alignment["alignment_success"])
        self.assertEqual(alignment["matched_entity"]["entity_id"], "red_truck")
        self.assertEqual(
            alignment["resolved_goal_conditions"][0]["object"],
            "red_truck",
        )

    def test_duplicate_red_trucks_are_rejected_as_ambiguous(self):
        state = copy.deepcopy(self.world_state)
        first = copy.deepcopy(state["objects"][0])
        second = copy.deepcopy(state["objects"][0])
        for index, obj in enumerate((first, second), start=1):
            obj["object_id"] = f"red_truck_{index}"
            obj["subtype"] = "vehicle.test.truck"
            obj["semantic_matches"] = [
                {
                    "camera_name": "front",
                    "visual_object_id": f"v-{index}",
                    "bbox_2d": [0.1 * index, 0.1, 0.2 * index, 0.4],
                    "description": "vehicle; red; truck; front",
                    "confidence": 0.98,
                }
            ]
        state["objects"] = [first, second]
        result = align_driving_intent(
            self._compositional_stop_intent(), state
        )
        alignment = result["step_alignments"][0]
        self.assertFalse(alignment["alignment_success"])
        self.assertEqual(
            alignment["reason_code"], "ambiguous_matching_entities"
        )


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


    def test_aligns_adjacent_lane_targets_in_batch(self):
        cases = (
            (
                "LEFT",
                True,
                "COMPLETE",
                "matched_adjacent_lane",
                "lane:12:-2",
            ),
            (
                "RIGHT",
                False,
                "FAILED",
                "right_lane_unavailable",
                None,
            ),
        )

        for (
            direction,
            expected_success,
            expected_status,
            expected_reason,
            expected_entity_id,
        ) in cases:
            with self.subTest(direction=direction):
                intent = complex_intent()
                intent["intent"]["steps"] = [
                    {
                        "step_id": "lane_step",
                        "action": "CHANGE_LANE",
                        "target": {
                            "type": "LANE",
                            "relation": "UNSPECIFIED",
                        },
                        "parameters": {
                            "direction": direction,
                            "lane_count": 1,
                        },
                    }
                ]

                result = align_driving_intent(
                    intent,
                    self.world_state,
                )
                alignment = result["step_alignments"][0]

                self.assertEqual(
                    result["alignment_status"],
                    expected_status,
                )
                self.assertEqual(
                    alignment["alignment_success"],
                    expected_success,
                )
                self.assertEqual(
                    alignment["reason_code"],
                    expected_reason,
                )

                if expected_entity_id is None:
                    self.assertIsNone(
                        alignment["matched_entity"]
                    )
                else:
                    self.assertEqual(
                        alignment["matched_entity"][
                            "entity_type"
                        ],
                        "lane",
                    )
                    self.assertEqual(
                        alignment["matched_entity"][
                            "entity_id"
                        ],
                        expected_entity_id,
                    )

    def test_aligns_current_junction_target_in_batch(self):
        intent = complex_intent()
        intent["intent"]["steps"] = [
            {
                "step_id": "junction_step",
                "action": "TURN",
                "target": {
                    "type": "JUNCTION",
                    "relation": "AT_JUNCTION",
                },
                "parameters": {
                    "direction": "LEFT",
                },
            }
        ]

        unavailable = align_driving_intent(
            intent,
            self.world_state,
        )
        unavailable_alignment = unavailable[
            "step_alignments"
        ][0]

        self.assertEqual(
            unavailable["alignment_status"],
            "FAILED",
        )
        self.assertFalse(
            unavailable_alignment[
                "alignment_success"
            ]
        )
        self.assertIsNone(
            unavailable_alignment["matched_entity"]
        )
        self.assertEqual(
            unavailable_alignment["reason_code"],
            "junction_not_currently_available",
        )

        junction_world = copy.deepcopy(
            self.world_state
        )
        junction_world["ego"]["is_junction"] = True
        junction_world["environment"][
            "is_intersection"
        ] = True

        available = align_driving_intent(
            intent,
            junction_world,
        )
        available_alignment = available[
            "step_alignments"
        ][0]

        self.assertEqual(
            available["alignment_status"],
            "COMPLETE",
        )
        self.assertTrue(
            available_alignment["alignment_success"]
        )
        self.assertEqual(
            available_alignment["reason_code"],
            "matched_current_junction",
        )
        self.assertEqual(
            available_alignment["matched_entity"][
                "entity_type"
            ],
            "junction",
        )
        self.assertEqual(
            available_alignment["matched_entity"][
                "category"
            ],
            "junction",
        )


if __name__ == "__main__":
    unittest.main()
