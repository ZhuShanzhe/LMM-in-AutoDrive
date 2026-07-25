import copy
import json
import unittest
from pathlib import Path

from scene_understanding.core.risk_assessment import (
    assess_world_state,
    compute_ttc_s,
    safe_following_distance_m,
    ttc_risk_level,
    validate_risk_assessment,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "schemas" / "examples" / "world_state.example.json"
RISK_EXAMPLE = ROOT / "schemas" / "examples" / "risk_assessment.example.json"


class RiskAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(WORLD_STATE_EXAMPLE.read_text(encoding="utf-8"))

    def test_speed_based_safe_distance_thresholds(self):
        self.assertEqual(safe_following_distance_m(29.9 / 3.6), 10.0)
        self.assertEqual(safe_following_distance_m(30.0 / 3.6), 20.0)
        self.assertEqual(safe_following_distance_m(60.0 / 3.6), 20.0)
        self.assertEqual(safe_following_distance_m(60.1 / 3.6), 40.0)

    def test_ttc_thresholds_match_plan(self):
        self.assertEqual(ttc_risk_level(4.1), "none")
        self.assertEqual(ttc_risk_level(4.0), "low")
        self.assertEqual(ttc_risk_level(2.0), "low")
        self.assertEqual(ttc_risk_level(1.0), "medium")
        self.assertEqual(ttc_risk_level(0.9), "high")

    def test_receding_object_has_no_ttc(self):
        self.assertIsNone(compute_ttc_s(10.0, 0.0))
        self.assertIsNone(compute_ttc_s(10.0, -2.0))

    def test_example_produces_valid_medium_risk(self):
        result = assess_world_state(self.world_state)
        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(result["safe_following_distance_m"], 20.0)
        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["object_assessments"][0]["ttc_s"], 5.0)

    def test_checked_in_risk_example_is_valid(self):
        example = json.loads(RISK_EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(validate_risk_assessment(example), [])

    def test_imminent_object_is_high_risk(self):
        state = copy.deepcopy(self.world_state)
        obj = state["objects"][0]
        obj["distance_m"] = 5.0
        obj["relative_position_ego_m"]["longitudinal"] = 5.0
        obj["closing_speed_mps"] = 10.0
        result = assess_world_state(state)
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["object_assessments"][0]["ttc_s"], 0.5)
        self.assertEqual(result["recommended_action"], "decelerate")

    def test_collision_requires_emergency_brake(self):
        state = copy.deepcopy(self.world_state)
        state["sensor_events"]["collisions"] = [
            {
                "event_id": "collision_00000123_0001",
                "frame": 123,
                "timestamp_s": 6.15,
                "other_actor_id": "42",
                "normal_impulse_ns": {"x": 1.0, "y": 0.0, "z": 0.0},
                "impulse_magnitude_ns": 1.0,
            }
        ]
        result = assess_world_state(state)
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["recommended_action"], "emergency_brake")
        self.assertIn("collision_detected", result["reason_codes"])

    def test_lane_change_blocked_by_close_rear_vehicle(self):
        state = copy.deepcopy(self.world_state)
        rear = copy.deepcopy(state["objects"][0])
        rear["object_id"] = "carla_actor_43"
        rear["source_object_id"] = "43"
        rear["lane_relation"] = "left_adjacent_lane"
        rear["relative_position_ego_m"]["longitudinal"] = -8.0
        rear["distance_m"] = 8.0
        rear["closing_speed_mps"] = 3.0
        rear["semantic_matches"] = []
        state["objects"].append(rear)
        result = assess_world_state(state)
        left = result["lane_change"]["left"]
        self.assertFalse(left["is_safe"])
        self.assertEqual(left["closest_rear_gap_m"], 8.0)
        self.assertIn("carla_actor_43", left["blocking_object_ids"])

    def test_lane_change_permission_is_enforced(self):
        state = copy.deepcopy(self.world_state)
        state["ego"]["lane_change"] = "none"
        result = assess_world_state(state)
        self.assertFalse(result["lane_change"]["left"]["is_safe"])
        self.assertFalse(result["lane_change"]["right"]["is_safe"])
        self.assertIn(
            "lane_change_not_permitted",
            result["lane_change"]["left"]["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
