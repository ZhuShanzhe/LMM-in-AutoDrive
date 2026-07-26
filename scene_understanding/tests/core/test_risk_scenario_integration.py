import copy
import json
import math
import unittest
from pathlib import Path

from scene_understanding.core.risk_assessment import (
    assess_world_state,
    validate_risk_assessment,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = (
    ROOT
    / "schemas"
    / "examples"
    / "world_state.example.json"
)


class RiskScenarioIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(
            WORLD_STATE_EXAMPLE.read_text(encoding="utf-8")
        )

    def _single_object_state(
        self,
        *,
        category="vehicle",
        subtype="vehicle.lincoln.mkz_2020",
        lane_relation="ego_lane",
    ):
        state = copy.deepcopy(self.world_state)
        obj = state["objects"][0]
        obj["category"] = category
        obj["subtype"] = subtype
        obj["lane_relation"] = lane_relation
        obj["semantic_matches"] = []
        state["objects"] = [obj]
        return state, obj

    @staticmethod
    def _set_relative_motion(
        obj,
        *,
        longitudinal,
        lateral,
        longitudinal_velocity,
        lateral_velocity,
    ):
        distance = math.hypot(longitudinal, lateral)
        if distance <= 1e-9:
            closing_speed = 0.0
        else:
            separation_rate = (
                longitudinal * longitudinal_velocity
                + lateral * lateral_velocity
            ) / distance
            closing_speed = max(0.0, -separation_rate)

        obj["relative_position_ego_m"] = {
            "longitudinal": float(longitudinal),
            "lateral": float(lateral),
            "vertical": 0.0,
        }
        obj["relative_velocity_ego_mps"] = {
            "longitudinal": float(longitudinal_velocity),
            "lateral": float(lateral_velocity),
            "vertical": 0.0,
        }
        obj["distance_m"] = float(distance)
        obj["relative_longitudinal_speed_mps"] = float(
            longitudinal_velocity
        )
        obj["closing_speed_mps"] = float(closing_speed)

    def test_adjacent_braking_vehicle_only_blocks_lane_change(self):
        state, obj = self._single_object_state(
            lane_relation="left_adjacent_lane"
        )
        self._set_relative_motion(
            obj,
            longitudinal=25.0,
            lateral=-3.5,
            longitudinal_velocity=-8.0,
            lateral_velocity=0.0,
        )

        result = assess_world_state(state)

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(result["risk_level"], "none")
        self.assertEqual(
            result["recommended_action"],
            "maintain_speed",
        )

        left = result["lane_change"]["left"]
        self.assertFalse(left["is_safe"])
        self.assertIn(
            obj["object_id"],
            left["blocking_object_ids"],
        )
        self.assertIn(
            "target_lane_ttc_at_most_4s",
            left["reason_codes"],
        )

    def test_predicts_adjacent_vehicle_cut_in_before_lane_entry(self):
        state, obj = self._single_object_state(
            lane_relation="left_adjacent_lane"
        )
        self._set_relative_motion(
            obj,
            longitudinal=18.0,
            lateral=-3.5,
            longitudinal_velocity=-1.0,
            lateral_velocity=2.0,
        )

        result = assess_world_state(state)
        assessment = result["object_assessments"][0]

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertTrue(
            assessment["relevant_to_ego_path"]
        )
        self.assertEqual(
            assessment["risk_level"],
            "high",
        )
        self.assertIn(
            "cut_in_path_conflict_imminent",
            assessment["reason_codes"],
        )
        self.assertEqual(
            result["recommended_action"],
            "emergency_brake",
        )

    def test_predicts_roadside_pedestrian_path_conflict(self):
        state, obj = self._single_object_state(
            category="pedestrian",
            subtype="walker.pedestrian.0001",
            lane_relation="roadside",
        )
        obj["speed_mps"] = 2.0
        self._set_relative_motion(
            obj,
            longitudinal=18.0,
            lateral=3.5,
            longitudinal_velocity=-10.0,
            lateral_velocity=-2.0,
        )

        result = assess_world_state(state)
        assessment = result["object_assessments"][0]

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertTrue(
            assessment["relevant_to_ego_path"]
        )
        self.assertEqual(
            assessment["risk_level"],
            "high",
        )
        self.assertIn(
            "pedestrian_path_conflict_imminent",
            assessment["reason_codes"],
        )
        self.assertEqual(
            result["recommended_action"],
            "emergency_brake",
        )

    def test_static_obstacle_inside_stopping_distance_is_emergency(
        self,
    ):
        state, obj = self._single_object_state(
            category="traffic_cone",
            subtype="static.prop.trafficcone01",
            lane_relation="ego_lane",
        )
        state["ego"]["speed_mps"] = 20.0
        state["ego"]["velocity_world_mps"] = {
            "x": 20.0,
            "y": 0.0,
            "z": 0.0,
        }
        obj["speed_mps"] = 0.0
        obj["velocity_world_mps"] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self._set_relative_motion(
            obj,
            longitudinal=30.0,
            lateral=0.0,
            longitudinal_velocity=-20.0,
            lateral_velocity=0.0,
        )

        result = assess_world_state(state)
        assessment = result["object_assessments"][0]

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(
            assessment["risk_level"],
            "high",
        )
        self.assertIn(
            "insufficient_emergency_stopping_distance",
            assessment["reason_codes"],
        )
        self.assertEqual(
            result["recommended_action"],
            "emergency_brake",
        )

    def test_static_obstacle_in_comfortable_braking_zone_decelerates(
        self,
    ):
        state, obj = self._single_object_state(
            category="traffic_cone",
            subtype="static.prop.trafficcone01",
            lane_relation="ego_lane",
        )
        state["ego"]["speed_mps"] = 20.0
        state["ego"]["velocity_world_mps"] = {
            "x": 20.0,
            "y": 0.0,
            "z": 0.0,
        }
        obj["speed_mps"] = 0.0
        obj["velocity_world_mps"] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self._set_relative_motion(
            obj,
            longitudinal=60.0,
            lateral=0.0,
            longitudinal_velocity=-20.0,
            lateral_velocity=0.0,
        )

        result = assess_world_state(state)
        assessment = result["object_assessments"][0]

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(
            assessment["risk_level"],
            "medium",
        )
        self.assertIn(
            "insufficient_stopping_distance",
            assessment["reason_codes"],
        )
        self.assertNotIn(
            "insufficient_emergency_stopping_distance",
            assessment["reason_codes"],
        )
        self.assertEqual(
            result["recommended_action"],
            "decelerate",
        )

    def test_distant_cut_in_is_monitored_without_emergency_brake(
        self,
    ):
        state, obj = self._single_object_state(
            lane_relation="left_adjacent_lane"
        )
        self._set_relative_motion(
            obj,
            longitudinal=60.0,
            lateral=-3.5,
            longitudinal_velocity=0.0,
            lateral_velocity=2.0,
        )

        result = assess_world_state(state)
        assessment = result["object_assessments"][0]

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertTrue(
            assessment["relevant_to_ego_path"]
        )
        self.assertEqual(
            assessment["risk_level"],
            "low",
        )
        self.assertNotIn(
            "cut_in_path_conflict_imminent",
            assessment["reason_codes"],
        )
        self.assertEqual(
            result["recommended_action"],
            "monitor",
        )

    def test_imminent_pre_collision_ttc_requires_emergency_brake(
        self,
    ):
        state, obj = self._single_object_state()
        self._set_relative_motion(
            obj,
            longitudinal=5.0,
            lateral=0.0,
            longitudinal_velocity=-10.0,
            lateral_velocity=0.0,
        )

        result = assess_world_state(state)

        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(
            result["object_assessments"][0]["ttc_s"],
            0.5,
        )
        self.assertEqual(
            result["recommended_action"],
            "emergency_brake",
        )


if __name__ == "__main__":
    unittest.main()
