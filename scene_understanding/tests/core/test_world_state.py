import copy
import json
import math
import unittest
from pathlib import Path

from scene_understanding.core.world_state import (
    relative_kinematics,
    validate_world_state,
    vector3,
    vector_speed_mps,
    world_vector_to_ego,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "schemas" / "examples" / "world_state.example.json"


class WorldStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_is_valid(self):
        self.assertEqual(validate_world_state(self.example), [])

    def test_vector_speed_uses_magnitude(self):
        self.assertEqual(vector_speed_mps(vector3(3, 4, 0)), 5.0)

    def test_world_vector_to_ego_respects_yaw(self):
        value = world_vector_to_ego({"x": 0.0, "y": 10.0, "z": 2.0}, 90.0)
        self.assertAlmostEqual(value["longitudinal"], 10.0)
        self.assertAlmostEqual(value["lateral"], 0.0)
        self.assertAlmostEqual(value["vertical"], 2.0)

    def test_relative_kinematics_computes_closing_speed(self):
        result = relative_kinematics(
            ego_position_world_m=vector3(10, 20, 0),
            ego_velocity_world_mps=vector3(10, 0, 0),
            ego_yaw_deg=0,
            object_position_world_m=vector3(30, 20, 0),
            object_velocity_world_mps=vector3(6, 0, 0),
        )
        self.assertAlmostEqual(result["distance_m"], 20.0)
        self.assertAlmostEqual(result["relative_position_ego_m"]["longitudinal"], 20.0)
        self.assertAlmostEqual(result["relative_longitudinal_speed_mps"], -4.0)
        self.assertAlmostEqual(result["closing_speed_mps"], 4.0)

    def test_rejects_duplicate_object_ids(self):
        data = copy.deepcopy(self.example)
        data["objects"].append(copy.deepcopy(data["objects"][0]))
        errors = validate_world_state(data)
        self.assertTrue(any("duplicate ID carla_actor_42" in error for error in errors))

    def test_rejects_traffic_light_state_on_vehicle(self):
        data = copy.deepcopy(self.example)
        data["objects"][0]["traffic_light_state"] = "red"
        errors = validate_world_state(data)
        self.assertTrue(any("only traffic_light objects" in error for error in errors))

    def test_rejects_unknown_adjacent_lane_fields(self):
        data = copy.deepcopy(self.example)
        data["ego"]["adjacent_lanes"]["left"]["unexpected"] = True
        errors = validate_world_state(data)
        self.assertTrue(any("unexpected fields: unexpected" in error for error in errors))

    def test_rejects_nonfinite_metric(self):
        data = copy.deepcopy(self.example)
        data["objects"][0]["distance_m"] = math.inf
        errors = validate_world_state(data)
        self.assertTrue(any("distance_m: expected a finite number" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
