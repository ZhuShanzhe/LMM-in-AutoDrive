import copy
import json
import unittest
from pathlib import Path

from scene_understanding.core.object_matcher import (
    normalize_instruction_reference,
    select_world_object,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "schemas" / "examples" / "world_state.example.json"


class ObjectMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(WORLD_STATE_EXAMPLE.read_text(encoding="utf-8"))

    def test_normalizes_plan_vocabulary(self):
        expected = {
            "行人": "pedestrian",
            "前车": "front_vehicle",
            "慢车": "slow_vehicle",
            "左车道": "left_lane",
            "路口": "junction",
        }
        for text, target_type in expected.items():
            with self.subTest(text=text):
                self.assertEqual(
                    normalize_instruction_reference(text)["target_type"],
                    target_type,
                )

    def test_accepts_common_parser_field_names(self):
        reference = normalize_instruction_reference({"target_object": "pedestrian"})
        self.assertEqual(reference["target_type"], "pedestrian")

    def test_selects_front_vehicle_in_ego_lane(self):
        reference = normalize_instruction_reference("前车")
        selected, count = select_world_object(reference, self.world_state)
        self.assertEqual(count, 1)
        self.assertEqual(selected["object_id"], "carla_actor_42")

    def test_selects_slow_vehicle(self):
        state = copy.deepcopy(self.world_state)
        fast = copy.deepcopy(state["objects"][0])
        fast["object_id"] = "carla_actor_43"
        fast["source_object_id"] = "43"
        fast["speed_mps"] = 15.0
        fast["distance_m"] = 10.0
        fast["relative_position_ego_m"]["longitudinal"] = 10.0
        fast["semantic_matches"] = []
        state["objects"].append(fast)
        reference = normalize_instruction_reference("慢车")
        selected, count = select_world_object(reference, state)
        self.assertEqual(count, 1)
        self.assertEqual(selected["object_id"], "carla_actor_42")

    def test_selects_extended_actor_categories(self):
        cases = {
            "cyclist": "cyclist",
            "traffic_cone": "traffic_cone",
            "obstacle": "road_barrier",
            "road_hazard": "other",
        }

        for index, (target_type, category) in enumerate(
            cases.items(),
            start=100,
        ):
            with self.subTest(target_type=target_type):
                state = copy.deepcopy(self.world_state)
                actor = copy.deepcopy(state["objects"][0])

                actor["object_id"] = f"carla_actor_{index}"
                actor["source_object_id"] = str(index)
                actor["category"] = category
                actor["subtype"] = f"test.{category}"
                actor["speed_mps"] = 0.0
                actor["distance_m"] = 15.0
                actor["relative_position_ego_m"][
                    "longitudinal"
                ] = 15.0
                actor["relative_position_ego_m"][
                    "lateral"
                ] = 0.0
                actor["lane_relation"] = "ego_lane"
                actor["semantic_matches"] = []

                state["objects"] = [actor]

                reference = {
                    "raw_text": target_type,
                    "target_type": target_type,
                    "position_hint": "front",
                    "lane_hint": "ego_lane",
                }

                selected, count = select_world_object(
                    reference,
                    state,
                )

                self.assertEqual(count, 1)
                self.assertIsNotNone(selected)
                self.assertEqual(
                    selected["object_id"],
                    f"carla_actor_{index}",
                )



if __name__ == "__main__":
    unittest.main()
