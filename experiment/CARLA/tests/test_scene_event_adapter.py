import unittest

from scene_event_adapter import scene_sensor_events


class SceneEventAdapterTests(unittest.TestCase):
    def test_converts_new_runner_events_to_shared_contract(self):
        result = scene_sensor_events(
            {
                "new_collision_events": [{"frame": 4, "other_actor_id": 9}],
                "new_lane_invasion_events": [{"frame": 5, "markings": ["Solid"]}],
            }
        )
        self.assertEqual(result["collisions"][0]["event_id"], "collision_4_0")
        self.assertEqual(result["collisions"][0]["other_actor_id"], "9")
        self.assertEqual(result["lane_invasions"][0]["event_id"], "lane_invasion_5_0")
        self.assertEqual(
            result["lane_invasions"][0]["crossed_lane_markings"], ["Solid"]
        )

    def test_ignores_missing_or_non_list_event_values(self):
        self.assertEqual(scene_sensor_events(None), {"collisions": [], "lane_invasions": []})
        self.assertEqual(
            scene_sensor_events({"new_collision_events": "bad", "new_lane_invasion_events": {}}),
            {"collisions": [], "lane_invasions": []},
        )


if __name__ == "__main__":
    unittest.main()
