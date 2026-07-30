from __future__ import annotations

import unittest
from types import SimpleNamespace

from run_scene2_closed_loop import world_state_sensor_events


class Scene2EventAdapterTests(unittest.TestCase):
    def test_legacy_lane_event_is_adapted_to_world_state_contract(self) -> None:
        safety = SimpleNamespace(
            collisions=[],
            lane_invasions=[
                {
                    "frame": 42,
                    "simulation_time_s": 1.5,
                    "markings": ["Broken"],
                }
            ],
        )

        result = world_state_sensor_events(safety)

        self.assertEqual(
            result["lane_invasions"][0],
            {
                "event_id": "lane_invasion_42_0",
                "frame": 42,
                "timestamp_s": 1.5,
                "crossed_lane_markings": ["Broken"],
            },
        )

    def test_collision_preserves_measured_impulse(self) -> None:
        safety = SimpleNamespace(
            collisions=[
                {
                    "event_id": "collision-1",
                    "frame": 43,
                    "timestamp_s": 2.0,
                    "other_actor_id": 99,
                    "normal_impulse_ns": {
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    },
                    "impulse_magnitude_ns": 3.741657,
                }
            ],
            lane_invasions=[],
        )

        event = world_state_sensor_events(safety)["collisions"][0]

        self.assertEqual(event["other_actor_id"], "99")
        self.assertEqual(event["normal_impulse_ns"]["z"], 3.0)
        self.assertAlmostEqual(event["impulse_magnitude_ns"], 3.741657)

    def test_cursor_consumes_each_event_once(self) -> None:
        safety = SimpleNamespace(
            collisions=[],
            lane_invasions=[
                {
                    "frame": 42,
                    "simulation_time_s": 2.1,
                    "markings": ["Solid"],
                }
            ],
        )
        cursor: dict[str, int] = {}

        first = world_state_sensor_events(safety, cursor)
        second = world_state_sensor_events(safety, cursor)

        self.assertEqual(len(first["lane_invasions"]), 1)
        self.assertEqual(second["lane_invasions"], [])
        self.assertEqual(cursor["lane_invasions"], 1)


if __name__ == "__main__":
    unittest.main()
