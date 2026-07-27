import json
import unittest
from pathlib import Path

from continuous.scenario_manager import ScenarioManager


class FakeRouteManager:
    def __init__(self):
        self.progress_m = 0.0

    def update(self, _ego_vehicle):
        return self.progress_m


class FakeEvent:
    def __init__(self, _world, _ego, event):
        self.event = event
        self.status = "INITIALIZED"
        self.ticks = 0

    def setup(self):
        self.status = "ACTIVE"

    def tick(self):
        self.ticks += 1
        if self.ticks >= 1:
            self.status = "COMPLETED"

    def finished(self):
        return self.status in {"COMPLETED", "FAILED"}

    def get_status(self):
        return {"status": self.status, "ticks": self.ticks}

    def destroy(self):
        pass


class ContinuousEventScheduleTests(unittest.TestCase):
    def test_event_is_visible_before_its_behavior_trigger(self):
        route = FakeRouteManager()
        manager = ScenarioManager(object(), route)
        manager.register("fake", lambda world, ego, event: FakeEvent(world, ego, event))
        manager.set_events([{
            "id": "brake",
            "scenario": "fake",
            "distance_m": 420,
            "activate_at_m": 300,
        }])

        route.progress_m = 299
        manager.tick(object())
        self.assertEqual(manager.snapshot()["pending"], ["brake"])

        route.progress_m = 300
        manager.tick(object())
        self.assertEqual(manager.snapshot()["active"], ["brake"])
        activation = manager.drain_event_log()[0]
        self.assertEqual(activation["transition"], "activated")
        self.assertEqual(activation["route_progress_m"], 300.0)

        route.progress_m = 420
        manager.tick(object())
        self.assertEqual(manager.snapshot()["completed"], ["brake"])

    def test_demo_events_arm_before_their_action_distance(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "basic_track_5km_demo.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for event in config["events"]:
            self.assertLess(event["activate_at_m"], event["distance_m"])
        lead, cut_in, pedestrian = config["events"]
        self.assertEqual(lead["brake_at_route_progress_m"], lead["distance_m"])
        self.assertEqual(cut_in["merge_at_route_progress_m"], cut_in["distance_m"])
        self.assertEqual(pedestrian["cross_at_route_progress_m"], pedestrian["distance_m"])


if __name__ == "__main__":
    unittest.main()
