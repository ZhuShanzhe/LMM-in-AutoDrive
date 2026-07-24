import unittest

from continuous.route_manager import RouteManager
from continuous.scenario_manager import ScenarioManager


class Location:
    def __init__(self, x, y=0.0):
        self.x = x
        self.y = y


class Ego:
    def __init__(self, x=0.0):
        self.location = Location(x)

    def get_location(self):
        return self.location


class World:
    def get_map(self):
        return object()


class ActiveScenario:
    def __init__(self):
        self.ticks = 0
        self.cleaned = False

    def setup(self):
        return None

    def tick(self):
        self.ticks += 1

    def finished(self):
        return self.ticks >= 1

    def destroy(self):
        self.cleaned = True


class ContinuousScenarioManagerTest(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.route = RouteManager(self.world)
        self.route.route = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "distance_m": 0.0},
            {"x": 10.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "distance_m": 10.0},
            {"x": 20.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "distance_m": 20.0},
        ]
        self.route.route_length_m = 20.0

    def test_progress_uses_the_nearest_forward_route_point(self):
        ego = Ego(11.0)
        self.assertEqual(self.route.update(ego), 10.0)
        ego.location = Location(19.0)
        self.assertEqual(self.route.update(ego), 20.0)

    def test_seek_initializes_progress_for_a_resumed_route_segment(self):
        self.assertEqual(self.route.seek(14.0), 10.0)
        self.assertEqual(self.route.current_index, 1)
        self.assertEqual(self.route.seek(100.0), 20.0)
        self.assertEqual(self.route.current_index, 2)

    def test_events_trigger_once_and_receive_the_shared_ego(self):
        manager = ScenarioManager(self.world, self.route)
        created = []

        def factory(world, ego, event):
            created.append((world, ego, event["scenario"]))
            return ActiveScenario()

        manager.register("static_obstacle", factory)
        manager.events = [{
            "distance_m": 10.0,
            "scenario": "static_obstacle",
            "triggered": False,
        }]
        ego = Ego(10.0)
        manager.tick(ego)
        manager.tick(ego)
        self.assertEqual(created, [(self.world, ego, "static_obstacle")])
        self.assertTrue(manager.events[0]["triggered"])


if __name__ == "__main__":
    unittest.main()
