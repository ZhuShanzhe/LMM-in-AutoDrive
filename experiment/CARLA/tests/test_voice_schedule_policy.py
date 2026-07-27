import math
import unittest

from continuous.route_manager import RouteManager
from control.voice_schedule_policy import VoiceSchedulePolicy


class Rotation:
    def __init__(self, yaw):
        self.yaw = yaw


class Location:
    def __init__(self, x, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Transform:
    def __init__(self, x, yaw=0.0):
        self.location = Location(x)
        self.rotation = Rotation(yaw)


class Waypoint:
    def __init__(self, x, yaw=0.0, lane_id=1, road_id=1):
        self.transform = Transform(x, yaw)
        self.lane_type = "driving"
        self.lane_id = lane_id
        self.road_id = road_id
        self.next_points = []
        self.left = None
        self.right = None

    def next(self, step_m):
        del step_m
        return self.next_points

    def get_left_lane(self):
        return self.left

    def get_right_lane(self):
        return self.right


class Map:
    def __init__(self, start):
        self.start = start

    def get_spawn_points(self):
        return []

    def get_waypoint(self, location, project_to_road=True):
        del location, project_to_road
        return self.start


class World:
    def __init__(self, start):
        self.map = Map(start)

    def get_map(self):
        return self.map


class VoiceSchedulePolicyTest(unittest.TestCase):
    def setUp(self):
        self.commands = [
            {"id": "speed", "announce_at_m": 0, "action": "keep_lane", "target_speed_kmh": 60},
            {"id": "turn", "announce_at_m": 100, "action": "turn_right", "target_speed_kmh": 50},
            {"id": "lane", "announce_at_m": 200, "action": "lane_change_left", "target_speed_kmh": 50},
        ]

    def test_policy_uses_latest_due_command_and_route_target(self):
        policy = VoiceSchedulePolicy(self.commands, default_speed_kmh=40)
        policy.set_context({"progress_m": 100, "route_target": {"x": 12, "y": 4, "z": 0}})
        intent = policy.decide({})
        self.assertEqual(intent["command_id"], "turn")
        self.assertEqual(intent["action"], "turn_right")
        self.assertEqual(intent["target_location"], {"x": 12, "y": 4, "z": 0})
        self.assertEqual(policy.telemetry()["emitted_command_ids"], ["speed", "turn"])

    def test_lane_change_defers_geometric_target_to_lane_controller(self):
        policy = VoiceSchedulePolicy(self.commands, default_speed_kmh=40)
        policy.set_context({"progress_m": 200, "route_target": {"x": 12, "y": 4, "z": 0}})
        intent = policy.decide({})
        self.assertEqual(intent["action"], "lane_change_left")
        self.assertNotIn("target_location", intent)

    def test_future_navigation_command_activates_after_its_announcement(self):
        commands = [
            {"id": "keep", "announce_at_m": 0, "action": "keep_lane", "target_speed_kmh": 60},
            {
                "id": "right", "announce_at_m": 100, "activate_at_m": 330,
                "action": "turn_right", "target_speed_kmh": 35,
            },
            {"id": "resume", "announce_at_m": 620, "action": "keep_lane", "target_speed_kmh": 50},
        ]
        policy = VoiceSchedulePolicy(commands, default_speed_kmh=40)
        policy.set_context({
            "progress_m": 250,
            "route_target": {"x": 1, "y": 2, "z": 0},
            "turn_route_target": {"x": 3, "y": 4, "z": 0},
        })
        self.assertEqual(policy.decide({})["command_id"], "keep")
        policy.set_context({
            "progress_m": 360,
            "route_target": {"x": 1, "y": 2, "z": 0},
            "turn_route_target": {"x": 3, "y": 4, "z": 0},
        })
        intent = policy.decide({})
        self.assertEqual(intent["command_id"], "right")
        self.assertEqual(intent["target_location"], {"x": 3, "y": 4, "z": 0})


class RouteDirectiveTest(unittest.TestCase):
    def test_turn_directive_selects_the_rightmost_branch(self):
        start = Waypoint(0, yaw=0)
        straight = Waypoint(5, yaw=0)
        right = Waypoint(5, yaw=35)
        start.next_points = [straight, right]
        right.next_points = [Waypoint(10, yaw=35)]
        manager = RouteManager(World(start))

        route = manager.build_route(
            start_location=Location(0),
            length_m=10,
            step_m=5,
            directives=[{"id": "right", "distance_m": 0, "action": "turn_right"}],
        )

        self.assertEqual(len(manager.applied_directives), 1)
        self.assertEqual(route[1]["yaw"], 35)

    def test_target_point_returns_a_future_route_point(self):
        start = Waypoint(0)
        next_point = Waypoint(5)
        final = Waypoint(10)
        start.next_points = [next_point]
        next_point.next_points = [final]
        manager = RouteManager(World(start))
        manager.build_route(start_location=Location(0), length_m=15, step_m=5)
        manager.progress_m = 5
        manager.current_index = 1

        target = manager.target_point(4)
        self.assertEqual(target["distance_m"], 10)


if __name__ == "__main__":
    unittest.main()
