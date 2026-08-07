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
    def __init__(self, x, yaw=0.0, y=0.0):
        self.location = Location(x, y)
        self.rotation = Rotation(yaw)


class Waypoint:
    def __init__(self, x, yaw=0.0, lane_id=1, road_id=1):
        self.transform = Transform(x, yaw)
        self.lane_type = "driving"
        self.lane_id = lane_id
        self.road_id = road_id
        self.section_id = 0
        self.is_junction = False
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


class Vehicle:
    def __init__(self, x, y=0.0, yaw=0.0):
        self.transform = Transform(x, yaw, y)

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform


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

    def test_resume_marks_earlier_required_commands_complete(self):
        policy = VoiceSchedulePolicy(self.commands, default_speed_kmh=40)

        policy.resume_to(150)
        policy.set_context({"progress_m": 150})
        intent = policy.decide({})

        self.assertEqual(intent["command_phase"], "WAITING")
        self.assertTrue(intent["continuous_safety_monitor"])
        self.assertEqual(intent["command_id"], "turn__continuous_cruise")
        self.assertIn("speed", policy.telemetry()["completed_command_ids"])
        self.assertIn("turn", policy.telemetry()["completed_command_ids"])

    def test_manual_driving_intent_is_not_overridden_by_schedule_speed(self):
        manual = {
            "schema_version": "1.2.0",
            "request_id": "manual-speed-40",
            "intent": {
                "steps": [{
                    "step_id": "step_1",
                    "action": "SET_SPEED",
                    "parameters": {"target_speed_mps": 11.111},
                    "trigger": {"type": "IMMEDIATE"},
                    "depends_on": [],
                    "preconditions": ["PATH_CLEAR"],
                    "on_blocked": "WAIT_FOR_SAFE",
                }]
            },
            "parse_result": {"status": "VALID", "confidence": 1.0, "latency_ms": 0.0},
        }
        policy = VoiceSchedulePolicy([{
            "id": "manual", "announce_at_m": 0, "action": "accelerate",
            "target_speed_kmh": 60, "driving_intent": manual,
        }], default_speed_kmh=45)
        policy.set_context({"progress_m": 0})
        result = policy.decide({})
        self.assertEqual(result["target_speed_kmh"], 39.9996)
        self.assertEqual(result["reason"], "manual_driving_intent")
        self.assertEqual(result["driving_intent"]["request_id"], "manual-speed-40")

    def test_configured_execution_contract_covers_speed_lane_and_turn(self):
        cases = [
            ({"id": "speed", "action": "accelerate", "target_speed_kmh": 60}, "SET_SPEED"),
            ({"id": "lane", "action": "lane_change_left", "target_speed_kmh": 45}, "CHANGE_LANE"),
            ({"id": "turn", "action": "turn_right", "target_speed_kmh": 30}, "TURN"),
        ]
        for command, expected_action in cases:
            with self.subTest(command=command["id"]):
                intent = VoiceSchedulePolicy._configured_driving_intent(command)
                self.assertEqual(intent["intent"]["steps"][0]["action"], expected_action)
                self.assertEqual(intent["parse_result"]["status"], "VALID")

    def test_completed_command_presents_success_then_waiting(self):
        policy = VoiceSchedulePolicy([{
            "id": "speed", "announce_at_m": 0,
            "action": "keep_lane", "target_speed_kmh": 45,
        }])
        policy.set_context({"progress_m": 0.0, "simulation_time_s": 0.0})
        policy.decide({})
        self.assertEqual(policy.telemetry()["command_presentation"]["phase"], "EXECUTING")
        policy.mark_completed("speed")
        self.assertEqual(policy.telemetry()["command_presentation"]["phase"], "SUCCESS")
        policy.set_context({"progress_m": 1.0, "simulation_time_s": 1.6})
        waiting = policy.decide({})
        self.assertEqual(waiting["command_phase"], "WAITING")
        self.assertEqual(waiting["target_speed_kmh"], 45.0)
        self.assertTrue(waiting["continuous_safety_monitor"])
        self.assertEqual(
            waiting["driving_intent"]["intent"]["steps"][0]["action"],
            "SET_SPEED",
        )
        presentation = policy.telemetry()["command_presentation"]
        self.assertIsNone(presentation["command_id"])
        self.assertEqual(presentation["voice_text"], "")

    def test_resume_skips_historical_success_hold(self):
        policy = VoiceSchedulePolicy([
            {"id": "first", "announce_at_m": 0, "action": "keep_lane", "target_speed_kmh": 30},
            {"id": "second", "announce_at_m": 200, "action": "keep_lane", "target_speed_kmh": 45},
        ])
        policy.resume_to(100)
        policy.set_context({"progress_m": 100, "simulation_time_s": 0.0})

        intent = policy.decide({})

        self.assertEqual(intent["command_phase"], "WAITING")
        self.assertEqual(intent["target_speed_kmh"], 30.0)

    def test_configured_turn_keeps_explicit_speed_setpoint(self):
        driving_intent = VoiceSchedulePolicy._configured_driving_intent({
            "id": "turn",
            "action": "turn_left",
            "target_speed_kmh": 30.0,
            "voice_text": "turn",
        })

        step = driving_intent["intent"]["steps"][0]

        self.assertEqual(step["action"], "TURN")
        self.assertAlmostEqual(
            step["parameters"]["target_speed_mps"], 30.0 / 3.6, places=5
        )

    def test_waiting_after_lane_change_does_not_reacquire_old_route_lane(self):
        policy = VoiceSchedulePolicy([{
            "id": "right",
            "announce_at_m": 0,
            "action": "lane_change_right",
            "target_speed_kmh": 45,
        }])
        policy.set_context({
            "progress_m": 0.0,
            "simulation_time_s": 0.0,
            "route_target": {"x": 20.0, "y": 3.5, "z": 0.0},
        })
        policy.decide({})
        policy.mark_completed("right")
        policy.set_context({
            "progress_m": 10.0,
            "simulation_time_s": 1.6,
            "route_target": {"x": 30.0, "y": 3.5, "z": 0.0},
        })

        waiting = policy.decide({})

        self.assertFalse(waiting["route_target_trusted"])

    def test_waiting_after_return_lane_change_reacquires_route(self):
        policy = VoiceSchedulePolicy([{
            "id": "return",
            "announce_at_m": 0,
            "action": "lane_change_right",
            "target_speed_kmh": 45,
            "return_to_route": True,
        }])
        policy.set_context({
            "progress_m": 0.0,
            "simulation_time_s": 0.0,
            "route_target": {"x": 20.0, "y": 0.0, "z": 0.0},
        })
        policy.decide({})
        policy.mark_completed("return")
        policy.set_context({
            "progress_m": 10.0,
            "simulation_time_s": 1.6,
            "route_target": {"x": 30.0, "y": 0.0, "z": 0.0},
        })

        waiting = policy.decide({})

        self.assertTrue(waiting["route_target_trusted"])


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
        self.assertEqual(target["distance_m"], 9)

    def test_target_point_interpolates_between_route_samples(self):
        start = Waypoint(0, yaw=0)
        next_point = Waypoint(5, yaw=45)
        start.next_points = [next_point]
        manager = RouteManager(World(start))
        manager.build_route(start_location=Location(0), length_m=10, step_m=5)
        manager.progress_m = 1
        manager.current_index = 0

        target = manager.target_point(2)

        self.assertAlmostEqual(target["x"], 3.0)
        # The sampled coordinates remain on the x axis, so their tangent is
        # zero degrees even if CARLA reports a premature connector yaw.
        self.assertAlmostEqual(target["yaw"], 0.0)
        self.assertAlmostEqual(target["distance_m"], 3.0)

    def test_progress_projects_smoothly_between_route_points(self):
        start = Waypoint(0)
        next_point = Waypoint(5)
        final = Waypoint(10)
        start.next_points = [next_point]
        next_point.next_points = [final]
        manager = RouteManager(World(start))
        manager.build_route(start_location=Location(0), length_m=15, step_m=5)

        progress = manager.update(Vehicle(2.5))

        self.assertAlmostEqual(progress, 2.5, places=2)
        self.assertAlmostEqual(manager.cross_track_error_m, 0.0, places=2)

    def test_progress_cannot_jump_to_a_distant_nearby_route_loop(self):
        start = Waypoint(0)
        points = [start]
        for index in range(1, 81):
            point = Waypoint(index * 5)
            points[-1].next_points = [point]
            points.append(point)
        manager = RouteManager(World(start))
        manager.build_route(start_location=Location(0), length_m=400, step_m=5)

        progress = manager.update(Vehicle(300))

        self.assertLessEqual(progress, 20.0)
        self.assertGreater(manager.cross_track_error_m, 250.0)


if __name__ == "__main__":
    unittest.main()
