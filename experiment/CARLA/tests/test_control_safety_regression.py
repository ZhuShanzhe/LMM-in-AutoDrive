import math
import unittest
from types import SimpleNamespace

import carla

from control.pid_controller import EgoPIDController
from run_control_experiment import RuleDecisionPolicy


def transform(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(
        location=carla.Location(x=x, y=y, z=0.0),
        rotation=SimpleNamespace(yaw=yaw),
    )


class FakeWaypoint:
    lane_id = 1
    lane_type = carla.LaneType.Driving
    road_id = 1

    def __init__(self, value, next_waypoint=None, lane_id=1, road_id=1):
        self.transform = value
        self._next_waypoint = next_waypoint
        self.lane_id = lane_id
        self.road_id = road_id
        self._left_lane = None
        self._right_lane = None

    def next(self, _distance):
        return [self._next_waypoint] if self._next_waypoint is not None else []

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane


class FakeVehicle:
    def __init__(self):
        self.current_transform = transform(yaw=0.0)
        self.current_velocity = carla.Vector3D()

    def get_transform(self):
        return self.current_transform

    def get_location(self):
        return self.current_transform.location

    def get_velocity(self):
        return self.current_velocity


class FakeMap:
    def __init__(self, waypoint):
        self.waypoint = waypoint

    def get_waypoint(self, *_args, **_kwargs):
        return self.waypoint


class ControlSafetyRegressionTests(unittest.TestCase):
    def test_keep_lane_follows_current_curved_waypoint(self):
        vehicle = FakeVehicle()
        next_waypoint = FakeWaypoint(transform(y=8.0, yaw=90.0))
        current_waypoint = FakeWaypoint(
            transform(yaw=90.0),
            next_waypoint=next_waypoint,
        )
        controller = EgoPIDController(vehicle, FakeMap(current_waypoint))

        vehicle.current_transform = transform(yaw=90.0)
        steer = controller._lateral_control(
            {"action": "keep_lane", "target_location": None}
        )

        self.assertTrue(math.isclose(steer, 0.0, abs_tol=1e-6))

    def test_turn_action_selects_requested_junction_branch(self):
        vehicle = FakeVehicle()
        straight = FakeWaypoint(transform(x=8.0, yaw=0.0))
        right = FakeWaypoint(transform(x=6.0, y=5.0, yaw=40.0))
        current_waypoint = FakeWaypoint(transform(yaw=0.0))
        current_waypoint.next = lambda _distance: [straight, right]
        controller = EgoPIDController(vehicle, FakeMap(current_waypoint))

        selected = controller._target_waypoint(
            current_waypoint, {"action": "turn_right"}
        )

        self.assertIs(selected, right)

    def test_lane_change_uses_vehicle_relative_left_side(self):
        vehicle = FakeVehicle()
        current = FakeWaypoint(transform(y=-1.75, yaw=-180.0), lane_id=-1)
        vehicle_left = FakeWaypoint(transform(y=1.75, yaw=-180.0), lane_id=-2)
        vehicle_right = FakeWaypoint(transform(y=-5.25, yaw=-180.0), lane_id=-2)
        current._left_lane = vehicle_left
        current._right_lane = vehicle_right
        controller = EgoPIDController(vehicle, FakeMap(current))

        selected_left = controller._adjacent_driving_lane(current, "lane_change_left")
        selected_right = controller._adjacent_driving_lane(current, "lane_change_right")

        self.assertIs(selected_left, vehicle_left)
        self.assertIs(selected_right, vehicle_right)

    def test_lane_change_requires_heading_alignment_before_completion(self):
        vehicle = FakeVehicle()
        target = FakeWaypoint(transform(y=-5.25, yaw=-180.0), lane_id=-2)
        controller = EgoPIDController(vehicle, FakeMap(target))
        controller._lane_change_target_lane_id = -2

        vehicle.current_transform = transform(y=-5.25, yaw=-174.0)
        self.assertFalse(controller.get_execution_state()["lane_change_completed"])

        vehicle.current_transform = transform(y=-5.25, yaw=-180.0)
        self.assertTrue(controller.get_execution_state()["lane_change_completed"])

    def test_lane_change_keeps_lateral_guidance_during_settle_period(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))
        lane_change = {"action": "lane_change_right", "target_speed_kmh": 45.0}

        self.assertEqual(
            controller._lateral_intent_for_control(lane_change)["action"],
            "lane_change_right",
        )
        settled = controller._lateral_intent_for_control(
            {"action": "keep_lane", "target_speed_kmh": 45.0}
        )
        self.assertEqual(settled["action"], "lane_change_right")
        self.assertEqual(settled["target_speed_kmh"], 45.0)

    def test_emergency_decision_remains_latched(self):
        policy = RuleDecisionPolicy("emergency_brake", 25.0)
        danger = {
            "ego": {"speed(km/h)": 20.0},
            "vehicles": [
                {
                    "distance": 10.0,
                    "speed_kmh": 0.0,
                    "relative_position": {"x": 10.0, "y": 0.0},
                }
            ],
            "pedestrians": [],
        }
        clear = {
            "ego": {"speed(km/h)": 0.0},
            "vehicles": [],
            "pedestrians": [],
        }

        self.assertEqual(policy.decide(danger)["action"], "emergency_brake")
        self.assertEqual(policy.decide(clear)["action"], "emergency_brake")

    def test_moderate_closing_speed_does_not_latch_emergency(self):
        policy = RuleDecisionPolicy("emergency_brake", 25.0)
        moderate = {
            "ego": {"speed(km/h)": 20.0},
            "vehicles": [
                {
                    "distance": 30.0,
                    "speed_kmh": 10.0,
                    "relative_position": {"x": 30.0, "y": 0.0},
                }
            ],
            "pedestrians": [],
        }
        clear = {
            "ego": {"speed(km/h)": 10.0},
            "vehicles": [],
            "pedestrians": [],
        }

        self.assertEqual(policy.decide(moderate)["action"], "decelerate")
        self.assertEqual(policy.decide(clear)["action"], "keep_lane")

    def test_nominal_control_is_smoothed_but_emergency_brake_is_immediate(self):
        vehicle = FakeVehicle()
        waypoint = FakeWaypoint(transform(x=8.0, yaw=0.0))
        controller = EgoPIDController(vehicle, FakeMap(waypoint))

        first, _ = controller.run_step(
            {"action": "keep_lane", "target_speed_kmh": 50.0}, 0.05
        )
        vehicle.current_velocity = carla.Vector3D(x=15.0)
        second, _ = controller.run_step(
            {"action": "decelerate", "target_speed_kmh": 0.0}, 0.05
        )
        emergency, _ = controller.run_step(
            {"action": "emergency_brake", "target_speed_kmh": 0.0}, 0.05
        )

        self.assertLessEqual(second.brake, first.brake + 0.24 + 1e-6)
        self.assertEqual(second.throttle, 0.0)
        self.assertEqual(emergency.throttle, 0.0)
        self.assertEqual(emergency.brake, 1.0)


if __name__ == "__main__":
    unittest.main()
