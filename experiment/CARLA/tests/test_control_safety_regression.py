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

    def __init__(self, value, next_waypoint=None):
        self.transform = value
        self._next_waypoint = next_waypoint

    def next(self, _distance):
        return [self._next_waypoint] if self._next_waypoint is not None else []


class FakeVehicle:
    def __init__(self):
        self.current_transform = transform(yaw=0.0)

    def get_transform(self):
        return self.current_transform

    def get_velocity(self):
        return carla.Vector3D()


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


if __name__ == "__main__":
    unittest.main()
