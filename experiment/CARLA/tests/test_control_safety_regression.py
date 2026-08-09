import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import carla

from control.pid_controller import EgoPIDController
from control.protocol import normalize_intent
from run_control_experiment import RuleDecisionPolicy, prepare_output_directory


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
    def test_output_directory_exists_before_controller_log_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "scene1"
            resolved = prepare_output_directory(output)
            self.assertEqual(resolved, output)
            self.assertTrue(output.is_dir())

    def test_direct_high_level_vla_actions_compile_before_pid_control(self):
        cases = (
            ({"action": "follow"}, "keep_lane"),
            ({"action": "wait"}, "stop"),
            ({"action": "yield"}, "decelerate"),
            ({"action": "u_turn"}, "turn_left"),
            ({"action": "pull_over", "target_lane": "right"}, "lane_change_right"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                normalized = normalize_intent(raw, 45.0)
                self.assertEqual(normalized["action"], expected)
                self.assertTrue(normalized["reason"].startswith("compiled_high_level_"))

    def test_accelerate_uses_absolute_requested_speed(self):
        vehicle = FakeVehicle()
        vehicle.current_velocity = carla.Vector3D(x=25.0)
        controller = EgoPIDController(
            vehicle, FakeMap(FakeWaypoint(transform(x=10.0)))
        )

        target = controller._resolve_target_speed(
            {"action": "accelerate", "target_speed_kmh": 60.0}
        )

        self.assertEqual(target, 60.0)

    def test_keep_lane_accepts_route_target_across_junction_road_ids(self):
        vehicle = FakeVehicle()
        fallback = FakeWaypoint(transform(x=0.0, y=12.0), road_id=31)
        current = FakeWaypoint(
            transform(x=0.0, y=0.0, yaw=0.0),
            next_waypoint=fallback,
            road_id=31,
        )
        controller = EgoPIDController(vehicle, FakeMap(current))
        requested = {"x": 15.0, "y": 0.4, "z": 0.0}

        target = controller._target_location(
            current,
            {"action": "keep_lane", "target_location": requested},
            vehicle.current_transform,
        )

        self.assertAlmostEqual(target.x, 15.0)
        self.assertAlmostEqual(target.y, 0.4)

    def test_keep_lane_rejects_old_route_after_lane_change(self):
        vehicle = FakeVehicle()
        fallback = FakeWaypoint(transform(x=15.0, y=0.0))
        current = FakeWaypoint(
            transform(x=0.0, y=0.0, yaw=0.0),
            next_waypoint=fallback,
            lane_id=2,
        )
        controller = EgoPIDController(vehicle, FakeMap(current))

        target = controller._target_location(
            current,
            {
                "action": "keep_lane",
                "target_location": {"x": 15.0, "y": 3.5, "z": 0.0},
            },
            vehicle.current_transform,
        )

        self.assertIs(target, fallback.transform.location)

    def test_trusted_route_target_remains_available_for_cross_track_recovery(self):
        vehicle = FakeVehicle()
        fallback = FakeWaypoint(transform(x=15.0, y=0.0))
        current = FakeWaypoint(
            transform(x=0.0, y=0.0, yaw=0.0), next_waypoint=fallback,
        )
        controller = EgoPIDController(vehicle, FakeMap(current))
        requested = {"x": 15.0, "y": 8.0, "z": 0.0}

        target = controller._target_location(
            current,
            {
                "action": "keep_lane",
                "target_location": requested,
                "route_target_trusted": True,
            },
            vehicle.current_transform,
        )

        self.assertAlmostEqual(target.x, requested["x"])
        self.assertAlmostEqual(target.y, requested["y"])

    def test_trusted_route_heading_is_not_reversed_by_nearest_lane_projection(self):
        vehicle = FakeVehicle()
        vehicle.current_transform = transform(x=0.0, y=2.0, yaw=0.0)
        projected_lane = FakeWaypoint(transform(x=0.0, y=0.0, yaw=0.0))
        controller = EgoPIDController(vehicle, FakeMap(projected_lane))

        intent = {
            "action": "keep_lane",
            "target_location": {"x": 15.0, "y": 3.0, "z": 0.0},
            "route_target_trusted": True,
        }
        steer = 0.0
        for _ in range(3):
            steer = controller._lateral_control(intent, 0.05)

        self.assertGreater(steer, 0.0)

    def test_trusted_route_uses_planned_center_when_nearest_lane_flips(self):
        vehicle = FakeVehicle()
        vehicle.current_transform = transform(x=0.0, y=0.5, yaw=0.0)
        adjacent_lane = FakeWaypoint(transform(x=0.0, y=3.5, yaw=0.0), lane_id=2)
        controller = EgoPIDController(vehicle, FakeMap(adjacent_lane))
        intent = {
            "action": "keep_lane",
            "target_location": {"x": 15.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "route_target_trusted": True,
        }

        steer = 0.0
        for _ in range(3):
            steer = controller._lateral_control(intent, 0.05)

        self.assertLess(steer, 0.0)

    def test_trusted_curved_route_does_not_apply_target_tangent_as_centerline(self):
        vehicle = FakeVehicle()
        adjacent_lane = FakeWaypoint(transform(x=0.0, y=3.5, yaw=0.0), lane_id=2)
        controller = EgoPIDController(vehicle, FakeMap(adjacent_lane))
        intent = {
            "action": "keep_lane",
            "target_location": {"x": 10.0, "y": 3.0, "z": 0.0, "yaw": 45.0},
            "route_target_trusted": True,
        }

        steer = 0.0
        for _ in range(3):
            steer = controller._lateral_control(intent, 0.05)

        self.assertGreater(steer, 0.0)

    def test_trusted_curved_route_uses_current_planned_reference(self):
        vehicle = FakeVehicle()
        vehicle.current_transform = transform(x=0.0, y=0.7, yaw=0.0)
        adjacent_lane = FakeWaypoint(transform(x=0.0, y=3.5, yaw=0.0), lane_id=2)
        controller = EgoPIDController(vehicle, FakeMap(adjacent_lane))
        intent = {
            "action": "keep_lane",
            "target_location": {
                "x": 10.0,
                "y": 3.0,
                "z": 0.0,
                "yaw": 45.0,
                "reference": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            },
            "route_target_trusted": True,
        }

        steer = 0.0
        for _ in range(3):
            steer = controller._lateral_control(intent, 0.05)

        # Stationary ego, offset right of the reference line: the cross-track
        # term (atan2(-lat, v+2)) dominates and steers back toward the line.
        self.assertLess(steer, 0.0)
        self.assertLess(steer, 0.2)

    def test_keep_lane_uses_route_heading_to_select_current_lane_branch(self):
        vehicle = FakeVehicle()
        straight = FakeWaypoint(transform(x=15.0, y=0.0, yaw=0.0), lane_id=2)
        planned_branch = FakeWaypoint(
            transform(x=10.0, y=10.0, yaw=45.0), lane_id=2
        )
        current = FakeWaypoint(transform(yaw=0.0), lane_id=2)
        current.next = lambda _distance: [straight, planned_branch]
        route_waypoint = FakeWaypoint(transform(y=3.5, yaw=45.0), lane_id=1)

        class BranchMap:
            def get_waypoint(self, *_args, **_kwargs):
                return route_waypoint

        controller = EgoPIDController(vehicle, BranchMap())
        target = controller._target_location(
            current,
            {
                "action": "keep_lane",
                "target_location": {"x": 15.0, "y": 3.5, "z": 0.0},
            },
            vehicle.current_transform,
        )

        self.assertIs(target, planned_branch.transform.location)

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

    def test_straight_lane_deadband_rejects_small_waypoint_noise(self):
        vehicle = FakeVehicle()
        waypoint = FakeWaypoint(transform(x=10.0, y=0.04, yaw=0.4))
        controller = EgoPIDController(vehicle, FakeMap(waypoint))

        steer = controller._lateral_control(
            {"action": "keep_lane", "target_location": None}, 0.05
        )

        self.assertEqual(steer, 0.0)

    def test_steering_filter_prevents_one_frame_turn_spike(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))

        first = controller._filter_steering(0.45, "turn_left", 0.05)
        second = controller._filter_steering(-0.45, "turn_left", 0.05)

        self.assertGreater(first, 0.0)
        self.assertGreater(second, -0.45)
        self.assertLess(second, first)

    def test_steering_deadband_does_not_erase_persistent_small_correction(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))

        outputs = [
            controller._filter_steering(0.03, "keep_lane", 0.05)
            for _ in range(8)
        ]

        self.assertEqual(outputs[0], 0.0)
        self.assertGreater(outputs[-1], 0.008)

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

    def test_lane_change_completes_only_after_centering_and_heading_settle(self):
        vehicle = FakeVehicle()
        target = FakeWaypoint(transform(y=-5.25, yaw=-180.0), lane_id=-2)
        controller = EgoPIDController(vehicle, FakeMap(target))
        controller._lane_change_target_lane_id = -2

        controller._lane_change_target_lane_id = -3
        self.assertFalse(controller.get_execution_state()["lane_change_completed"])

        controller._lane_change_target_lane_id = -2
        for _ in range(9):
            self.assertFalse(controller.get_execution_state()["lane_change_completed"])
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

    def test_steering_rate_limit_is_time_based_mode_aware_and_speed_adaptive(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))
        controller._last_control = carla.VehicleControl(steer=0.0)

        lane_change = controller._smooth_control(
            0.0, 0.0, 1.0, dt=0.05, lateral_action="lane_change_left"
        )
        controller._last_control = carla.VehicleControl(steer=0.0)
        turn = controller._smooth_control(
            0.0, 0.0, 1.0, dt=0.05, lateral_action="turn_left"
        )
        controller._last_control = carla.VehicleControl(steer=0.0)
        lane_keep = controller._smooth_control(
            0.0, 0.0, 1.0, dt=0.05, lateral_action="keep_lane"
        )

        # Stationary vehicle: response scale is 1.6 (maximum authority for
        # slow junction work), so per-tick rates are 0.48/0.85/0.75 * 1.6.
        self.assertAlmostEqual(lane_change.steer, 0.48 * 1.6 * 0.05, places=6)
        self.assertAlmostEqual(turn.steer, 0.85 * 1.6 * 0.05, places=6)
        self.assertAlmostEqual(lane_keep.steer, 0.75 * 1.6 * 0.05, places=6)

        # At 70 km/h the response scale drops to 0.65: same command is
        # smoothed much harder so high-speed corrections stay gentle.
        vehicle.current_velocity = carla.Vector3D(x=70.0 / 3.6)
        controller._last_control = carla.VehicleControl(steer=0.0)
        fast_lane_change = controller._smooth_control(
            0.0, 0.0, 1.0, dt=0.05, lateral_action="lane_change_left"
        )
        self.assertAlmostEqual(
            fast_lane_change.steer, 0.48 * 0.65 * 0.05, places=6
        )

    def test_curvature_speed_cap_only_limits_a_sharp_upcoming_bend(self):
        vehicle = FakeVehicle()
        straight = FakeWaypoint(transform(x=30.0, yaw=3.0))
        sharp = FakeWaypoint(transform(x=20.0, y=20.0, yaw=24.0))
        tight = FakeWaypoint(transform(x=30.0, y=30.0, yaw=60.0))
        current = FakeWaypoint(transform(yaw=0.0))
        current.next = lambda _distance: [straight]
        controller = EgoPIDController(vehicle, FakeMap(current))
        self.assertEqual(controller._curvature_speed_cap({"action": "keep_lane"}), 100.0)

        current.next = lambda _distance: [sharp]
        sharp_curvature = math.radians(24.0) / 30.0
        sharp_expected = math.sqrt(2.2 / sharp_curvature) * 3.6
        sharp_cap = controller._curvature_speed_cap({"action": "keep_lane"})
        self.assertAlmostEqual(sharp_cap, sharp_expected, places=4)

        current.next = lambda _distance: [tight]
        tight_curvature = math.radians(60.0) / 30.0
        tight_cap = controller._curvature_speed_cap({"action": "keep_lane"})
        self.assertLess(tight_cap, sharp_cap)
        self.assertAlmostEqual(
            tight_cap,
            min(100.0, math.sqrt(2.2 / tight_curvature) * 3.6),
            places=4,
        )

        # The cap is an actuator ceiling: it never raises the requested speed.
        current.next = lambda _distance: [sharp]
        self.assertEqual(
            controller._curvature_speed_cap(
                {"action": "keep_lane", "target_speed_kmh": 30.0}
            ),
            30.0,
        )

    def test_route_curvature_caps_speed_from_reference_heading(self):
        vehicle = FakeVehicle()
        straight = FakeWaypoint(transform(x=30.0, yaw=0.0))
        current = FakeWaypoint(transform(yaw=0.0), next_waypoint=straight)
        controller = EgoPIDController(vehicle, FakeMap(current))

        cap = controller._curvature_speed_cap({
            "action": "keep_lane",
            "target_location": {
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 40.0,
                "reference": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            },
            "route_target_trusted": True,
        })

        expected = math.sqrt(2.2 / (math.radians(40.0) / 10.0)) * 3.6
        self.assertAlmostEqual(cap, expected, places=4)

        straight_cap = controller._curvature_speed_cap({
            "action": "keep_lane",
            "target_location": {
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "reference": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            },
            "route_target_trusted": True,
        })
        self.assertEqual(straight_cap, 100.0)

    def test_dynamic_steering_limit_tracks_curvature_speed_and_lane_width(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))

        base = controller._dynamic_steering_limit(
            speed_kmh=20.0,
            heading_error_rad=math.radians(10.0),
            curvature=0.05,
            lane_width_m=3.5,
        )
        tighter = controller._dynamic_steering_limit(
            speed_kmh=20.0,
            heading_error_rad=math.radians(10.0),
            curvature=0.12,
            lane_width_m=3.5,
        )
        slower = controller._dynamic_steering_limit(
            speed_kmh=10.0,
            heading_error_rad=math.radians(10.0),
            curvature=0.05,
            lane_width_m=3.5,
        )
        faster = controller._dynamic_steering_limit(
            speed_kmh=60.0,
            heading_error_rad=math.radians(10.0),
            curvature=0.05,
            lane_width_m=3.5,
        )
        wider = controller._dynamic_steering_limit(
            speed_kmh=20.0,
            heading_error_rad=math.radians(10.0),
            curvature=0.05,
            lane_width_m=4.5,
        )

        self.assertGreater(tighter, base)
        self.assertGreater(slower, base)
        self.assertLess(faster, base)
        self.assertGreaterEqual(wider, base)

    def test_predict_arc_points_follow_bicycle_kinematics(self):
        vehicle = FakeVehicle()
        controller = EgoPIDController(vehicle, FakeMap(FakeWaypoint(transform())))
        origin = carla.Location(x=0.0, y=0.0, z=0.0)

        straight = controller._predict_arc_points(
            origin, 0.0, 0.0, 6.0, step_m=2.0
        )
        self.assertAlmostEqual(straight[-1][1], 6.0, places=6)
        self.assertAlmostEqual(straight[-1][2], 0.0, places=6)

        arc = controller._predict_arc_points(
            origin, 0.0, 0.1, 10.0, step_m=5.0
        )
        final_heading = math.atan2(
            arc[-1][2] - origin.y, arc[-1][1] - origin.x
        )
        self.assertAlmostEqual(final_heading, 0.5, places=2)

    def test_controller_releases_emergency_after_two_clear_frames(self):
        vehicle = FakeVehicle()
        vehicle.current_velocity = carla.Vector3D(x=12.0)
        waypoint = FakeWaypoint(transform(x=8.0, yaw=0.0))
        controller = EgoPIDController(vehicle, FakeMap(waypoint))

        emergency, _ = controller.run_step(
            {"action": "emergency_brake", "target_speed_kmh": 0.0}, 0.05
        )
        first_clear, _ = controller.run_step(
            {"action": "keep_lane", "target_speed_kmh": 50.0}, 0.05
        )
        second_clear, _ = controller.run_step(
            {"action": "keep_lane", "target_speed_kmh": 50.0}, 0.05
        )

        self.assertEqual(emergency.brake, 1.0)
        self.assertEqual(first_clear.brake, 1.0)
        self.assertLess(second_clear.brake, 1.0)
        self.assertFalse(controller.get_execution_state()["emergency_latched"])


if __name__ == "__main__":
    unittest.main()
