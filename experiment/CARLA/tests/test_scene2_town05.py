"""Contract and route-progress tests for the Town05 Scene 2 runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_complex_avoidance_town05 import (
    SafetyMonitor,
    audit_command_route_alignment,
    carla_world_can_be_reused,
    configure_competition_artifacts,
    configure_traffic_manager_physics,
    lane_invasion_is_restricted,
    load_config,
    ready_commands_in_order,
    road_option_name,
    route_aware_preview_speed_kmh,
    planned_turn_window_active,
    route_centering_steer_correction,
)
from evaluation.multimodal import (
    ExactFrameSensorSuite,
    REQUIRED_SENSOR_NAMES,
)
from scene2_runtime_interface import (
    build_multimodal_frame_bundle,
    build_scheduled_driving_intent,
)
from scene_understanding.src.control_decision import _map_step_action
from scenarios.complex.town05_scene2 import (
    RouteProgressTracker,
    ScriptedWalker,
    cumulative_route_distances,
    materialize_event_variants,
    stable_variant_index,
    vehicle_spawn_offsets,
    walker_spawn_offsets,
)


@dataclass
class Location:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Transform:
    location: Location


@dataclass
class Waypoint:
    transform: Transform


def route_at(*coordinates: tuple[float, float]):
    return [
        (Waypoint(Transform(Location(x, y))), None)
        for x, y in coordinates
    ]


class Scene2Town05Tests(unittest.TestCase):
    def test_hybrid_physics_keeps_auditable_near_field_radius(self):
        class FakeTrafficManager:
            def __init__(self):
                self.calls = []

            def set_hybrid_physics_mode(self, enabled):
                self.calls.append(("mode", enabled))

            def set_hybrid_physics_radius(self, radius_m):
                self.calls.append(("radius", radius_m))

        manager = FakeTrafficManager()
        configure_traffic_manager_physics(
            manager,
            hybrid_enabled=True,
            hybrid_radius_m=100.0,
        )
        self.assertEqual(manager.calls, [("mode", True), ("radius", 100.0)])

        manager = FakeTrafficManager()
        configure_traffic_manager_physics(
            manager,
            hybrid_enabled=False,
            hybrid_radius_m=100.0,
        )
        self.assertEqual(manager.calls, [("mode", False)])

        with self.assertRaisesRegex(ValueError, "positive"):
            configure_traffic_manager_physics(
                manager, hybrid_enabled=True, hybrid_radius_m=0.0
            )

    def test_logs_only_competition_keeps_online_vla_and_skips_disk_artifacts(self):
        args = SimpleNamespace(
            competition_run=True,
            competition_logs_only=True,
            no_video=False,
            record_multimodal=True,
            record_ground_truth=True,
            video_overlay=True,
        )

        configure_competition_artifacts(args, vla_enabled=True)

        self.assertTrue(args.no_video)
        self.assertFalse(args.record_multimodal)
        self.assertFalse(args.record_ground_truth)
        self.assertFalse(args.video_overlay)

    def test_full_competition_retains_all_artifact_recorders(self):
        args = SimpleNamespace(
            competition_run=True,
            competition_logs_only=False,
            no_video=False,
            record_multimodal=False,
            record_ground_truth=False,
            video_overlay=False,
        )

        configure_competition_artifacts(args, vla_enabled=True)

        self.assertTrue(args.record_multimodal)
        self.assertTrue(args.record_ground_truth)
        self.assertTrue(args.video_overlay)

    def test_logs_only_requires_competition_and_vla(self):
        args = SimpleNamespace(
            competition_run=False,
            competition_logs_only=True,
            no_video=False,
            record_multimodal=False,
            record_ground_truth=False,
            video_overlay=False,
        )
        with self.assertRaisesRegex(ValueError, "competition-run"):
            configure_competition_artifacts(args, vla_enabled=True)
        args.competition_run = True
        with self.assertRaisesRegex(ValueError, "vla-checkpoint"):
            configure_competition_artifacts(args, vla_enabled=False)

    def test_matching_actor_clean_world_can_be_reused(self):
        class FakeWorld:
            def __init__(self, map_name, actor_types=()):
                self.map_name = map_name
                self.actor_types = actor_types

            def get_map(self):
                return SimpleNamespace(name=self.map_name)

            def get_actors(self):
                return [
                    SimpleNamespace(type_id=type_id)
                    for type_id in self.actor_types
                ]

        self.assertTrue(
            carla_world_can_be_reused(
                FakeWorld("Carla/Maps/Town05_Opt"), "Town05_Opt"
            )
        )
        self.assertFalse(
            carla_world_can_be_reused(
                FakeWorld("Carla/Maps/Town04"), "Town05_Opt"
            )
        )
        self.assertFalse(
            carla_world_can_be_reused(
                FakeWorld("Carla/Maps/Town05_Opt", ("vehicle.audi.a2",)),
                "Town05_Opt",
            )
        )

    def test_all_competition_steps_normalize_to_executable_actions(self):
        config = load_config(
            ROOT / "configs" / "scene_2_town05_runtime.json"
        )
        normalized_actions = set()
        for command in config["commands"]:
            intent = build_scheduled_driving_intent(
                command,
                simulation_frame=100,
                route_s_m=float(command["announce_at_m"]),
                timestamp_s=5.0,
            )
            previous_step_id = None
            for step in intent["intent"]["steps"]:
                action, speed, lane, _location = _map_step_action(
                    step,
                    current_speed_kmh=35.0,
                )
                normalized_actions.add(action)
                self.assertGreaterEqual(speed, 0.0)
                self.assertEqual(
                    step["depends_on"],
                    [previous_step_id] if previous_step_id else [],
                )
                if action.startswith("lane_change_"):
                    self.assertIn(lane, {"left", "right"})
                previous_step_id = step["step_id"]

        self.assertTrue(
            {
                "keep_lane",
                "decelerate",
                "stop",
                "accelerate",
                "lane_change_left",
                "lane_change_right",
                "turn_right",
            }.issubset(normalized_actions)
        )

    def test_scene2_speed_and_wait_steps_use_typed_parameters(self):
        command = {
            "id": "typed",
            "category": "NAVIGATION",
            "urgency": "NORMAL",
            "spoken_text": "typed test",
            "steps": [
                "SET_SPEED:12.50mps",
                "CHECK:PATH_CLEAR",
                "WAIT:PEDESTRIAN_CLEAR",
            ],
        }
        intent = build_scheduled_driving_intent(
            command,
            simulation_frame=1,
            route_s_m=0.0,
            timestamp_s=0.0,
        )
        steps = intent["intent"]["steps"]
        self.assertEqual(steps[0]["parameters"], {"target_speed_mps": 12.5})
        self.assertEqual(steps[1]["parameters"], {"condition": "PATH_CLEAR"})
        self.assertEqual(steps[1]["on_blocked"], "WAIT_FOR_SAFE")
        self.assertEqual(steps[2]["on_blocked"], "WAIT_FOR_SAFE")
        self.assertEqual(steps[0]["completion"], {"type": "TARGET_SPEED_REACHED"})
        self.assertEqual(steps[1]["completion"], {"type": "ACTION_REACHED"})
        self.assertEqual(intent["input"]["raw_text"], "typed test")

    def test_scene2_lateral_and_turn_steps_have_runtime_completion(self):
        command = {
            "id": "motion",
            "category": "NAVIGATION",
            "urgency": "NORMAL",
            "spoken_text": "motion test",
            "steps": ["CHANGE_LANE:LEFT", "TURN:RIGHT", "U_TURN:LEGAL"],
        }
        intent = build_scheduled_driving_intent(command, 1, 0.0, 0.0)
        completions = [step["completion"]["type"] for step in intent["intent"]["steps"]]
        self.assertEqual(
            completions,
            ["LANE_CHANGE_COMPLETED", "JUNCTION_EXITED", "JUNCTION_EXITED"],
        )

    def test_scripted_walker_restores_hidden_staging_on_start(self):
        class FakeActor:
            is_alive = True

            def __init__(self):
                self.calls = []

            def set_transform(self, transform):
                self.calls.append(("transform", transform))

            def set_simulate_physics(self, enabled):
                self.calls.append(("physics", enabled))

        actor = FakeActor()
        activation_transform = object()
        walker = ScriptedWalker(
            actor,
            Location(10.0, 0.0),
            1.25,
            activation_transform=activation_transform,
        )
        walker.start()
        self.assertTrue(walker.active)
        self.assertEqual(
            actor.calls,
            [("transform", activation_transform)],
        )

        walker.update()
        self.assertEqual(
            actor.calls,
            [("transform", activation_transform)],
        )

        walker.update()
        self.assertEqual(
            actor.calls,
            [
                ("transform", activation_transform),
                ("physics", True),
                ("transform", activation_transform),
            ],
        )

    def test_command_gate_blocks_later_commands_until_order_is_restored(self):
        commands = [
            {
                "id": "first",
                "announce_at_m": 10.0,
                "requires_event_states": {"pedestrian": "RESOLVED"},
            },
            {"id": "second", "announce_at_m": 20.0},
        ]
        self.assertEqual(
            ready_commands_in_order(
                commands,
                set(),
                100.0,
                {"pedestrian": "ACTIVE"},
            ),
            [],
        )
        self.assertEqual(
            [
                command["id"]
                for command in ready_commands_in_order(
                    commands,
                    set(),
                    100.0,
                    {"pedestrian": "RESOLVED"},
                )
            ],
            ["first", "second"],
        )

    def test_road_option_name_supports_int_enum(self):
        class RoadOption(IntEnum):
            LEFT = 1
            RIGHT = 2

        self.assertEqual(str(RoadOption.RIGHT), "2")
        self.assertEqual(road_option_name(RoadOption.LEFT), "LEFT")
        self.assertEqual(road_option_name(RoadOption.RIGHT), "RIGHT")

    def test_route_aware_preview_speed_caps_turn_window(self):
        maneuvers = [
            {"route_option": "RIGHT", "progress_m": 100.0},
            {"route_option": "STRAIGHT", "progress_m": 200.0},
        ]
        speed = lambda progress: route_aware_preview_speed_kmh(
            progress,
            maneuvers,
            45.0,
            18.0,
            60.0,
            30.0,
        )
        self.assertEqual(speed(39.9), 45.0)
        self.assertEqual(speed(40.0), 18.0)
        self.assertEqual(speed(100.0), 18.0)
        self.assertEqual(speed(130.0), 18.0)
        self.assertEqual(speed(130.1), 45.0)

    def test_planned_turn_window_and_centering_correction(self):
        from types import SimpleNamespace

        maneuvers = [{"route_option": "RIGHT", "progress_m": 100.0}]
        self.assertTrue(planned_turn_window_active(75.0, maneuvers, 25, 45))
        self.assertTrue(planned_turn_window_active(145.0, maneuvers, 25, 45))
        self.assertFalse(planned_turn_window_active(145.1, maneuvers, 25, 45))

        waypoint = SimpleNamespace(
            transform=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            )
        )
        correction, lateral = route_centering_steer_correction(
            SimpleNamespace(x=0.0, y=2.0),
            waypoint,
            0.22,
            0.35,
        )
        self.assertAlmostEqual(lateral, 2.0)
        self.assertAlmostEqual(correction, -0.35)

    def test_hidden_vehicle_staging_is_clear_of_road_actors(self):
        hidden = vehicle_spawn_offsets(True)
        visible = vehicle_spawn_offsets(False)
        self.assertTrue(all(vertical >= 30.0 for _, vertical in hidden))
        self.assertTrue(all(vertical < 1.0 for _, vertical in visible))

    def test_lane_invasion_audit_distinguishes_broken_and_solid(self):
        self.assertFalse(
            lane_invasion_is_restricted({"markings": ["Broken"]})
        )
        self.assertTrue(
            lane_invasion_is_restricted({"markings": ["Solid"]})
        )
        self.assertTrue(
            lane_invasion_is_restricted(
                {"markings": ["LaneMarkingType.Curb"]}
            )
        )

    def test_collision_callbacks_are_grouped_into_physical_events(self):
        monitor = SafetyMonitor(None, None, None)
        vehicle = SimpleNamespace(
            id=17,
            type_id="vehicle.test.suv",
            attributes={"role_name": "scene2_traffic_004"},
        )
        for frame in (100, 101, 102):
            monitor.simulation_time_s = frame / 20.0
            monitor._on_collision(
                SimpleNamespace(frame=frame, other_actor=vehicle)
            )
        self.assertEqual(len(monitor.collision_samples), 3)
        self.assertEqual(len(monitor.collisions), 1)
        self.assertEqual(
            monitor.collisions[0]["contact_samples"],
            3,
        )
        self.assertEqual(
            monitor.collisions[0]["other_actor_role"],
            "scene2_traffic_004",
        )

        monitor._on_collision(
            SimpleNamespace(frame=120, other_actor=vehicle)
        )
        self.assertEqual(len(monitor.collisions), 2)

    def test_runtime_contract_has_required_scope(self):
        config = load_config(
            ROOT / "configs" / "scene_2_town05_runtime.json"
        )
        self.assertEqual(config["map"], "Town05_Opt")
        self.assertEqual(len(config["commands"]), 15)
        self.assertGreaterEqual(
            config["route"]["target_length_m"],
            8000.0,
        )
        self.assertEqual(
            config["route"]["turnaround_spawn_index"],
            109,
        )
        self.assertGreaterEqual(
            config["route"]["minimum_curvature_degrees"],
            180.0,
        )
        self.assertGreaterEqual(config["traffic"]["vehicles"], 55)
        self.assertFalse(config["traffic"]["hybrid_physics"])
        self.assertFalse(
            config["traffic"]["respawn_dormant_vehicles"]
        )
        self.assertFalse(
            config["traffic"]["ambient_auto_lane_change"]
        )
        self.assertGreaterEqual(
            config["traffic"]["ego_spawn_exclusion_m"],
            35.0,
        )
        self.assertGreaterEqual(
            config["traffic"]["startup_corridor_length_m"],
            400.0,
        )
        self.assertGreaterEqual(
            config["traffic"]["startup_corridor_radius_m"],
            10.0,
        )
        self.assertLess(
            config["traffic"]["startup_corridor_length_m"],
            min(
                event["anchor_progress_m"]
                for event in config["special_events"]
            ),
        )
        for event in config["special_events"]:
            self.assertGreaterEqual(len(event["variants"]), 2)
        self.assertEqual(config["weather"]["preset"], "cloudy-evening")
        self.assertLessEqual(
            config["weather"]["sun_altitude_angle"],
            10.0,
        )
        self.assertTrue(
            set(REQUIRED_SENSOR_NAMES).issubset(
                config["sensors"]["required"]
            )
        )
        self.assertFalse(
            config["interfaces"]["allow_adjacent_frame_fill"]
        )
        for command in config["commands"]:
            self.assertEqual(command["text"], command["spoken_text"])
            self.assertGreaterEqual(len(command["steps"]), 2)

    def test_exact_frame_barrier_rejects_stale_rgb(self):
        suite = ExactFrameSensorSuite(
            None,
            None,
            None,
            ROOT / "unused-test-output",
            0.2,
        )
        for name in REQUIRED_SENSOR_NAMES:
            suite._mark_saved(name, 100)
        complete, frames = suite.wait_for_frame(100, 0.0)
        self.assertTrue(complete)
        self.assertEqual(set(frames.values()), {100})

        suite._mark_saved("lidar", 101)
        complete, frames = suite.wait_for_frame(101, 0.0)
        self.assertFalse(complete)
        self.assertEqual(frames["lidar"], 101)
        self.assertEqual(frames["front_rgb"], 100)

    def test_sensor_phase_ignores_irregular_first_interval(self):
        suite = ExactFrameSensorSuite(
            None,
            None,
            None,
            ROOT / "unused-test-output",
            0.2,
        )
        for frame in (100, 105, 109, 113):
            for name in REQUIRED_SENSOR_NAMES:
                suite._mark_saved(name, frame)
        phase, observed = suite.wait_for_stable_phase(
            list(range(100, 114)),
            expected_stride=4,
            timeout_s=0.0,
        )
        self.assertEqual(phase, 113)
        self.assertEqual(observed, [100, 105, 109, 113])

    def test_bundle_requires_exact_world_state_and_all_sensors(self):
        sensor_frames = {
            name: 77 for name in REQUIRED_SENSOR_NAMES
        }
        exact = build_multimodal_frame_bundle(
            "scene", 77, 77, sensor_frames, None
        )
        self.assertEqual(exact["status"], "COMPLETE")
        self.assertTrue(exact["synchronization"]["exact"])

        stale_world = build_multimodal_frame_bundle(
            "scene", 77, 76, sensor_frames, None
        )
        self.assertEqual(stale_world["status"], "INCOMPLETE")
        self.assertFalse(
            stale_world["synchronization"]["world_state_exact"]
        )

    def test_route_command_audit_detects_wrong_turn(self):
        route = [
            (Waypoint(Transform(Location(0.0, 0.0))), "RoadOption.LANEFOLLOW"),
            (Waypoint(Transform(Location(10.0, 0.0))), "RoadOption.RIGHT"),
        ]
        commands = [
            {
                "id": "turn-left",
                "announce_at_m": 0.0,
                "steps": ["TURN:LEFT", "KEEP_LANE"],
            }
        ]
        audit = audit_command_route_alignment(
            commands,
            route,
            [0.0, 10.0],
            default_horizon_m=20.0,
        )
        self.assertEqual(audit["mismatch_count"], 1)
        self.assertFalse(audit["competition_ready"])

    def test_route_command_audit_deduplicates_and_checks_order(self):
        options = (
            "RoadOption.LANEFOLLOW",
            "RoadOption.LEFT",
            "RoadOption.LEFT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.RIGHT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.LEFT",
        )
        route = [
            (Waypoint(Transform(Location(float(index), 0.0))), option)
            for index, option in enumerate(options)
        ]
        command = {
            "id": "left-right-left",
            "announce_at_m": 0.0,
            "steps": ["TURN:LEFT", "TURN:RIGHT", "TURN:LEFT"],
        }
        audit = audit_command_route_alignment(
            [command],
            route,
            [float(index * 10) for index in range(len(route))],
            default_horizon_m=70.0,
        )
        self.assertTrue(audit["competition_ready"])
        self.assertEqual(
            [
                item["route_option"]
                for item in audit["global_maneuvers"]
            ],
            ["LEFT", "RIGHT", "LEFT"],
        )

        command["steps"] = ["TURN:RIGHT", "TURN:LEFT", "TURN:RIGHT"]
        audit = audit_command_route_alignment(
            [command],
            route,
            [float(index * 10) for index in range(len(route))],
            default_horizon_m=70.0,
        )
        self.assertFalse(audit["competition_ready"])

    def test_variants_are_reproducible_and_change_by_episode(self):
        config = load_config(
            ROOT / "configs" / "scene_2_town05_runtime.json"
        )
        first, first_ids = materialize_event_variants(
            config["special_events"],
            0,
        )
        again, again_ids = materialize_event_variants(
            config["special_events"],
            0,
        )
        second, second_ids = materialize_event_variants(
            config["special_events"],
            1,
        )
        self.assertEqual(first, again)
        self.assertEqual(first_ids, again_ids)
        self.assertNotEqual(first_ids, second_ids)
        self.assertNotEqual(first, second)
        self.assertEqual(stable_variant_index("event", 3, 2), 0)

    def test_walker_spawn_retries_cover_offset_and_height(self):
        offsets = walker_spawn_offsets()
        self.assertGreaterEqual(len(offsets), 20)
        self.assertIn((0.0, 0.45), offsets)
        self.assertTrue(any(abs(side) >= 4.0 for side, _ in offsets))
        self.assertTrue(any(height >= 1.0 for _, height in offsets))

    def test_cumulative_route_distances(self):
        route = route_at((0.0, 0.0), (3.0, 4.0), (9.0, 4.0))
        self.assertEqual(
            cumulative_route_distances(route),
            [0.0, 5.0, 11.0],
        )

    def test_progress_does_not_jump_to_repeated_geometry(self):
        route = route_at(
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
        )
        distances = cumulative_route_distances(route)
        tracker = RouteProgressTracker(
            route,
            distances,
            search_ahead=2,
            search_behind=1,
        )
        first = tracker.update(Location(1.0, 0.0))
        second = tracker.update(Location(11.0, 0.0))
        self.assertEqual(first, 0.0)
        self.assertEqual(second, 10.0)
        self.assertLess(tracker.index, 3)


if __name__ == "__main__":
    unittest.main()
