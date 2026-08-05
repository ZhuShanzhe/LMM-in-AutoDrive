import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from xml.etree import ElementTree as ET


CARLA_DIR = Path(__file__).resolve().parents[1]
MAPS_DIR = CARLA_DIR / "maps"
CONFIG_PATH = (
    CARLA_DIR
    / "configs"
    / "scene_3_emergency_6km_runtime.json"
)
XODR_PATH = (
    MAPS_DIR
    / "maps"
    / "output"
    / "VLA_EmergencyRoad_6km.xodr"
)

sys.path.insert(0, str(CARLA_DIR))

import emergency_scene_3_events as scene_events
import run_emergency_response_6km as runner
import scene3_video_preview as video_preview


def load_road_generator():
    path = MAPS_DIR / "generate_emergency_road_xodr.py"
    spec = importlib.util.spec_from_file_location(
        "generate_emergency_road_xodr",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load emergency road generator"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


road_generator = load_road_generator()


class RouteObstacleLaneTests(unittest.TestCase):
    def test_adjacent_lane_actor_does_not_block_ego(self):
        ego = SimpleNamespace(
            road_id=7,
            section_id=0,
            lane_id=-2,
            is_junction=False,
        )
        adjacent = SimpleNamespace(
            road_id=7,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )
        self.assertFalse(
            runner.actor_can_block_ego_lane(ego, adjacent, 2.8)
        )

    def test_same_lane_actor_blocks_ego(self):
        ego = SimpleNamespace(
            road_id=7,
            section_id=0,
            lane_id=-2,
            is_junction=False,
        )
        ahead = SimpleNamespace(
            road_id=7,
            section_id=0,
            lane_id=-2,
            is_junction=False,
        )
        self.assertTrue(
            runner.actor_can_block_ego_lane(ego, ahead, 0.2)
        )

    def test_close_actor_at_road_boundary_uses_geometry(self):
        ego = SimpleNamespace(
            road_id=7,
            section_id=0,
            lane_id=-2,
            is_junction=False,
        )
        ahead = SimpleNamespace(
            road_id=8,
            section_id=0,
            lane_id=-2,
            is_junction=False,
        )
        self.assertTrue(
            runner.actor_can_block_ego_lane(ego, ahead, 0.3)
        )


class RecordingEventHandler:
    def __init__(self):
        self.activated = []
        self.resolved = []
        self.updates = []

    def on_activate(self, event, **context):
        self.activated.append(
            (event["id"], dict(context))
        )

    def update(self, **context):
        self.updates.append(dict(context))

    def on_resolve(self, event, **context):
        self.resolved.append(
            (event["id"], dict(context))
        )


class FakeWaypoint:
    def __init__(
        self,
        route_s_m,
        *,
        road_id=runner.ROAD_ID,
        lane_id=runner.EGO_LANE_ID,
    ):
        self.s = route_s_m
        self.road_id = road_id
        self.lane_id = lane_id


class FakeMap:
    def __init__(self, waypoints):
        self._waypoints = iter(waypoints)

    def get_waypoint(self, *args, **kwargs):
        del args, kwargs
        return next(self._waypoints)


class FakeWorld:
    def __init__(self):
        self.frame = 0

    def tick(self):
        self.frame += 1
        return self.frame


class FakeEgo:
    def get_location(self):
        return object()


class RecordingScheduler:
    def __init__(self):
        self.updates = []

    def update(self, **context):
        self.updates.append(dict(context))


class EmergencyRuntimeConfigTests(unittest.TestCase):
    def test_route_pid_is_default_and_external_control_is_supported(self):
        parser = runner.build_parser()

        self.assertEqual(parser.parse_args([]).ego_controller, "route-pid")
        self.assertEqual(
            parser.parse_args(
                ["--ego-controller", "external"]
            ).ego_controller,
            "external",
        )

    def test_checked_in_config_is_valid(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )

        self.assertEqual(
            config["scene_id"],
            "scene_3_emergency_6km",
        )
        self.assertEqual(len(config["events"]), 7)
        self.assertEqual(
            [event["order"] for event in config["events"]],
            list(range(1, 8)),
        )
        self.assertTrue(
            all(
                event["safety"]["recoverable"]
                for event in config["events"]
            )
        )

    def test_config_declares_exact_actor_counts(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )

        self.assertEqual(
            config["traffic"],
            {
                "private_vehicle_count": 16,
                "work_vehicle_count": 2,
                "maintenance_vehicle_count": 1,
                "worker_count": 2,
                "minimum_ego_spawn_clearance_m": 35.0,
            },
        )
        self.assertEqual(
            len(scene_events.BACKGROUND_TRAFFIC_PLAN),
            14,
        )

    def test_config_requires_four_rgb_views(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )

        self.assertEqual(
            config["sensors"][
                "required_camera_names"
            ],
            [
                "front_rgb",
                "left_rgb",
                "right_rgb",
                "rear_rgb",
            ],
        )
        self.assertEqual(
            {
                name: transform[3]
                for name, transform
                in runner.CAMERA_TRANSFORMS.items()
            },
            {
                "front_rgb": 0.0,
                "left_rgb": -90.0,
                "right_rgb": 90.0,
                "rear_rgb": 180.0,
            },
        )

    def test_chase_camera_mode_is_independent(
        self,
    ):
        chase = runner.camera_transforms_for_mode(
            "chase-only"
        )
        combined = (
            runner.camera_transforms_for_mode(
                "four-view-plus-chase"
            )
        )

        self.assertEqual(
            list(chase),
            ["chase_rgb"],
        )
        self.assertEqual(
            set(combined),
            {
                *runner.CAMERA_TRANSFORMS,
                "chase_rgb",
            },
        )
        self.assertEqual(
            len(
                runner.CHASE_CAMERA_TRANSFORMS[
                    "chase_rgb"
                ]
            ),
            5,
        )
        self.assertEqual(
            runner.CHASE_CAMERA_TRANSFORMS[
                "chase_rgb"
            ],
            (
                -5.5,
                0.0,
                2.8,
                -15.0,
                0.0,
            ),
        )

    def test_daylight_profile_requires_chase_camera(
        self,
    ):
        args = runner.build_parser().parse_args(
            [
                "--presentation-lighting",
                "rainy-daylight",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires chase-only",
        ):
            runner.validate_args(args)

        args.camera_mode = "chase-only"
        runner.validate_args(args)

    def test_presentation_weather_is_visible(
        self,
    ):
        self.assertGreater(
            runner.PRESENTATION_WEATHER[
                "sun_altitude_angle"
            ],
            0.0,
        )
        self.assertLess(
            runner.PRESENTATION_WEATHER[
                "fog_density"
            ],
            35.0,
        )
        self.assertEqual(
            runner.CLEAR_PRESENTATION_WEATHER[
                "precipitation"
            ],
            0.0,
        )
        self.assertEqual(
            runner.CLEAR_PRESENTATION_WEATHER[
                "fog_density"
            ],
            0.0,
        )
        self.assertGreaterEqual(
            runner.CLEAR_PRESENTATION_WEATHER[
                "sun_altitude_angle"
            ],
            60.0,
        )

    def test_clear_daylight_requires_chase_camera(
        self,
    ):
        args = runner.build_parser().parse_args(
            [
                "--presentation-lighting",
                "clear-daylight",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires chase-only",
        ):
            runner.validate_args(args)

        args.camera_mode = "chase-only"
        runner.validate_args(args)

    def test_chase_camera_reduces_daylight_exposure(
        self,
    ):
        chase = runner.camera_sensor_attributes(
            camera_name="chase_rgb",
            image_width=960,
            image_height=540,
            fov=90.0,
            camera_tick=0.2,
        )
        front = runner.camera_sensor_attributes(
            camera_name="front_rgb",
            image_width=960,
            image_height=540,
            fov=90.0,
            camera_tick=0.2,
        )

        self.assertEqual(chase["gamma"], "2.2")
        self.assertEqual(
            chase["exposure_mode"],
            "manual",
        )
        self.assertEqual(
            chase["exposure_compensation"],
            "0.0",
        )
        self.assertEqual(
            chase["shutter_speed"],
            "200.0",
        )
        self.assertEqual(
            chase["iso"],
            "100.0",
        )
        self.assertEqual(
            chase["fstop"],
            "8.0",
        )
        self.assertEqual(front["gamma"], "3.0")
        self.assertEqual(
            front["exposure_compensation"],
            "3.0",
        )

    def test_camera_frames_are_published_atomically(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            capture_state = runner.CaptureState(
                ("chase_rgb",)
            )
            callback = runner.make_camera_callback(
                camera_name="chase_rgb",
                output_dir=output_dir,
                capture_state=capture_state,
            )
            image = SimpleNamespace(
                frame=123,
                save_to_disk=lambda path: (
                    Path(path).write_bytes(b"PNG")
                ),
            )

            callback(image)

            published = (
                output_dir
                / "rgb"
                / "chase_rgb"
                / "00000123.png"
            )
            pending = (
                output_dir
                / "rgb"
                / "chase_rgb"
                / ".pending"
                / "00000123.png"
            )
            self.assertEqual(
                published.read_bytes(),
                b"PNG",
            )
            self.assertFalse(pending.exists())
            self.assertEqual(
                capture_state.snapshot(),
                {"chase_rgb": 1},
            )

    def test_presentation_lane_markings_cover_route(
        self,
    ):
        debug = mock.Mock()
        world = SimpleNamespace(debug=debug)
        fake_carla = SimpleNamespace(
            Location=lambda **values: values,
            Color=lambda *values: values,
        )

        with mock.patch.object(
            runner,
            "carla",
            fake_carla,
        ):
            count = (
                runner.draw_presentation_lane_markings(
                    world,
                    road_length_m=25.0,
                    life_time_s=300.0,
                )
            )

        # Three dashes on each lane divider plus center and edge.
        self.assertEqual(count, 8)
        self.assertEqual(
            debug.draw_line.call_count,
            8,
        )
        first_call = debug.draw_line.call_args_list[
            0
        ]
        self.assertEqual(
            first_call.args[0]["y"],
            3.5,
        )
        self.assertEqual(
            first_call.args[1]["x"],
            4.0,
        )
        self.assertEqual(
            first_call.kwargs["thickness"],
            0.045,
        )
        self.assertEqual(
            first_call.kwargs["life_time"],
            0.0,
        )

    def test_rejects_duplicate_event_id(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )
        invalid = deepcopy(config)
        invalid["events"][1]["id"] = (
            invalid["events"][0]["id"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(
                json.dumps(invalid),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "duplicate id",
            ):
                runner.load_runtime_config(path)

    def test_rejects_unrecoverable_event(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )
        invalid = deepcopy(config)
        invalid["events"][0]["safety"][
            "recoverable"
        ] = False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(
                json.dumps(invalid),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "must be recoverable",
            ):
                runner.load_runtime_config(path)


class EmergencyRoadContractTests(unittest.TestCase):
    def test_behavior_agent_plan_preserves_junction_topology(
        self,
    ):
        locations = [object() for _ in range(4)]
        route_context = SimpleNamespace(
            route=[
                (
                    SimpleNamespace(
                        transform=SimpleNamespace(location=location),
                        is_junction=True,
                    ),
                    None,
                )
                for location in locations
            ],
            distances_m=[0.0, 2.0, 5.0, 10.0],
        )
        plan = runner.build_ego_route_plan(
            route_context,
            runner.load_runtime_config(CONFIG_PATH)["events"],
        )

        self.assertEqual(
            [waypoint.transform.location for waypoint, _ in plan],
            locations,
        )

    def test_checked_in_xodr_is_valid(self):
        root = ET.parse(XODR_PATH).getroot()

        self.assertEqual(
            road_generator.validate_opendrive(root),
            [],
        )
        self.assertEqual(root.findall("junction"), [])

    def test_physical_road_has_terminal_runoff(self):
        root = ET.parse(XODR_PATH).getroot()
        road = root.find("road")
        self.assertIsNotNone(road)
        assert road is not None
        physical_length_m = float(
            road.get("length", "nan")
        )
        config = runner.load_runtime_config(
            CONFIG_PATH
        )

        self.assertEqual(
            config["map"]["length_m"],
            road_generator.SCENE_LENGTH_M,
        )
        self.assertEqual(
            physical_length_m,
            (
                road_generator.SCENE_LENGTH_M
                + road_generator.RUNOFF_LENGTH_M
            ),
        )
        self.assertGreaterEqual(
            physical_length_m
            - float(
                config["map"]["finish_s_m"]
            ),
            100.0,
        )

    def test_xodr_is_reproducible(self):
        root = road_generator.build_opendrive()
        self.assertEqual(
            road_generator.validate_opendrive(root),
            [],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            generated = (
                Path(temp_dir)
                / "VLA_EmergencyRoad_6km.xodr"
            )
            road_generator.write_opendrive(
                root,
                generated,
            )
            self.assertEqual(
                generated.read_text(encoding="utf-8"),
                XODR_PATH.read_text(encoding="utf-8"),
            )


class Scene3VideoPreviewTests(unittest.TestCase):
    def test_route_progress_is_interpolated(self):
        frames, routes = (
            video_preview.route_anchors(
                [100, 300],
                [
                    {
                        "simulation_frame": 200,
                        "route_s_m": 3000.0,
                    }
                ],
                route_completed=True,
            )
        )

        self.assertEqual(
            video_preview.interpolate_route_s(
                200,
                frames,
                routes,
            ),
            3000.0,
        )
        self.assertEqual(routes[-1], 5990.0)

    def test_hud_tracks_active_event(self):
        timeline = [
            {
                "simulation_frame": 100,
                "event_id": "scene3_cut_in",
                "state": "ACTIVE",
            },
            {
                "simulation_frame": 200,
                "event_id": "scene3_cut_in",
                "state": "RESOLVED",
            },
        ]

        self.assertEqual(
            video_preview.active_event_label(
                150,
                timeline,
            ),
            "CUT-IN VEHICLE",
        )
        self.assertEqual(
            video_preview.active_event_label(
                250,
                timeline,
            ),
            "ALL EVENTS RESOLVED",
        )


class EmergencyEventSchedulerTests(unittest.TestCase):
    def test_all_seven_events_activate_and_resolve(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )
        handler = RecordingEventHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = (
                Path(temp_dir) / "timeline.jsonl"
            )
            scheduler = runner.EmergencyEventScheduler(
                config["events"],
                output_path=timeline,
                event_handler=handler,
            )
            scheduler.update(
                route_s_m=6000.0,
                simulation_frame=10,
                elapsed_s=120.0,
            )

            self.assertEqual(
                scheduler.summary(),
                {
                    "PENDING": 0,
                    "ACTIVE": 0,
                    "RESOLVED": 7,
                },
            )
            self.assertEqual(len(handler.activated), 7)
            self.assertEqual(len(handler.resolved), 7)
            records = [
                json.loads(line)
                for line in timeline.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(records), 14)


class EmergencyActorRuntimeTests(unittest.TestCase):
    def test_background_traffic_falls_back_to_existing_lane(
        self,
    ):
        available = {
            (-2, s_m)
            for _, s_m, _ in (
                scene_events.BACKGROUND_TRAFFIC_PLAN
            )
        }
        carla_map = mock.Mock()
        carla_map.get_waypoint_xodr.side_effect = (
            lambda _road_id, lane_id, s_m: (
                SimpleNamespace(
                    transform=SimpleNamespace(
                        location=SimpleNamespace(z=0.0)
                    )
                )
                if (lane_id, s_m) in available
                else None
            )
        )
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=mock.Mock(),
                world=mock.Mock(),
                carla_map=carla_map,
                traffic_manager=mock.Mock(),
                traffic_manager_port=8000,
                actor_sink=[],
            )
        )
        runtime._spawn_moving_vehicle = mock.Mock(
            side_effect=lambda **values: values[
                "actor_config"
            ]
        )

        runtime.spawn_background_traffic(
            {"private_vehicle_count": 16}
        )
        for _lane_id, s_m, _speed_kmh in (
            scene_events.BACKGROUND_TRAFFIC_PLAN
        ):
            runtime._update_background_traffic(s_m + 80.0)

        actual_lane_ids = [
            call.kwargs["actor_config"]["lane_id"]
            for call in (
                runtime._spawn_moving_vehicle.call_args_list
            )
        ]
        self.assertEqual(
            actual_lane_ids,
            [-2] * len(scene_events.BACKGROUND_TRAFFIC_PLAN),
        )

    def test_background_traffic_does_not_occupy_initial_route(self):
        first_background_s_m = min(
            s_m
            for _lane_id, s_m, _speed_kmh in (
                scene_events.BACKGROUND_TRAFFIC_PLAN
            )
        )
        self.assertGreater(first_background_s_m, 1550.0)

    def test_worker_spawn_retries_equivalent_pose(
        self,
    ):
        worker = mock.Mock()
        blueprints = [
            mock.Mock(id="walker.pedestrian.0001"),
            mock.Mock(id="walker.pedestrian.0002"),
        ]
        for blueprint in blueprints:
            blueprint.has_attribute.return_value = (
                True
            )
        world = mock.Mock()
        world.try_spawn_actor.side_effect = [
            None,
            worker,
        ]
        base = SimpleNamespace(
            location=SimpleNamespace(
                x=3275.0,
                y=12.0,
                z=0.0,
            ),
            rotation=SimpleNamespace(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
            ),
        )
        carla_map = mock.Mock()
        carla_map.get_waypoint_xodr.return_value = (
            SimpleNamespace(transform=base)
        )
        fake_carla = SimpleNamespace(
            Location=lambda **values: SimpleNamespace(
                **values
            ),
            Rotation=lambda **values: SimpleNamespace(
                **values
            ),
            Transform=lambda location, rotation: (
                SimpleNamespace(
                    location=location,
                    rotation=rotation,
                )
            ),
        )
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=fake_carla,
                world=world,
                carla_map=carla_map,
                traffic_manager=mock.Mock(),
                traffic_manager_port=8000,
                actor_sink=[],
            )
        )

        result = runtime._spawn_work_zone_worker(
            worker_config={
                "role_name": "scene3_crossing_worker",
                "start_lane_id": -4,
                "start_s_m": 3275.0,
            },
            blueprints=blueprints,
        )

        self.assertIs(result, worker)
        self.assertEqual(
            world.try_spawn_actor.call_count,
            2,
        )

    def test_crossing_worker_spawns_at_trigger_only(
        self,
    ):
        worker = mock.Mock()
        worker.get_location.return_value = (
            SimpleNamespace(
                x=3275.0,
                y=12.0,
                z=0.3,
            )
        )
        fake_carla = SimpleNamespace(
            WalkerControl=lambda **values: (
                SimpleNamespace(**values)
            ),
            Vector3D=lambda **values: (
                SimpleNamespace(**values)
            ),
            Location=lambda **values: (
                SimpleNamespace(**values)
            ),
        )
        actor_sink = []
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=fake_carla,
                world=mock.Mock(),
                carla_map=mock.Mock(),
                traffic_manager=mock.Mock(),
                traffic_manager_port=8000,
                actor_sink=actor_sink,
            )
        )
        runtime._worker_phase = "ARMED"
        runtime._crossing_worker_config = {
            "role_name": "scene3_crossing_worker",
            "start_lane_id": -4,
            "start_s_m": 3275.0,
        }
        runtime._crossing_worker_target_y = 8.75
        runtime._walker_blueprint_candidates = [
            mock.Mock()
        ]
        runtime._spawn_work_zone_worker = mock.Mock(
            return_value=worker
        )

        runtime._update_worker_crossing(
            ego_route_s_m=3199.0,
            elapsed_s=0.0,
        )
        runtime._spawn_work_zone_worker.assert_not_called()

        runtime._update_worker_crossing(
            ego_route_s_m=3200.0,
            elapsed_s=1.0,
        )

        runtime._spawn_work_zone_worker.assert_called_once()
        worker.apply_control.assert_called_once()
        self.assertEqual(
            runtime._worker_phase,
            "CROSSING",
        )
        self.assertEqual(actor_sink, [worker])

        runtime._update_worker_crossing(
            ego_route_s_m=3235.0,
            elapsed_s=5.0,
        )

        worker.set_location.assert_called()
        self.assertEqual(
            runtime._worker_phase,
            "YIELDED_CLEAR",
        )

    def test_crossing_worker_recovers_once_if_actor_is_retired(self):
        retired = mock.Mock()
        retired.is_alive = False
        recovered = mock.Mock()
        recovered.is_alive = True
        recovered.get_location.return_value = SimpleNamespace(
            x=3275.0,
            y=12.0,
            z=0.3,
        )
        fake_carla = SimpleNamespace(
            WalkerControl=lambda **values: SimpleNamespace(**values),
            Vector3D=lambda **values: SimpleNamespace(**values),
            Location=lambda **values: SimpleNamespace(**values),
        )
        actor_sink = []
        runtime = scene_events.EmergencySceneActorRuntime(
            carla_module=fake_carla,
            world=mock.Mock(),
            carla_map=mock.Mock(),
            traffic_manager=mock.Mock(),
            traffic_manager_port=8000,
            actor_sink=actor_sink,
        )
        runtime._worker_phase = "CROSSING"
        runtime._crossing_worker = retired
        runtime._crossing_worker_config = {
            "role_name": "scene3_crossing_worker",
            "start_lane_id": -4,
            "start_s_m": 3275.0,
        }
        runtime._walker_blueprint_candidates = [mock.Mock()]
        runtime._crossing_worker_start_location = SimpleNamespace(
            x=3275.0,
            y=12.0,
            z=0.3,
        )
        runtime._crossing_worker_target_location = SimpleNamespace(
            x=3275.0,
            y=8.75,
            z=0.3,
        )
        runtime._crossing_worker_start_elapsed_s = 1.0
        runtime._spawn_work_zone_worker = mock.Mock(return_value=recovered)

        runtime._update_worker_crossing(
            ego_route_s_m=3230.0,
            elapsed_s=2.0,
        )

        runtime._spawn_work_zone_worker.assert_called_once()
        self.assertIs(runtime._crossing_worker, recovered)
        self.assertEqual(runtime._crossing_worker_respawn_count, 1)
        self.assertEqual(actor_sink, [recovered])
        recovered.set_location.assert_called_once()

    def test_ego_starts_with_automatic_lane_changes_disabled(
        self,
    ):
        ego = mock.Mock()
        blueprint = mock.Mock()
        blueprint.has_attribute.return_value = True
        library = mock.Mock()
        library.find.return_value = blueprint
        world = mock.Mock()
        world.get_blueprint_library.return_value = (
            library
        )
        world.try_spawn_actor.return_value = ego
        carla_map = mock.Mock()
        carla_map.get_waypoint_xodr.return_value = (
            SimpleNamespace(
                transform=SimpleNamespace(
                    location=SimpleNamespace(z=0.0)
                )
            )
        )
        traffic_manager = mock.Mock()
        fake_carla = SimpleNamespace(
            VehicleLightState=mock.Mock(
                Position=1,
                LowBeam=2,
                Fog=4,
            ),
            VehicleControl=mock.Mock(),
        )

        with mock.patch.object(
            runner,
            "carla",
            fake_carla,
        ):
            result = runner.spawn_ego(
                world,
                carla_map,
                traffic_manager=traffic_manager,
                traffic_manager_port=8000,
                target_speed_kmh=40.0,
                stationary=False,
                lights_enabled=False,
            )

        self.assertIs(result, ego)
        traffic_manager.auto_lane_change.assert_called_once_with(
            ego,
            False,
        )
        traffic_manager.update_vehicle_lights.assert_called_once_with(
            ego,
            False,
        )

    def test_daylight_event_vehicles_keep_lights_off(
        self,
    ):
        traffic_manager = mock.Mock()
        fake_carla = SimpleNamespace(
            VehicleLightState=mock.Mock(
                Position=1,
                LowBeam=2,
                Fog=4,
            ),
        )
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=fake_carla,
                world=mock.Mock(),
                carla_map=mock.Mock(),
                traffic_manager=traffic_manager,
                traffic_manager_port=8000,
                actor_sink=[],
                lights_enabled=False,
            )
        )
        actor = mock.Mock()

        runtime._set_vehicle_lights(
            actor,
            traffic_manager_controlled=True,
        )

        traffic_manager.update_vehicle_lights.assert_called_once_with(
            actor,
            False,
        )
        fake_carla.VehicleLightState.assert_called_once_with(
            0,
        )

    def test_cut_in_actor_spawns_only_at_trigger(self):
        config = runner.load_runtime_config(
            CONFIG_PATH
        )
        event = config["events"][0]
        actor = mock.Mock()
        actor.id = 46
        actor.is_alive = True
        actor.get_location.return_value = object()

        blueprint = mock.Mock()
        blueprint.has_attribute.return_value = True
        library = mock.Mock()
        library.find.return_value = blueprint
        world = mock.Mock()
        world.get_blueprint_library.return_value = (
            library
        )
        world.try_spawn_actor.return_value = actor

        transform = SimpleNamespace(
            location=SimpleNamespace(z=0.0)
        )
        carla_map = mock.Mock()
        carla_map.get_waypoint_xodr.return_value = (
            SimpleNamespace(transform=transform)
        )
        carla_map.get_waypoint.return_value = (
            SimpleNamespace(
                s=event["actor"]["spawn_s_m"],
                lane_id=-1,
            )
        )
        fake_carla = SimpleNamespace(
            VehicleLightState=mock.Mock(
                Position=1,
                LowBeam=2,
                Fog=4,
            ),
            VehicleControl=mock.Mock(),
            LaneType=SimpleNamespace(
                Driving="driving"
            ),
        )
        traffic_manager = mock.Mock()
        actor_sink = []
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=fake_carla,
                world=world,
                carla_map=carla_map,
                traffic_manager=traffic_manager,
                traffic_manager_port=8000,
                actor_sink=actor_sink,
            )
        )

        runtime.on_activate(
            event,
            route_s_m=event["activate_at_m"],
            simulation_frame=1,
            elapsed_s=1.0,
        )
        world.try_spawn_actor.assert_not_called()
        self.assertEqual(actor_sink, [])

        runtime.update(
            route_s_m=event["distance_m"],
            simulation_frame=2,
            elapsed_s=2.0,
        )
        world.try_spawn_actor.assert_called_once()
        actor.set_simulate_physics.assert_not_called()
        actor.set_autopilot.assert_called_once_with(
            True,
            8000,
        )
        self.assertEqual(actor_sink, [actor])

    def test_resolved_dynamic_actors_are_not_polled(
        self,
    ):
        cut_in_actor = mock.Mock()
        crossing_worker = mock.Mock()
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=SimpleNamespace(),
                world=mock.Mock(),
                carla_map=mock.Mock(),
                traffic_manager=mock.Mock(),
                traffic_manager_port=8000,
                actor_sink=[],
            )
        )
        runtime._cut_in_event = {
            "id": "scene3_cut_in"
        }
        runtime._cut_in_actor = cut_in_actor
        runtime._cut_in_phase = "RESOLVED"
        runtime._crossing_worker = crossing_worker
        runtime._crossing_worker_target_y = 8.75
        runtime._worker_phase = "RESOLVED"

        runtime.update(
            route_s_m=5500.0,
            simulation_frame=1,
            elapsed_s=600.0,
        )

        cut_in_actor.get_location.assert_not_called()
        crossing_worker.get_location.assert_not_called()

    def test_blocked_lane_uses_gap_checked_left_change(
        self,
    ):
        ego = mock.Mock()
        front = mock.Mock()
        rear = mock.Mock()
        front.get_location.return_value = object()
        rear.get_location.return_value = object()
        carla_map = mock.Mock()
        carla_map.get_waypoint.side_effect = [
            SimpleNamespace(s=4250.0),
            SimpleNamespace(s=4160.0),
        ]
        traffic_manager = mock.Mock()
        runtime = (
            scene_events.EmergencySceneActorRuntime(
                carla_module=SimpleNamespace(
                    LaneType=SimpleNamespace(
                        Driving="driving"
                    )
                ),
                world=mock.Mock(),
                carla_map=carla_map,
                traffic_manager=traffic_manager,
                traffic_manager_port=8000,
                actor_sink=[],
                ego_actor=ego,
            )
        )
        runtime._maintenance_vehicle = mock.Mock()
        runtime._blocked_lane_event = {
            "safety": {
                "minimum_front_gap_m": 30.0,
                "minimum_rear_gap_m": 25.0,
            }
        }
        runtime._blocked_lane_activation_s = 1.0
        runtime._gap_release_commanded = True
        runtime._gap_control_vehicles = {
            "front": front,
            "rear": rear,
        }

        runtime._update_blocked_lane(
            ego_route_s_m=4200.0,
            elapsed_s=10.0,
        )

        traffic_manager.force_lane_change.assert_called_once_with(
            ego,
            False,
        )
        self.assertTrue(
            runtime._target_lane_released
        )


class SafetyAuditTests(unittest.TestCase):
    def test_records_collisions_and_lane_use(self):
        audit = runner.SafetyAuditState()

        audit.record_collision(
            SimpleNamespace(frame=12)
        )
        audit.record_lane_invasion(
            SimpleNamespace(frame=18)
        )
        for lane_id in (-1, -2, -3, -4, 0):
            audit.record_lane_id(lane_id)

        self.assertEqual(
            audit.snapshot(),
            {
                "collision_count": 1,
                "collision_frames": [12],
                "lane_invasion_event_count": 1,
                "lane_invasion_frames": [18],
                "invalid_lane_samples": 2,
            },
        )

    def test_route_completion_uses_measured_waypoint(self):
        scheduler = RecordingScheduler()
        audit = runner.SafetyAuditState()
        fake_carla = SimpleNamespace(
            LaneType=SimpleNamespace(
                Driving="driving"
            )
        )

        with (
            mock.patch.object(
                runner,
                "carla",
                fake_carla,
            ),
            mock.patch.object(
                runner,
                "set_spectator_view",
            ),
        ):
            completed = runner.run_simulation(
                world=FakeWorld(),
                carla_map=FakeMap(
                    [
                        FakeWaypoint(60.0),
                        FakeWaypoint(95.0),
                        FakeWaypoint(120.0),
                    ]
                ),
                ego=FakeEgo(),
                scheduler=scheduler,
                safety_audit=audit,
                finish_s_m=100.0,
                duration_s=1.0,
                fixed_delta_seconds=0.05,
            )

        self.assertTrue(completed)
        self.assertEqual(len(scheduler.updates), 3)
        self.assertEqual(
            scheduler.updates[-1]["route_s_m"],
            120.0,
        )
        self.assertEqual(
            audit.snapshot()["invalid_lane_samples"],
            0,
        )

    def test_route_recovers_stale_ego_handle(self):
        scheduler = RecordingScheduler()
        audit = runner.SafetyAuditState()
        stale_ego = mock.Mock()
        stale_ego.id = 7
        stale_ego.get_location.side_effect = (
            RuntimeError("destroyed actor proxy")
        )
        refreshed_ego = mock.Mock()
        refreshed_ego.id = 7
        refreshed_ego.get_location.return_value = (
            object()
        )
        world = FakeWorld()
        world.get_actor = mock.Mock(
            return_value=refreshed_ego
        )
        fake_carla = SimpleNamespace(
            LaneType=SimpleNamespace(
                Driving="driving"
            )
        )

        with (
            mock.patch.object(
                runner,
                "carla",
                fake_carla,
            ),
            mock.patch.object(
                runner,
                "set_spectator_view",
            ),
        ):
            completed = runner.run_simulation(
                world=world,
                carla_map=FakeMap(
                    [FakeWaypoint(120.0)]
                ),
                ego=stale_ego,
                scheduler=scheduler,
                safety_audit=audit,
                finish_s_m=100.0,
                duration_s=1.0,
                fixed_delta_seconds=0.05,
            )

        self.assertTrue(completed)
        world.get_actor.assert_called_once_with(7)
        refreshed_ego.get_location.assert_called_once()

    def test_spectator_failure_does_not_stop_route(
        self,
    ):
        scheduler = RecordingScheduler()
        audit = runner.SafetyAuditState()
        fake_carla = SimpleNamespace(
            LaneType=SimpleNamespace(
                Driving="driving"
            )
        )

        with (
            mock.patch.object(
                runner,
                "carla",
                fake_carla,
            ),
            mock.patch.object(
                runner,
                "set_spectator_view",
                side_effect=RuntimeError(
                    "destroyed spectator"
                ),
            ),
        ):
            completed = runner.run_simulation(
                world=FakeWorld(),
                carla_map=FakeMap(
                    [FakeWaypoint(120.0)]
                ),
                ego=FakeEgo(),
                scheduler=scheduler,
                safety_audit=audit,
                finish_s_m=100.0,
                duration_s=1.0,
                fixed_delta_seconds=0.05,
            )

        self.assertTrue(completed)

    def test_rejects_route_departure(self):
        scheduler = RecordingScheduler()
        audit = runner.SafetyAuditState()
        fake_carla = SimpleNamespace(
            LaneType=SimpleNamespace(
                Driving="driving"
            )
        )

        with (
            mock.patch.object(
                runner,
                "carla",
                fake_carla,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "left the emergency route",
            ),
        ):
            runner.run_simulation(
                world=FakeWorld(),
                carla_map=FakeMap(
                    [
                        FakeWaypoint(
                            100.0,
                            road_id=99,
                        )
                    ]
                ),
                ego=FakeEgo(),
                scheduler=scheduler,
                safety_audit=audit,
                finish_s_m=5990.0,
                duration_s=0.05,
                fixed_delta_seconds=0.05,
            )


class DirectVideoRecordingTests(unittest.TestCase):
    def test_direct_presentation_camera_uses_manual_exposure(
        self,
    ):
        attributes = (
            runner.DIRECT_PRESENTATION_CAMERA_ATTRIBUTES
        )

        self.assertEqual(
            attributes["exposure_mode"],
            "manual",
        )
        self.assertEqual(
            attributes["exposure_compensation"],
            "0.0",
        )
        self.assertEqual(
            attributes["iso"],
            "100.0",
        )
        self.assertEqual(
            attributes["fstop"],
            "5.6",
        )

    def test_direct_video_matches_control_runner_cli(
        self,
    ):
        args = runner.build_parser().parse_args(
            [
                "--presentation-lighting",
                "clear-daylight",
                "--video-output",
                "scene3.mp4",
                "--video-overlay",
                "--camera-width",
                "1920",
                "--camera-height",
                "1080",
                "--video-fps",
                "30",
            ]
        )

        runner.validate_args(args)
        self.assertEqual(
            args.video_output,
            Path("scene3.mp4"),
        )
        self.assertTrue(args.video_overlay)
        self.assertFalse(
            args.draw_presentation_lane_markings
        )

    def test_scene_event_is_mapped_to_hud(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = runner.EmergencyEventScheduler(
                [
                    {
                        "id": "scene3_cut_in",
                        "scenario": "cut_in_vehicle",
                        "activate_at_m": 10.0,
                        "resolve_after_m": 100.0,
                    }
                ],
                output_path=(
                    Path(temp_dir)
                    / "timeline.jsonl"
                ),
            )
            scheduler.update(
                route_s_m=20.0,
                simulation_frame=7,
                elapsed_s=0.35,
            )
            ego = SimpleNamespace(
                get_velocity=lambda: SimpleNamespace(
                    x=10.0,
                    y=0.0,
                    z=0.0,
                ),
                get_control=lambda: SimpleNamespace(
                    throttle=0.25,
                    brake=0.0,
                    steer=-0.1,
                ),
            )

            overlay = runner.make_video_overlay(
                ego=ego,
                scheduler=scheduler,
                safety_audit=(
                    runner.SafetyAuditState()
                ),
                route_s_m=20.0,
                finish_s_m=5990.0,
                simulation_frame=7,
                elapsed_s=0.35,
                cruise_speed_kmh=40.0,
            )

        self.assertEqual(
            overlay["policy_state"],
            "CUT_IN_RESPONSE",
        )
        self.assertEqual(
            overlay["risk_level"],
            "HIGH",
        )
        self.assertEqual(
            overlay["action"],
            "decelerate",
        )
        self.assertAlmostEqual(
            overlay["speed_kmh"],
            36.0,
        )

    def test_simulation_submits_synchronized_frame(
        self,
    ):
        class VideoCamera:
            def __init__(self):
                self.calls = []

            def save_frame(self, frame, overlay):
                self.calls.append((frame, overlay))
                return True

        class VideoEgo(FakeEgo):
            def get_velocity(self):
                return SimpleNamespace(
                    x=5.0,
                    y=0.0,
                    z=0.0,
                )

            def get_control(self):
                return SimpleNamespace(
                    throttle=0.3,
                    brake=0.0,
                    steer=0.0,
                )

        camera = VideoCamera()
        audit = runner.SafetyAuditState()
        scheduler = runner.EmergencyEventScheduler(
            [],
            output_path=Path(
                tempfile.gettempdir()
            )
            / "scene3_video_test_timeline.jsonl",
        )
        fake_carla = SimpleNamespace(
            LaneType=SimpleNamespace(
                Driving="driving"
            )
        )

        with (
            mock.patch.object(
                runner,
                "carla",
                fake_carla,
            ),
            mock.patch.object(
                runner,
                "set_spectator_view",
            ),
        ):
            completed = runner.run_simulation(
                world=FakeWorld(),
                carla_map=FakeMap(
                    [FakeWaypoint(120.0)]
                ),
                ego=VideoEgo(),
                scheduler=scheduler,
                safety_audit=audit,
                finish_s_m=100.0,
                duration_s=1.0,
                fixed_delta_seconds=0.05,
                video_camera=camera,
            )

        self.assertTrue(completed)
        self.assertEqual(len(camera.calls), 1)
        self.assertEqual(camera.calls[0][0], 1)
        self.assertEqual(
            camera.calls[0][1]["route_s_m"],
            120.0,
        )


if __name__ == "__main__":
    unittest.main()
