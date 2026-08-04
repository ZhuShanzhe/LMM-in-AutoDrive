from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


CARLA_DIR = Path(__file__).resolve().parents[1]
CONFIG = CARLA_DIR / "configs" / "scene_3_emergency_6km_runtime.json"

import sys

if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

import run_emergency_response_6km as runner
import scene3_video_preview as preview
from scene3_town05_route import Town05RouteMapAdapter


class Location:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class Waypoint:
    def __init__(self, lane_id: int, lane_type: str, x: float, y: float):
        self.lane_id = lane_id
        self.road_id = 7
        self.lane_type = lane_type
        self.transform = SimpleNamespace(
            location=Location(x, y),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        )
        self._left = None
        self._right = None

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


def lane_family(x: float = 0.0):
    left = Waypoint(-1, "Driving", x, -3.5)
    centre = Waypoint(-2, "Driving", x, 0.0)
    right = Waypoint(-3, "Driving", x, 3.5)
    shoulder = Waypoint(-4, "Shoulder", x, 6.5)
    sidewalk = Waypoint(-5, "Sidewalk", x, 8.0)
    left._right = centre
    centre._left = left
    centre._right = right
    right._left = centre
    right._right = shoulder
    shoulder._left = right
    shoulder._right = sidewalk
    sidewalk._left = shoulder
    return left, centre, right, shoulder, sidewalk


class Town05Scene3ContractTests(unittest.TestCase):
    def test_config_uses_official_map_and_exact_six_km(self):
        config = runner.load_runtime_config(CONFIG)
        self.assertEqual(config["map"]["name"], "Town05_Opt")
        self.assertTrue(config["map"]["official_carla_asset"])
        self.assertNotIn("xodr", config["map"])
        self.assertEqual(config["map"]["route"]["target_length_m"], 6000.0)
        self.assertEqual(config["map"]["finish_progress_m"], 6000.0)

    def test_required_environment_parameters_are_active_contracts(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        weather = config["weather"]
        self.assertGreater(weather["precipitation"], 0.0)
        self.assertLess(weather["sun_altitude_angle"], 0.0)
        self.assertGreater(weather["fog_density"], 0.0)
        self.assertEqual(weather["wetness"], 100.0)
        self.assertTrue(
            config["surface_and_visibility"]["friction_triggers"]["enabled"]
        )
        self.assertLess(
            config["surface_and_visibility"]["friction_triggers"]["friction"],
            1.0,
        )
        self.assertTrue(config["sensors"]["low_signal_rgb"]["enabled"])

    def test_logical_lanes_map_to_official_waypoints(self):
        left, centre, right, shoulder, sidewalk = lane_family()
        official_map = mock.Mock()
        adapter = Town05RouteMapAdapter(
            official_map,
            [(centre, None), (centre, None)],
            [0.0, 6000.0],
        )
        self.assertIs(adapter.logical_waypoint(-1, 1000.0), left)
        self.assertIs(adapter.logical_waypoint(-2, 1000.0), centre)
        self.assertIs(adapter.logical_waypoint(-3, 1000.0), right)
        self.assertIs(adapter.logical_waypoint(-4, 1000.0), shoulder)
        self.assertIs(adapter.logical_waypoint(-5, 1000.0), sidewalk)
        self.assertEqual(adapter.legal_driving_lane_ids(1000.0), {-1, -2, -3})

    def test_low_signal_values_reach_camera_attributes(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        attributes = runner.camera_sensor_attributes(
            camera_name="front_rgb",
            image_width=1280,
            image_height=720,
            fov=90.0,
            camera_tick=0.2,
            low_signal_config=config["sensors"]["low_signal_rgb"],
        )
        self.assertEqual(attributes["exposure_mode"], "manual")
        self.assertEqual(attributes["iso"], "1600.0")
        self.assertEqual(attributes["gamma"], "1.8")
        self.assertEqual(attributes["motion_blur_intensity"], "0.15")

    def test_voice_and_competition_thresholds(self):
        config = runner.load_runtime_config(CONFIG)
        commands = config["voice_input"]["commands"]
        texts = {item["text"] for item in commands}
        self.assertIn("前方路况危险，保持安全车速", texts)
        self.assertIn("突发车辆加塞，紧急避让", texts)
        self.assertIn("施工路段，减速并道至左侧车道", texts)
        self.assertLessEqual(
            config["success_conditions"]["maximum_emergency_response_latency_ms"],
            120,
        )
        self.assertGreaterEqual(
            config["success_conditions"][
                "minimum_multimodal_semantic_alignment_accuracy"
            ],
            0.97,
        )

    def test_dynamic_vehicle_state_is_persisted_by_simulation_frame(self):
        vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        ego = SimpleNamespace(
            get_velocity=lambda: vector(10.0, 0.5, 0.0),
            get_acceleration=lambda: vector(0.3, 0.0, 0.0),
            get_angular_velocity=lambda: vector(0.0, 0.0, 1.2),
            get_control=lambda: SimpleNamespace(
                throttle=0.2,
                steer=-0.1,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                gear=2,
            ),
        )
        waypoint = SimpleNamespace(road_id=7, section_id=0, lane_id=-2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vehicle_state.jsonl"
            recorder = runner.VehicleStateRecorder(path)
            recorder.record(
                ego=ego,
                waypoint=waypoint,
                simulation_frame=123,
                timestamp_s=6.15,
                route_progress_m=82.4,
            )
            recorder.close()
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["simulation_frame"], 123)
        self.assertEqual(row["lane"]["lane_id"], -2)
        self.assertGreater(row["speed_kmh"], 0.0)
        self.assertEqual(row["control"]["gear"], 2)

    def test_preview_validation_runs_without_carla(self):
        result = preview.main(
            [
                "--validate-only",
                "--no-ground-truth",
                "--no-strict-completion",
            ]
        )
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
