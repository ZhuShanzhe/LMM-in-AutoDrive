import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scene_understanding_capture as capture_module


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame


class FakeWorld:
    def __init__(self, frame):
        self.frame = frame

    def get_snapshot(self):
        return FakeSnapshot(self.frame)

    def get_actors(self):
        return []


class FakeSensors:
    camera_record = {
        "camera_name": "front_rgb",
        "frame": 20,
        "image_path": "/tmp/00000020.png",
    }

    def __init__(self, *args, **kwargs):
        self.front_camera_sensor = object()
        self.setup_called = False
        self.destroy_called = False

    def setup(self):
        self.setup_called = True

    def wait_for_camera_frame(self, frame, *, timeout_s):
        return self.camera_record if frame == 20 else None

    def drain_events_through(self, frame):
        return {"collisions": [], "lane_invasions": []}

    def destroy(self):
        self.destroy_called = True


class FakeCollector:
    def __init__(self, *args, **kwargs):
        pass

    def collect(self, *, sensor_events):
        return {"frame_id": "carla_00000020", "simulation_frame": 20}


class SceneUnderstandingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(capture_module, "CarlaSensorManager", FakeSensors),
            patch.object(capture_module, "CarlaWorldStateCollector", FakeCollector),
            patch.object(capture_module, "project_world_state_objects", return_value={}),
            patch.object(
                capture_module,
                "write_capture_bundle",
                return_value={
                    "frame_id": "carla_00000020",
                    "simulation_frame": 20,
                },
            ),
            patch.object(capture_module, "append_jsonl"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_captures_selected_exact_frame(self):
        capture = capture_module.SceneUnderstandingCapture(
            FakeWorld(20), object(), output_dir=Path(self.temp_dir.name), every_n_frames=10
        )
        capture.setup()
        result = capture.capture_current_frame()
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["simulation_frame"], 20)
        self.assertEqual(capture.stats(), {"captured": 1, "camera_timeouts": 0})

    def test_skips_unselected_frame_and_reports_timeout(self):
        world = FakeWorld(21)
        capture = capture_module.SceneUnderstandingCapture(
            world, object(), output_dir=Path(self.temp_dir.name), every_n_frames=10
        )
        self.assertIsNone(capture.capture_current_frame())
        world.frame = 30
        self.assertEqual(
            capture.capture_current_frame(),
            {"status": "camera_timeout", "simulation_frame": 30},
        )
        self.assertEqual(capture.stats(), {"captured": 0, "camera_timeouts": 1})


if __name__ == "__main__":
    unittest.main()
