from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scene_understanding.core.qwen_scene_service import (
    LatestFrameWorker,
    QwenSceneConfig,
)
from scene_understanding.core.run_qwen_scene_inference import infer_one_record


class _Service:
    def infer(self, record):
        return {"frame_id": record["frame_id"], "status": "valid"}


class _FailingService:
    def infer(self, record):
        raise RuntimeError(f"failed {record['frame_id']}")


class _BlockingService:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, record):
        if record["frame_id"] == "frame_1":
            self.started.set()
            self.release.wait(1.0)
        return {"frame_id": record["frame_id"], "status": "valid"}


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class QwenSceneServiceTests(unittest.TestCase):
    def test_config_rejects_invalid_token_bounds(self):
        with self.assertRaisesRegex(ValueError, "visual token"):
            QwenSceneConfig(Path("/model"), min_visual_tokens=1024, max_visual_tokens=256)

    @mock.patch("scene_understanding.core.run_qwen_scene_inference.generate_one")
    def test_single_record_helper_preserves_cli_result_contract(self, generate_one):
        output = {
            "schema_version": "1.0",
            "frame_id": "frame_1",
            "source": "carla",
            "camera_name": "front_rgb",
            "scene": {
                "summary": "No decision-relevant object is visible.",
                "road_type": "unknown",
                "is_intersection": None,
                "weather": "unknown",
                "visibility": "unknown",
                "traffic_light_state": "not_visible",
                "left_lane_marking": "unknown",
                "right_lane_marking": "unknown",
            },
            "objects": [],
            "potential_hazards": [],
        }
        generate_one.return_value = (json.dumps(output), 0.25, 2.5, (800, 600))
        result = infer_one_record(
            {
                "frame_id": "frame_1",
                "source": "carla",
                "camera_name": "front_rgb",
                "image_path": "/tmp/frame.png",
                "prompt": "prompt",
            },
            model=object(),
            processor=object(),
            torch_module=object(),
            model_path=Path("/models/qwen"),
            max_new_tokens=256,
            min_visual_tokens=128,
            max_visual_tokens=512,
        )
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["elapsed_seconds"], 0.25)
        self.assertEqual(result["parsed_output"]["frame_id"], "frame_1")

    def test_worker_returns_latest_result(self):
        worker = LatestFrameWorker(_Service())
        try:
            worker.submit({"frame_id": "frame_1"})
            result = worker.wait_for_result(1.0)
            self.assertIsNotNone(result)
            self.assertEqual(result["frame_id"], "frame_1")
            self.assertIn("service_elapsed_seconds", result)
        finally:
            worker.close()

    def test_worker_records_inference_errors(self):
        worker = LatestFrameWorker(_FailingService())
        try:
            worker.submit({"frame_id": "frame_2", "source": "carla"})
            result = worker.wait_for_result(1.0)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "RuntimeError")
            self.assertEqual(worker.stats()["errors"], 1)
        finally:
            worker.close()

    def test_latest_result_can_expire(self):
        clock = _Clock()
        worker = LatestFrameWorker(_Service(), monotonic_clock=clock)
        try:
            worker.submit({"frame_id": "frame_3"})
            self.assertIsNotNone(worker.wait_for_result(1.0))
            clock.advance(0.002)
            self.assertIsNone(worker.latest(max_age_seconds=0.001))
        finally:
            worker.close()

    def test_pending_frame_is_replaced_by_newest_frame(self):
        service = _BlockingService()
        worker = LatestFrameWorker(service)
        try:
            worker.submit({"frame_id": "frame_1"})
            self.assertTrue(service.started.wait(1.0))
            worker.submit({"frame_id": "frame_2"})
            worker.submit({"frame_id": "frame_3"})
            service.release.set()
            deadline = time.monotonic() + 1.0
            while worker.stats()["completed"] < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(worker.latest()["frame_id"], "frame_3")
            self.assertEqual(worker.stats()["dropped"], 1)
        finally:
            service.release.set()
            worker.close()


if __name__ == "__main__":
    unittest.main()
