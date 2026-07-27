import unittest
from collections import deque

from evaluation.camera import ExperimentCamera


class ExperimentCameraTimingTests(unittest.TestCase):
    def test_fixed_step_frames_preserve_requested_video_duration(self):
        camera = ExperimentCamera.__new__(ExperimentCamera)
        camera.sensor_tick = 0.05
        camera.video_fps = 30.0
        camera._video_frame_remainder = 0.0

        repeats = [camera._video_repeat_count() for _ in range(20)]

        self.assertEqual(sum(repeats), 30)
        self.assertEqual(repeats[:4], [1, 2, 1, 2])

    def test_observed_simulation_period_overrides_requested_sensor_period(self):
        camera = ExperimentCamera.__new__(ExperimentCamera)
        camera.sensor_tick = 1.0 / 30.0
        camera.video_fps = 30.0
        camera._video_frame_remainder = 0.0

        repeats = [camera._video_repeat_count(0.05) for _ in range(20)]

        self.assertEqual(sum(repeats), 30)

    def test_telemetry_overlay_accepts_live_latency_and_qwen_status(self):
        camera = ExperimentCamera.__new__(ExperimentCamera)
        camera.width = 640
        camera.height = 360
        camera._font_cache = {}
        camera._speed_history = deque()
        raw = bytes(camera.width * camera.height * 4)
        overlay = {
            "status": "RUNNING",
            "route_progress_m": 1250,
            "route_length_m": 5000,
            "asr_text": "Change to the right lane.",
            "action": "lane_change_right",
            "active_step_id": "step_2",
            "risk_level": "MEDIUM",
            "traffic_count": 8,
            "pedestrian_count": 1,
            "policy_state": "READY",
            "speed_kmh": 43.2,
            "target_speed_kmh": 50.0,
            "parse_latency_ms": 14.2,
            "perception_latency_ms": 38.5,
            "scene_decision_latency_ms": 1.7,
            "end_to_end_ms": 41.0,
            "qwen_status": "submitted",
            "qwen_worker": {"completed": 1, "dropped": 0},
            "sim_time_s": 2.1,
            "collisions": 0,
            "lane_events": 0,
        }

        rendered = camera._render_overlay(raw, overlay)

        self.assertEqual(len(rendered), len(raw))
        self.assertNotEqual(rendered, raw)


if __name__ == "__main__":
    unittest.main()
