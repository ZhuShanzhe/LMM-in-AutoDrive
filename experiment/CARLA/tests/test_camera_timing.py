import unittest

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


if __name__ == "__main__":
    unittest.main()
