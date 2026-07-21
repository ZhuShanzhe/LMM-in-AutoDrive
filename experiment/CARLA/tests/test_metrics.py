"""Regression tests for control/evaluation result semantics."""

import unittest

from evaluation.metrics import summarize


def make_record(frame, distance_m, speed_kmh=10.0):
    return {
        "frame": frame,
        "sim_time_s": frame * 0.05,
        "distance_m": distance_m,
        "ego": {"speed_kmh": speed_kmh},
        "intent": {"action": "keep_lane", "target_speed_kmh": 20.0},
        "events": {"collision_count": 0, "lane_invasion_count": 0},
        "latency_ms": {"end_to_end": 1.0, "control": 0.5},
    }


class MetricsResultSemanticsTest(unittest.TestCase):
    def test_missing_external_goal_is_not_automatically_complete(self):
        metrics = summarize(
            [make_record(1, 0.0), make_record(2, 1.0)],
            "emergency_brake",
        )

        self.assertFalse(metrics["goal_reached"])
        self.assertFalse(metrics["task_completed"])


if __name__ == "__main__":
    unittest.main()
