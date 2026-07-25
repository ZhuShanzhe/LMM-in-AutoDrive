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

    def test_deceleration_tracking_is_not_a_speeding_violation(self):
        records = [
            make_record(1, 0.0, speed_kmh=25.0),
            make_record(2, 1.0, speed_kmh=20.0),
        ]
        for record in records:
            record["intent"] = {
                "action": "decelerate",
                "target_speed_kmh": 10.0,
            }
        metrics = summarize(
            records,
            "emergency_brake",
            goal_distance_m=1.0,
        )
        self.assertEqual(metrics["speeding_frames"], 0)
        self.assertTrue(metrics["violation_free"])

    def test_non_illegal_lane_observations_do_not_fail_a_run(self):
        records = [make_record(1, 0.0), make_record(2, 1.0)]
        for record in records:
            record["events"] = {
                "collision_count": 0,
                "lane_invasion_count": 3,
                "illegal_lane_invasion_count": 0,
            }
        metrics = summarize(
            records,
            "basic_track_5km",
            goal_distance_m=1.0,
        )
        self.assertEqual(metrics["lane_invasion_events"], 3)
        self.assertEqual(metrics["illegal_lane_invasion_events"], 0)
        self.assertTrue(metrics["violation_free"])

    def test_illegal_lane_invasion_fails_the_violation_check(self):
        records = [make_record(1, 0.0), make_record(2, 1.0)]
        for record in records:
            record["events"]["illegal_lane_invasion_count"] = 1
        metrics = summarize(
            records,
            "basic_track_5km",
            goal_distance_m=1.0,
        )
        self.assertFalse(metrics["violation_free"])


if __name__ == "__main__":
    unittest.main()
