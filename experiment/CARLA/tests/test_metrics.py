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
        "control": {"steer": 0.0},
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
            record["intent"] = {"action": "decelerate", "target_speed_kmh": 10.0}
        metrics = summarize(records, "emergency_brake", goal_distance_m=1.0)
        self.assertEqual(metrics["speeding_frames"], 0)
        self.assertTrue(metrics["violation_free"])

    def test_scene_deceleration_during_lane_change_is_not_speeding(self):
        records = [
            make_record(1, 0.0, speed_kmh=48.0),
            make_record(2, 1.0, speed_kmh=43.0),
        ]
        for record in records:
            record["intent"] = {
                "action": "lane_change_right",
                "target_speed_kmh": 38.0,
            }
            record["scene_decision"] = {
                "control_decision": {"action": "decelerate"}
            }
        metrics = summarize(records, "basic_voice_urban_5km", goal_distance_m=1.0)
        self.assertEqual(metrics["speeding_frames"], 0)
        self.assertTrue(metrics["violation_free"])

    def test_turn_speed_reduction_is_not_a_speeding_violation(self):
        records = [
            make_record(1, 0.0, speed_kmh=45.0),
            make_record(2, 1.0, speed_kmh=35.0),
        ]
        for record in records:
            record["intent"] = {"action": "turn_right", "target_speed_kmh": 20.0}
        metrics = summarize(records, "basic_voice_control_5km", goal_distance_m=1.0)
        self.assertEqual(metrics["speeding_frames"], 0)

    def test_active_braking_toward_new_set_speed_is_not_speeding(self):
        records = [
            make_record(1, 0.0, speed_kmh=42.0),
            make_record(2, 1.0, speed_kmh=38.0),
        ]
        for record in records:
            record["intent"] = {"action": "keep_lane", "target_speed_kmh": 30.0}
            record["control"] = {"steer": 0.0, "brake": 0.4}

        metrics = summarize(records, "basic_voice_control_5km", goal_distance_m=1.0)

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
        metrics = summarize(records, "basic_track_5km", goal_distance_m=1.0)
        self.assertEqual(metrics["lane_invasion_events"], 3)
        self.assertEqual(metrics["illegal_lane_invasion_events"], 0)
        self.assertTrue(metrics["violation_free"])

    def test_illegal_lane_invasion_fails_the_violation_check(self):
        records = [make_record(1, 0.0), make_record(2, 1.0)]
        for record in records:
            record["events"]["illegal_lane_invasion_count"] = 1
        metrics = summarize(records, "basic_track_5km", goal_distance_m=1.0)
        self.assertFalse(metrics["violation_free"])

    def test_steering_smoothness_metrics_use_per_frame_dynamics(self):
        records = [
            make_record(1, 0.0),
            make_record(2, 1.0),
            make_record(3, 2.0),
            make_record(4, 3.0),
        ]
        values = [
            (0.00, 0.0, 0.0),
            (0.04, 0.8, 16.0),
            (0.08, 0.8, 0.0),
            (-0.04, -2.4, -64.0),
        ]
        for record, (steer, rate, accel) in zip(records, values):
            record["steering_dynamics"] = {
                "normalized_steer": steer,
                "steer_rate_per_s": rate,
                "steer_accel_per_s2": accel,
                "action": "keep_lane",
            }

        metrics = summarize(records, "basic_track_5km")

        self.assertEqual(metrics["steer_rate_abs_max_per_s"], 2.4)
        self.assertEqual(metrics["steer_accel_abs_max_per_s2"], 64.0)
        self.assertEqual(metrics["steer_direction_reversal_count"], 1)
        self.assertEqual(metrics["straight_steer_abs_p95"], 0.08)


if __name__ == "__main__":
    unittest.main()
