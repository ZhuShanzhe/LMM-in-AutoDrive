import subprocess
import sys
import unittest

from scene_understanding.scripts.run_overtake_control_experiment import (
    _speed_capped_decision,
    overtake_step_completed,
)


class OvertakeControlExperimentTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "observed_slow_vehicle_ahead": True,
            "slow_vehicle_present": True,
            "slow_vehicle_longitudinal_m": -9.0,
            "rear_clearance_m": 8.0,
            "current_lane_id": 1,
            "passing_lane_id": 1,
            "lane_center_offset_m": 0.4,
            "maximum_lane_center_offset_m": 0.9,
            "heading_alignment": 0.99,
            "minimum_heading_alignment": 0.95,
            "stable_frames": 5,
            "required_stable_frames": 5,
            "collision_count": 0,
        }
        values.update(overrides)
        return overtake_step_completed(**values)

    def test_completes_only_after_measured_rear_clearance(self):
        completed, reasons = self.evaluate()
        self.assertTrue(completed)
        self.assertEqual(
            reasons,
            [
                "slow_vehicle_passed_with_rear_clearance",
                "ego_stable_in_passing_lane",
                "slow_vehicle_grounded",
                "collision_free",
            ],
        )

    def test_requires_slow_vehicle_to_have_started_ahead(self):
        completed, reasons = self.evaluate(observed_slow_vehicle_ahead=False)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["slow_vehicle_was_not_observed_ahead"])

    def test_requires_full_rear_clearance(self):
        completed, reasons = self.evaluate(slow_vehicle_longitudinal_m=-7.9)
        self.assertFalse(completed)
        self.assertIn("rear_clearance_not_reached", reasons)

    def test_requires_ego_to_remain_in_passing_lane(self):
        completed, reasons = self.evaluate(current_lane_id=2)
        self.assertFalse(completed)
        self.assertIn("ego_left_passing_lane", reasons)

    def test_collision_always_prevents_completion(self):
        completed, reasons = self.evaluate(collision_count=1)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["collision_detected"])

    def test_speed_cap_converts_only_acceleration_to_bounded_keep_lane(self):
        decision = {
            "decision_status": "READY",
            "action": "accelerate",
            "target_speed_kmh": 40.0,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "driving_intent_overtake",
            "blocked_reason_codes": [],
        }
        capped = _speed_capped_decision(
            decision,
            current_speed_kmh=40.1,
            maximum_speed_kmh=40.0,
        )
        self.assertEqual(capped["action"], "keep_lane")
        self.assertEqual(capped["target_speed_kmh"], 40.0)
        self.assertEqual(capped["reason"], "overtake_speed_cap")

    def test_speed_cap_never_weakens_emergency_braking(self):
        decision = {
            "decision_status": "BLOCKED",
            "action": "emergency_brake",
            "target_speed_kmh": 0.0,
            "target_lane": None,
            "target_location": None,
            "emergency": True,
            "reason": "risk_requires_emergency_brake",
            "blocked_reason_codes": ["risk_requires_emergency_brake"],
        }
        result = _speed_capped_decision(
            decision,
            current_speed_kmh=45.0,
            maximum_speed_kmh=40.0,
        )
        self.assertEqual(result, decision)

    def test_help_does_not_require_carla_or_team_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scene_understanding.scripts.run_overtake_control_experiment",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--rear-clearance-m", result.stdout)
        self.assertIn("--maximum-ego-speed-kmh", result.stdout)


if __name__ == "__main__":
    unittest.main()
