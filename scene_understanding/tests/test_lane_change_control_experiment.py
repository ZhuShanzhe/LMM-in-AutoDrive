import subprocess
import sys
import unittest

from scene_understanding.scripts.run_lane_change_control_experiment import (
    _target_lane_hold_decision,
    lane_change_step_completed,
)


class LaneChangeControlExperimentTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "observed_slow_vehicle": True,
            "slow_vehicle_present": True,
            "current_lane_id": -1,
            "target_lane_id": -1,
            "lane_center_offset_m": 0.4,
            "maximum_lane_center_offset_m": 0.9,
            "heading_alignment": 0.99,
            "minimum_heading_alignment": 0.95,
            "stable_frames": 5,
            "required_stable_frames": 5,
            "collision_count": 0,
        }
        values.update(overrides)
        return lane_change_step_completed(**values)

    def test_completes_on_grounded_stable_collision_free_lane_change(self):
        completed, reasons = self.evaluate()
        self.assertTrue(completed)
        self.assertEqual(
            reasons,
            [
                "target_lane_reached",
                "target_lane_centered_and_stable",
                "slow_vehicle_grounded",
                "collision_free",
            ],
        )

    def test_requires_target_lane_identity(self):
        completed, reasons = self.evaluate(current_lane_id=-2)
        self.assertFalse(completed)
        self.assertIn("target_lane_not_reached", reasons)

    def test_completion_allows_connected_opendrive_road_segments(self):
        completed, _ = self.evaluate(current_lane_id=-1)
        self.assertTrue(completed)

    def test_requires_lane_center_and_heading_alignment(self):
        completed, reasons = self.evaluate(
            lane_center_offset_m=1.2,
            heading_alignment=0.8,
        )
        self.assertFalse(completed)
        self.assertIn("ego_not_centered_in_target_lane", reasons)
        self.assertIn("ego_not_aligned_with_target_lane", reasons)

    def test_requires_stable_frames(self):
        completed, reasons = self.evaluate(stable_frames=4)
        self.assertFalse(completed)
        self.assertIn("target_lane_not_stable", reasons)

    def test_requires_slow_vehicle_to_remain_grounded(self):
        completed, reasons = self.evaluate(slow_vehicle_present=False)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["slow_vehicle_not_in_world_state"])

    def test_collision_always_prevents_completion(self):
        completed, reasons = self.evaluate(collision_count=1)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["collision_detected"])

    def test_help_does_not_require_carla_or_team_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scene_understanding.scripts.run_lane_change_control_experiment",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--spawn-index", result.stdout)
        self.assertIn("--required-stable-frames", result.stdout)

    def test_target_lane_hold_never_weakens_emergency_braking(self):
        decision = {
            "decision_status": "BLOCKED",
            "action": "emergency_brake",
            "target_speed_kmh": 0.0,
            "target_lane": "left",
            "target_location": None,
            "emergency": True,
            "reason": "risk_requires_emergency_brake",
            "blocked_reason_codes": ["risk_requires_emergency_brake"],
        }
        held = _target_lane_hold_decision(
            decision,
            current_speed_kmh=20.0,
            hold_location={"x": 1.0, "y": 2.0, "z": 0.0},
        )
        self.assertEqual(held["action"], "emergency_brake")
        self.assertTrue(held["emergency"])
        self.assertEqual(held["target_speed_kmh"], 0.0)
        self.assertEqual(
            held["blocked_reason_codes"],
            ["risk_requires_emergency_brake"],
        )


if __name__ == "__main__":
    unittest.main()
