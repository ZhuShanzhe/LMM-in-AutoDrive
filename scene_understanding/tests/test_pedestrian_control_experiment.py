import subprocess
import sys
import unittest

from scene_understanding.scripts.run_pedestrian_control_experiment import (
    pedestrian_step_completed,
)


def pedestrian(*, lane_relation="roadside", lateral=-3.0):
    return {
        "lane_relation": lane_relation,
        "relative_position_ego_m": {"lateral": lateral},
    }


class PedestrianControlExperimentTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "observed_crossing": True,
            "pedestrian": pedestrian(),
            "initial_speed_mps": 8.3,
            "current_speed_mps": 4.0,
            "minimum_speed_reduction_mps": 3.0,
            "collision_count": 0,
        }
        values.update(overrides)
        return pedestrian_step_completed(**values)

    def test_completes_only_after_measured_clearance_and_deceleration(self):
        completed, reasons = self.evaluate()
        self.assertTrue(completed)
        self.assertEqual(
            reasons,
            [
                "pedestrian_crossing_cleared",
                "ego_speed_reduced",
                "collision_free",
            ],
        )

    def test_does_not_complete_while_pedestrian_is_crossing(self):
        completed, reasons = self.evaluate(
            pedestrian=pedestrian(
                lane_relation="crossing_ego_path",
                lateral=-1.0,
            )
        )
        self.assertFalse(completed)
        self.assertIn("pedestrian_still_crossing", reasons)
        self.assertIn("pedestrian_has_not_cleared_far_side", reasons)

    def test_requires_crossing_to_have_been_observed(self):
        completed, reasons = self.evaluate(observed_crossing=False)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["crossing_not_observed"])

    def test_requires_sufficient_speed_reduction(self):
        completed, reasons = self.evaluate(current_speed_mps=6.0)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["ego_speed_reduction_insufficient"])

    def test_collision_always_prevents_completion(self):
        completed, reasons = self.evaluate(collision_count=1)
        self.assertFalse(completed)
        self.assertEqual(reasons, ["collision_detected"])

    def test_help_does_not_require_carla_or_team_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scene_understanding.scripts.run_pedestrian_control_experiment",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scenario-root", result.stdout)
        self.assertIn("--control-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
