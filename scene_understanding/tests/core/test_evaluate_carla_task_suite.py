from __future__ import annotations

import unittest

from scene_understanding.core.evaluate_carla_task_suite import summarize


class CarlaTaskSuiteTests(unittest.TestCase):
    def test_completion_threshold_is_system_level(self) -> None:
        records = [
            {
                "scenario": f"scenario_{index}",
                "task_completed": index < 9,
                "collision_free": True,
                "violation_free": True,
            }
            for index in range(10)
        ]
        result = summarize(records)
        self.assertEqual(0.9, result["task_completion_rate"])
        self.assertTrue(result["meets_90_percent_task_completion"])
        self.assertIn("system-level", result["note"])


if __name__ == "__main__":
    unittest.main()
