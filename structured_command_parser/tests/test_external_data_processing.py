from __future__ import annotations

import unittest

from structured_command_parser.scripts.prepare_external_commands import (
    expected_for_simlingo,
)
from structured_command_parser.scripts.translate_external_commands import contains_chinese
from structured_command_parser.scripts.translate_external_commands import preserve_urgency


class ExternalDataProcessingTest(unittest.TestCase):
    def test_unsafe_description_is_not_treated_as_speed_command(self) -> None:
        candidate = {
            "mode": "faster",
            "instruction": "A crash with a car occurs after speeding up.",
        }
        self.assertIsNone(expected_for_simlingo(candidate))

    def test_mode_label_without_matching_text_is_rejected(self) -> None:
        candidate = {
            "mode": "faster",
            "instruction": "Commit a red light infraction.",
        }
        self.assertIsNone(expected_for_simlingo(candidate))

    def test_implicit_crash_trajectory_is_not_textually_unsupported(self) -> None:
        candidate = {
            "mode": "crash",
            "instruction": "Advance towards the object at x: 8m, y: 2m.",
        }
        self.assertIsNone(expected_for_simlingo(candidate))

    def test_explicit_crash_command_is_unsupported(self) -> None:
        candidate = {
            "mode": "crash",
            "instruction": "Crash into the parked vehicle.",
        }
        expected = expected_for_simlingo(candidate)
        self.assertIsNotNone(expected)
        self.assertEqual(expected["status"], "UNSUPPORTED")

    def test_chinese_translation_detection(self) -> None:
        self.assertTrue(contains_chinese("超过银色汽车后右转。"))
        self.assertFalse(contains_chinese("turn right after the silver car"))

    def test_urgent_meaning_is_preserved_after_translation(self) -> None:
        self.assertEqual(
            preserve_urgency("Stop the vehicle at once.", "停车。"),
            "立即停车。",
        )

    def test_target_speed_gold_uses_instruction_units(self) -> None:
        candidate = {
            "mode": "target_speed",
            "instruction": "Adjust your speed to 58.3 km/h.",
            "info": {"target_speed": 16.2},
        }
        expected = expected_for_simlingo(candidate)
        self.assertEqual(expected["target_speed_mps"], [16.194])

    def test_immediate_stop_maps_to_emergency_brake(self) -> None:
        candidate = {
            "mode": "stop",
            "instruction": "Stop the car immediately.",
        }
        expected = expected_for_simlingo(candidate)
        self.assertEqual(expected["category"], "EMERGENCY_RESPONSE")
        self.assertEqual(expected["actions"], ["EMERGENCY_BRAKE"])


if __name__ == "__main__":
    unittest.main()
