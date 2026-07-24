from __future__ import annotations

import unittest

from structured_command_parser.src.english_parser import QwenEnglishIntentParser
from structured_command_parser.src.intent_boundaries import (
    classify_chinese_braking,
    classify_english_braking,
)
from structured_command_parser.src.llm_parser import QwenIntentParser


class IntentBoundaryUnitTests(unittest.TestCase):
    def assert_boundary(self, boundary, action: str, urgency: str) -> None:
        self.assertIsNotNone(boundary)
        self.assertEqual(boundary.action, action)
        self.assertEqual(boundary.urgency, urgency)

    def test_english_four_level_boundary(self) -> None:
        cases = {
            "Slam on the brakes.": ("EMERGENCY_BRAKE", "EMERGENCY"),
            "A child appeared suddenly; stop immediately.": (
                "EMERGENCY_BRAKE",
                "EMERGENCY",
            ),
            "Stop immediately so I can talk to that person.": ("STOP", "URGENT"),
            "Stop the vehicle.": ("STOP", "NORMAL"),
            "Brake immediately.": ("ADJUST_SPEED", "URGENT"),
            "Hit the brakes.": ("ADJUST_SPEED", "NORMAL"),
            "Slow down.": ("ADJUST_SPEED", "NORMAL"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assert_boundary(classify_english_braking(text), *expected)

    def test_chinese_four_level_boundary(self) -> None:
        cases = {
            "猛踩刹车": ("EMERGENCY_BRAKE", "EMERGENCY"),
            "前方突然冲出行人，立即停车": ("EMERGENCY_BRAKE", "EMERGENCY"),
            "请立即停车": ("STOP", "URGENT"),
            "把车停下来": ("STOP", "NORMAL"),
            "立即踩刹车": ("ADJUST_SPEED", "URGENT"),
            "轻踩刹车": ("ADJUST_SPEED", "NORMAL"),
            "减速": ("ADJUST_SPEED", "NORMAL"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assert_boundary(classify_chinese_braking(text), *expected)


class IntentBoundaryIntegrationTests(unittest.TestCase):
    @staticmethod
    def payload(action: str) -> dict:
        return {
            "category": "EMERGENCY_RESPONSE",
            "urgency": "EMERGENCY",
            "commands": [{"action": action}],
            "driving_style": "NORMAL",
            "status": "VALID",
            "missing_slots": [],
            "warnings": [],
        }

    def test_english_postprocessor_downgrades_unsupported_emergency_guess(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            self.payload("EMERGENCY_BRAKE"), "Stop immediately."
        )
        self.assertEqual(payload["urgency"], "URGENT")
        self.assertEqual(payload["commands"], [{"action": "STOP"}])

    def test_chinese_postprocessor_downgrades_unsupported_emergency_guess(self) -> None:
        payload = QwenIntentParser._normalize_payload(
            self.payload("EMERGENCY_BRAKE"), "立即停止车辆"
        )
        self.assertEqual(payload["urgency"], "URGENT")
        self.assertEqual(payload["commands"], [{"action": "STOP"}])

    def test_explicit_emergency_evidence_is_preserved(self) -> None:
        english = QwenEnglishIntentParser._normalize_payload(
            self.payload("STOP"), "Apply the emergency brake."
        )
        chinese = QwenIntentParser._normalize_payload(
            self.payload("STOP"), "立即紧急制动"
        )
        for payload in (english, chinese):
            self.assertEqual(payload["urgency"], "EMERGENCY")
            self.assertEqual(payload["commands"], [{"action": "EMERGENCY_BRAKE"}])


if __name__ == "__main__":
    unittest.main()
