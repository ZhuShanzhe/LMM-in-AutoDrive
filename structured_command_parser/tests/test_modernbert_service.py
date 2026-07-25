from __future__ import annotations

import unittest

from structured_command_parser.src.modernbert_service import ModernBertCommandService
from structured_command_parser.src.modernbert_parser import ModernBertEnglishIntentParser


class FakeParser:
    def __init__(self) -> None:
        self.warmups = 0
        self.calls: list[tuple[str, str, str | None]] = []

    def warmup(self) -> None:
        self.warmups += 1

    def parse(
        self,
        text: str,
        *,
        modality: str,
        request_id: str | None,
    ) -> dict:
        self.calls.append((text, modality, request_id))
        return {"request_id": request_id, "text": text, "modality": modality}


class ModernBertCommandServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FakeParser()
        self.service = ModernBertCommandService(parser=self.parser)

    def test_warmup_is_forwarded(self) -> None:
        self.service.warmup()
        self.assertEqual(self.parser.warmups, 1)

    def test_parse_text_normalizes_and_forwards_metadata(self) -> None:
        result = self.service.parse_text(
            "  slow   down  ", request_id="req-1", modality="VOICE"
        )
        self.assertEqual(result["text"], "slow down")
        self.assertEqual(self.parser.calls, [("slow down", "VOICE", "req-1")])

    def test_handle_message_accepts_translation_contract(self) -> None:
        result = self.service.handle_message(
            {
                "request_id": "req-2",
                "text": "Turn right at the junction.",
                "language": "en-US",
                "modality": "TEXT",
            }
        )
        self.assertEqual(result["request_id"], "req-2")

    def test_invalid_messages_fail_before_inference(self) -> None:
        with self.assertRaises(TypeError):
            self.service.handle_message({"text": 3})
        with self.assertRaises(ValueError):
            self.service.handle_message({"text": "turn right", "language": "zh-CN"})
        with self.assertRaises(ValueError):
            self.service.parse_text(" ")
        self.assertEqual(self.parser.calls, [])

    def test_withholds_ungrounded_stop_classifier_label(self) -> None:
        payload = {
            "commands": [
                {"action": "TURN", "direction": "RIGHT"},
                {"action": "STOP"},
            ],
            "warnings": [],
        }
        result = ModernBertEnglishIntentParser._withhold_ungrounded_terminal_actions(
            payload,
            "Turn right at the next junction.",
        )
        self.assertEqual(result["commands"], [{"action": "TURN", "direction": "RIGHT"}])
        self.assertEqual(
            result["warnings"],
            ["classifier_action_withheld_without_text_evidence:STOP"],
        )

    def test_keeps_explicit_stop_classifier_label(self) -> None:
        payload = {"commands": [{"action": "STOP"}], "warnings": []}
        result = ModernBertEnglishIntentParser._withhold_ungrounded_terminal_actions(
            payload,
            "Stop before the red truck.",
        )
        self.assertEqual(result, payload)


if __name__ == "__main__":
    unittest.main()
