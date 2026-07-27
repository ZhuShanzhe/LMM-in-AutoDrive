from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from structured_command_parser.scripts.run_english_parser_service import (
    message_signature,
    process_message,
    read_message,
)


class _StubService:
    def __init__(self) -> None:
        self.messages = []

    def handle_message(self, message):
        self.messages.append(dict(message))
        return {
            "schema_version": "1.1.0",
            "request_id": message["request_id"],
            "input": {"modality": message.get("modality", "VOICE")},
            "parse_result": {
                "status": "VALID",
                "method": "modernbert",
                "latency_ms": 12.5,
            },
        }


class EnglishParserServiceTests(unittest.TestCase):
    def test_reads_bom_english_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "translated.json"
            path.write_text(
                json.dumps(
                    {
                        "request_id": "asr-001",
                        "text": "Keep the current lane.",
                        "language": "en-US",
                        "modality": "VOICE",
                    }
                ),
                encoding="utf-8-sig",
            )
            self.assertEqual(read_message(path)["request_id"], "asr-001")

    def test_rejects_non_english_message_before_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "translated.json"
            path.write_text(
                json.dumps(
                    {
                        "request_id": "asr-002",
                        "text": "保持当前车道",
                        "language": "zh-CN",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identify English"):
                read_message(path)

    def test_process_writes_intent_and_receipt_atomically(self):
        service = _StubService()
        message = {
            "request_id": "translation-003",
            "text": "Change to the left lane.",
            "language": "en-US",
            "modality": "VOICE",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            intent_path = directory / "driving_intent.json"
            receipt_path = directory / "receipt.json"
            receipt = process_message(
                service,
                message,
                output_path=intent_path,
                receipt_path=receipt_path,
            )
            self.assertEqual(receipt["request_id"], "translation-003")
            self.assertEqual(json.loads(intent_path.read_text(encoding="utf-8"))["request_id"], "translation-003")
            self.assertEqual(json.loads(receipt_path.read_text(encoding="utf-8"))["parser_latency_ms"], 12.5)
            self.assertEqual(list(directory.glob(".*.tmp")), [])

    def test_signature_ignores_object_key_order(self):
        left = {"request_id": "r", "text": "Go straight.", "language": "en-US"}
        right = {"language": "en-US", "text": "Go straight.", "request_id": "r"}
        self.assertEqual(message_signature(left), message_signature(right))


if __name__ == "__main__":
    unittest.main()
