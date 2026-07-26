from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from structured_command_parser.scripts.prepare_asr_ingress import (
    build_ingress_message,
    main,
)
from structured_command_parser.scripts.run_english_parser_service import read_message


class AsrIngressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "chinese_text": "保持当前车道并将速度提高到六十公里每小时",
            "english_translation": "Keep the current lane and set speed to 60 km/h.",
            "asr_processing_time_seconds": 0.12,
            "translation_time_seconds": 0.08,
            "total_time_seconds": 0.2,
        }

    def test_builds_parser_message_with_voice_provenance(self):
        message = build_ingress_message(self.result, request_id="voice-001")
        self.assertEqual(message["request_id"], "voice-001")
        self.assertEqual(message["language"], "en-US")
        self.assertEqual(message["modality"], "VOICE")
        self.assertEqual(message["source_language"], "zh-CN")
        self.assertEqual(message["voice_pipeline_latency_ms"], 200.0)
        self.assertEqual(read_message_from_value(message)["text"], message["text"])

    def test_rejects_missing_translation(self):
        result = dict(self.result)
        result["english_translation"] = ""
        with self.assertRaisesRegex(ValueError, "english_translation"):
            build_ingress_message(result, request_id="voice-002")

    def test_command_writes_parser_compatible_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "asr_result.json"
            output_path = root / "translated_command.json"
            input_path.write_text(json.dumps(self.result), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--asr-result",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--request-id",
                        "voice-003",
                    ]
                ),
                0,
            )
            message = read_message(output_path)
            self.assertEqual(message["request_id"], "voice-003")
            self.assertEqual(message["source_text"], self.result["chinese_text"])


def read_message_from_value(message: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "message.json"
        path.write_text(json.dumps(message), encoding="utf-8")
        return read_message(path)
