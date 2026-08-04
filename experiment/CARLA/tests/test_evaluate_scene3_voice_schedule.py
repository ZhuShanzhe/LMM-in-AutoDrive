from __future__ import annotations

import sys
import unittest
from pathlib import Path


CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from tools.evaluate_scene3_voice_schedule import evaluate


class _Parser:
    def parse(self, text, **_kwargs):
        if text == "none":
            return None
        if text == "error":
            raise ValueError("bad command")
        return {
            "parse_result": {"status": "VALID", "method": "RULE", "latency_ms": 1.0},
            "intent": {"steps": [{"action": "KEEP_LANE"}]},
        }


class EvaluateScene3VoiceScheduleTests(unittest.TestCase):
    def test_preserves_empty_and_exception_results(self):
        schedule = [
            {"command_id": "a", "text": "valid"},
            {"command_id": "b", "text": "none"},
            {"command_id": "c", "text": "error"},
        ]

        rows = evaluate(schedule, _Parser())

        self.assertEqual(rows[0]["parse_status"], "VALID")
        self.assertIsNone(rows[1]["parse_status"])
        self.assertEqual(rows[2]["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
