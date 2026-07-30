from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lightweight_vla_adapter.src.teacher import (
    JsonlTeacherStore,
    compare_action_predictions,
    parse_teacher_prediction,
)


class TeacherAdapterTest(unittest.TestCase):
    def teacher_record(self):
        return {
            "sample_id": "sample-1",
            "model": "unidrivevla-base",
            "action_logits": [0.0, 0.1, 0.2, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            "target_speed_kmh": 20.0,
            "latency_ms": 250.0,
            "trajectory": [[0.0, 0.0], [1.0, 0.1]],
        }

    def test_teacher_record_normalization(self):
        prediction = parse_teacher_prediction(self.teacher_record())
        self.assertEqual(prediction.action, "lane_change_left")

    def test_simlingo_teacher_is_supported(self):
        record = self.teacher_record()
        record["model"] = "simlingo"
        prediction = parse_teacher_prediction(record)
        self.assertEqual(prediction.model, "simlingo")

    def test_jsonl_store_and_student_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            path.write_text(
                json.dumps(self.teacher_record()) + "\n",
                encoding="utf-8",
            )
            metrics = compare_action_predictions(
                [{"sample_id": "sample-1", "action": "lane_change_left"}],
                JsonlTeacherStore.from_path(path),
            )
        self.assertEqual(metrics["teacher_action_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
