from __future__ import annotations

import unittest

from scene_understanding.training.build_carla_domain_dataset import (
    split_by_scenario,
    yolo_line,
)


class CarlaDomainDatasetTests(unittest.TestCase):
    def test_yolo_line_converts_normalized_xyxy(self) -> None:
        self.assertEqual(
            "4 0.500000 0.400000 0.400000 0.400000",
            yolo_line("traffic_light", [0.3, 0.2, 0.7, 0.6]),
        )
        self.assertIsNone(yolo_line("unknown", [0.3, 0.2, 0.7, 0.6]))
        self.assertIsNone(yolo_line("vehicle", [0.7, 0.2, 0.3, 0.6]))

    def test_split_keeps_validation_examples_from_each_scenario(self) -> None:
        records = [
            {"_scenario": scenario, "frame_id": f"{scenario}_{index}"}
            for scenario in ("a", "b")
            for index in range(10)
        ]
        train, val = split_by_scenario(records, val_fraction=0.2, seed=2026)
        self.assertEqual(16, len(train))
        self.assertEqual(4, len(val))
        self.assertEqual({"a", "b"}, {item["_scenario"] for item in val})


if __name__ == "__main__":
    unittest.main()
