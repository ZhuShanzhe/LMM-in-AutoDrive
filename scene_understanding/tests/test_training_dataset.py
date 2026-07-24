from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scene_understanding.training.build_heldout_test_manifests import select_heldout
from scene_understanding.training.build_specialized_yolo_dataset import (
    BDD_TO_DRIVING,
    CLASSES,
    category_name,
    normalized_box,
)
from scene_understanding.training.calibrate_class_thresholds import best_threshold


class SpecializedTrainingDatasetTests(unittest.TestCase):
    def test_class_order_is_stable(self) -> None:
        self.assertEqual("vehicle", CLASSES[0])
        self.assertEqual("traffic_cone", CLASSES[7])
        self.assertEqual("cyclist", BDD_TO_DRIVING["rider"])

    def test_bdd_box_is_normalized(self) -> None:
        record = {"dataset": "BDD100K", "width": 1280, "height": 720}
        annotation = {"category": "person", "bbox_xyxy": [128, 72, 640, 360]}
        self.assertEqual("pedestrian", category_name(record, annotation))
        self.assertEqual((0.1, 0.1, 0.5, 0.5), normalized_box(record, annotation))

    def test_nuscenes_internal_category_is_kept(self) -> None:
        record = {"dataset": "nuScenes", "width": 1600, "height": 900}
        annotation = {
            "category": "traffic_cone",
            "bbox_2d": [0.2, 0.3, 0.4, 0.7],
        }
        self.assertEqual("traffic_cone", category_name(record, annotation))
        self.assertEqual((0.2, 0.3, 0.4, 0.7), normalized_box(record, annotation))

    def test_threshold_calibration_prefers_recall_weighted_cutoff(self) -> None:
        truth = {"frame": [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]}
        predictions = [
            (0.9, "frame", (0.0, 0.0, 10.0, 10.0)),
            (0.4, "frame", (20.0, 20.0, 30.0, 30.0)),
            (0.2, "frame", (40.0, 40.0, 50.0, 50.0)),
        ]
        result = best_threshold(
            predictions,
            truth,
            truth_count=2,
            iou_threshold=0.5,
            beta=2.0,
            floor=0.01,
            minimum_precision=0.5,
        )
        self.assertEqual(0.4, result["threshold"])
        self.assertEqual(1.0, result["recall"])

    def test_heldout_selection_excludes_validation_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        '{"image_path": "/data/a.jpg", "image_id": "a"}',
                        '{"image_path": "/data/b.jpg", "image_id": "b"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            selected = select_heldout(
                source,
                {str(Path("/data/a.jpg").resolve())},
                limit=1,
                seed=2026,
            )
            self.assertEqual("b", selected[0]["image_id"])


if __name__ == "__main__":
    unittest.main()
