from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scene_understanding.realtime_perception.composite_backend import box_iou, class_aware_nms
from scene_understanding.realtime_perception.detector import Detection
from scene_understanding.realtime_perception.evaluate_dataset import (
    BDD_TO_DRIVING,
    load_truth,
    normalized_box,
    normalize_category,
)
from scene_understanding.realtime_perception.run_dataset import normalize_source, parse_args
from scene_understanding.realtime_perception.ultralytics_backend import (
    infrastructure_tile_regions,
    load_category_thresholds,
    offset_box,
)
from scene_understanding.realtime_perception.road_structure import road_structure_from_world_state
from scene_understanding.realtime_perception.taxonomy import (
    INTENT_TARGET_COVERAGE,
    map_coco_label,
    map_detector_label,
)


class RealtimePerceptionTests(unittest.TestCase):
    def test_driving_intent_target_coverage_is_complete(self) -> None:
        expected = {
            "VEHICLE", "SLOW_VEHICLE", "PEDESTRIAN", "CYCLIST", "OBSTACLE",
            "TRAFFIC_CONE", "CONSTRUCTION_ZONE", "TRAFFIC_LIGHT", "TRAFFIC_SIGN",
            "CROSSWALK", "STOP_LINE", "JUNCTION", "LANE", "ROAD", "AREA",
            "PARKING_AREA", "PARKING_SPACE", "CURB", "LANDMARK", "DESTINATION",
            "PICKUP_POINT", "DROPOFF_POINT", "ROAD_HAZARD", "COORDINATE", "UNKNOWN",
        }
        self.assertEqual(expected, set(INTENT_TARGET_COVERAGE))

    def test_coco_mapping_keeps_safety_classes_only(self) -> None:
        self.assertEqual(("pedestrian", "person"), map_coco_label("person"))
        self.assertEqual(("traffic_light", "traffic_light"), map_coco_label("traffic light"))
        self.assertIsNone(map_coco_label("cat"))
        self.assertEqual(("traffic_cone", "traffic_cone"), map_detector_label("traffic_cone"))
        self.assertEqual(("road_barrier", "road_barrier"), map_detector_label("road_barrier"))
        self.assertIsNone(map_detector_label("cat"))

    def test_carla_map_lane_facts_are_safety_eligible(self) -> None:
        state = {
            "source": "carla",
            "ego": {
                "road_id": 4, "section_id": 1, "lane_id": -1, "lane_type": "driving",
                "lane_change": "left", "is_junction": False,
                "adjacent_lanes": {
                    "left": {"road_id": 4, "lane_id": -2, "lane_type": "driving", "is_junction": False},
                    "right": None,
                },
            },
            "environment": {"is_intersection": False},
        }
        road = road_structure_from_world_state(state)
        self.assertTrue(road["safety_eligible"])
        self.assertTrue(road["adjacent_lanes"]["left"]["exists"])
        self.assertTrue(road["adjacent_lanes"]["left"]["change_legal"])
        self.assertFalse(road["adjacent_lanes"]["right"]["exists"])
        self.assertEqual("unknown", road["adjacent_lanes"]["left"]["dynamic_safe"])

    def test_composite_backend_merges_overlapping_same_class_boxes(self) -> None:
        low = Detection((0, 0, 100, 100), 0.6, "vehicle", "vehicle", 2)
        high = Detection((2, 2, 98, 98), 0.9, "vehicle", "car", 2)
        pedestrian = Detection((2, 2, 98, 98), 0.7, "pedestrian", "person", 0)
        self.assertGreater(box_iou(low.bbox_xyxy, high.bbox_xyxy), 0.9)
        merged = class_aware_nms([low, high, pedestrian], threshold=0.55)
        self.assertEqual([high, pedestrian], merged)

    def test_infrastructure_tiles_overlap_upper_scene(self) -> None:
        self.assertEqual(
            [(0, 0, 480, 420), (320, 0, 800, 420)],
            infrastructure_tile_regions(800, 600),
        )
        self.assertEqual(
            (321.0, 2.0, 330.0, 12.0),
            offset_box([1, 2, 10, 12], 320, 0),
        )

    def test_bdd_taxonomy_and_box_normalization(self) -> None:
        self.assertEqual("vehicle", BDD_TO_DRIVING["car"])
        self.assertEqual("pedestrian", BDD_TO_DRIVING["person"])
        self.assertEqual("traffic_light", BDD_TO_DRIVING["traffic light"])
        self.assertEqual([0.1, 0.2, 0.5, 0.8], normalized_box([128, 144, 640, 576], 1280, 720))
        self.assertEqual("other", normalize_source("BDD100K"))
        self.assertEqual("nuscenes", normalize_source("nuScenes"))
        self.assertEqual("vehicle", normalize_category("vehicle"))
        self.assertEqual("vehicle", normalize_category("truck"))
        self.assertIsNone(normalize_category("unknown-object"))

    def test_category_thresholds_load_calibration_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(
                '{"by_category": {"pedestrian": 0.08, "vehicle": 0.2}}',
                encoding="utf-8",
            )
            self.assertEqual(
                {"pedestrian": 0.08, "vehicle": 0.2},
                load_category_thresholds(path),
            )

    def test_object_detector_can_use_independent_resolution(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_dataset",
                "--manifest",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--summary",
                "summary.json",
                "--image-size",
                "640",
                "--object-image-size",
                "768",
                "--infrastructure-tiles",
            ],
        ):
            args = parse_args()
        self.assertEqual(640, args.image_size)
        self.assertEqual(768, args.object_image_size)
        self.assertTrue(args.infrastructure_tiles)

    def test_carla_normalized_ground_truth_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "carla.jsonl"
            path.write_text(
                '{"frame_id":"carla_1","source":"carla",'
                '"ground_truth_objects":[{"category":"pedestrian",'
                '"bbox_2d":[0.1,0.2,0.3,0.8]}]}\n',
                encoding="utf-8",
            )
            truth, dataset = load_truth(path, limit=None)
        self.assertEqual("carla", dataset)
        self.assertEqual(
            [{"category": "pedestrian", "bbox_2d": [0.1, 0.2, 0.3, 0.8]}],
            truth["carla_1"],
        )


if __name__ == "__main__":
    unittest.main()
