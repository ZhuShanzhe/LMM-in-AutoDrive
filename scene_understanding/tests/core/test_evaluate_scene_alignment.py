import unittest

from scene_understanding.core.evaluate_scene_alignment import bbox_iou, evaluate_frame, summarize


class SceneAlignmentEvaluationTests(unittest.TestCase):
    def test_grouped_truth_supports_two_tight_predictions(self):
        manifest = {
            "frame_id": "frame_001",
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
            "ground_truth_objects": [
                {
                    "category": "traffic_light",
                    "visual_description": "Green light.",
                    "status_raw": None,
                    "bbox_2d": [0.42, 0.0, 0.91, 0.20],
                }
            ],
        }
        output = {
            "schema_version": "1.0",
            "frame_id": "frame_001",
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
            "scene": {
                "summary": "Two green traffic lights.",
                "road_type": "urban",
                "is_intersection": True,
                "weather": "rain",
                "visibility": "reduced",
                "traffic_light_state": "green",
                "left_lane_marking": "unknown",
                "right_lane_marking": "unknown",
            },
            "objects": [
                self.make_light("vlm_obj_001", [0.43, 0.01, 0.46, 0.12]),
                self.make_light("vlm_obj_002", [0.63, 0.01, 0.67, 0.14]),
            ],
            "potential_hazards": [],
        }
        result = evaluate_frame(manifest, {"parsed_output": output})
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["truth_center_hits"], 1)
        self.assertEqual(result["supported_predictions"], 2)
        self.assertTrue(result["traffic_light_state_correct"])

        summary = summarize([result])
        self.assertEqual(summary["category_center_hit_truth_recall"], 1.0)
        self.assertEqual(summary["category_center_supported_prediction_rate"], 1.0)
        self.assertEqual(summary["traffic_light_state_accuracy"], 1.0)
        self.assertEqual(summary["categories"]["traffic_light"]["truth_recall"], 1.0)
        self.assertEqual(
            summary["categories"]["traffic_light"]["supported_prediction_rate"], 1.0
        )

    def test_bbox_iou_is_zero_for_disjoint_boxes(self):
        self.assertEqual(bbox_iou([0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]), 0.0)

    @staticmethod
    def make_light(object_id, bbox):
        return {
            "object_id": object_id,
            "category": "traffic_light",
            "subtype": "unknown",
            "color": "green",
            "bbox_2d": bbox,
            "relative_position": "unknown",
            "lane_relation": "unknown",
            "motion_state": "unknown",
            "distance_level": "near",
            "occlusion": "none",
            "confidence": 1.0,
        }


if __name__ == "__main__":
    unittest.main()
