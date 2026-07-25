import copy
import json
import unittest
from pathlib import Path

from scene_understanding.core.validate_scene_output import parse_json_text, validate_scene_output


EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "schemas" / "examples" / "scene_understanding.example.json"


class SceneOutputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_record = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

    def test_example_is_valid(self):
        errors = validate_scene_output(
            self.valid_record,
            expected_frame_id="nuscenes_sample_000001",
            expected_source="nuscenes",
            expected_camera_name="CAM_FRONT",
        )
        self.assertEqual(errors, [])

    def test_rejects_invalid_bbox_order(self):
        record = copy.deepcopy(self.valid_record)
        record["objects"][0]["bbox_2d"] = [0.8, 0.2, 0.3, 0.7]
        errors = validate_scene_output(record)
        self.assertTrue(any("x_min must be smaller" in error for error in errors))

    def test_rejects_unknown_hazard_object(self):
        record = copy.deepcopy(self.valid_record)
        record["potential_hazards"][0]["related_object_ids"] = ["vlm_obj_999"]
        errors = validate_scene_output(record)
        self.assertTrue(any("unknown IDs: vlm_obj_999" in error for error in errors))

    def test_rejects_unexpected_fields(self):
        record = copy.deepcopy(self.valid_record)
        record["ttc_s"] = 2.5
        errors = validate_scene_output(record)
        self.assertTrue(any("unexpected fields: ttc_s" in error for error in errors))

    def test_parses_one_outer_json_fence(self):
        parsed = parse_json_text("```json\n" + json.dumps(self.valid_record) + "\n```")
        self.assertEqual(parsed, self.valid_record)

    def test_rejects_zero_confidence_placeholder(self):
        record = copy.deepcopy(self.valid_record)
        record["objects"][0]["confidence"] = 0
        errors = validate_scene_output(record)
        self.assertTrue(any("must be greater than 0" in error for error in errors))

    def test_rejects_motion_for_fixed_traffic_light(self):
        record = copy.deepcopy(self.valid_record)
        record["objects"][0]["category"] = "traffic_light"
        record["objects"][0]["motion_state"] = "moving_away"
        errors = validate_scene_output(record)
        self.assertTrue(any("fixed infrastructure" in error for error in errors))

    def test_requires_visible_light_state_to_have_grounded_object(self):
        record = copy.deepcopy(self.valid_record)
        record["scene"]["traffic_light_state"] = "red"
        errors = validate_scene_output(record)
        self.assertTrue(any("must be grounded" in error for error in errors))

    def test_allows_null_object_confidence(self):
        record = copy.deepcopy(self.valid_record)
        record["objects"][0]["confidence"] = None
        errors = validate_scene_output(record)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
