import copy
import json
import unittest

from scene_understanding.core.merge_traffic_control_results import merge_record
from scene_understanding.core.run_qwen_traffic_control_grounding import (
    normalize_grounding_output,
    parse_grounding_text,
    validate_grounding_output,
)
from scene_understanding.tests.core.test_run_qwen_scene_inference import VALID_OUTPUT


class TrafficControlGroundingTests(unittest.TestCase):
    def test_extracts_json_after_short_model_prose(self):
        parsed = parse_grounding_text('Detected controls:\n{"traffic_lights": [], "traffic_signs": []}\nDone.')
        self.assertEqual(parsed, {"traffic_lights": [], "traffic_signs": []})

    def test_rejects_truncated_outer_json_instead_of_accepting_inner_object(self):
        truncated = '{"traffic_lights": [{"state": "red"}, {"state":'
        with self.assertRaises(json.JSONDecodeError):
            parse_grounding_text(truncated)

    def test_normalizes_native_qwen_grounding_list(self):
        raw = [
            {
                "bbox_2d": [500, 100, 550, 180],
                "label": "red traffic light",
            },
            {
                "bbox_2d": [300, 200, 340, 260],
                "label": "no entry sign",
            },
        ]
        parsed, actions = normalize_grounding_output(
            raw,
            frame_id="frame_001",
            source="nuscenes",
            camera_name="CAM_FRONT",
            processed_width=1000,
            processed_height=500,
        )
        self.assertEqual(len(parsed["traffic_lights"]), 1)
        self.assertEqual(parsed["traffic_lights"][0]["state"], "red")
        self.assertEqual(parsed["traffic_lights"][0]["bbox_2d"], [0.5, 0.2, 0.55, 0.36])
        self.assertIsNone(parsed["traffic_lights"][0]["confidence"])
        self.assertEqual(parsed["traffic_signs"][0]["sign_type"], "no_entry")
        self.assertTrue(any("stored null" in action for action in actions))
        record = {"frame_id": "frame_001", "source": "nuscenes", "camera_name": "CAM_FRONT"}
        self.assertEqual(validate_grounding_output(parsed, record), [])

    def test_merges_focused_light_and_updates_scene_state(self):
        base_output = copy.deepcopy(VALID_OUTPUT)
        base_record = {
            "frame_id": "frame_001",
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
            "status": "valid",
            "parsed_output": base_output,
        }
        grounding_record = {
            "frame_id": "frame_001",
            "status": "valid",
            "prompt_sha256": "abc",
            "processed_image_size": {"width": 1000, "height": 500},
            "parsed_output": {
                "traffic_lights": [
                    {
                        "grounding_id": "tc_light_001",
                        "bbox_2d": [0.4, 0.2, 0.45, 0.3],
                        "state": "red",
                        "confidence": None,
                    }
                ],
                "traffic_signs": [],
            },
        }
        merged = merge_record(base_record, grounding_record)
        self.assertEqual(merged["status"], "valid")
        self.assertEqual(merged["parsed_output"]["scene"]["traffic_light_state"], "red")
        obj = merged["parsed_output"]["objects"][0]
        self.assertEqual(obj["category"], "traffic_light")
        self.assertIsNone(obj["confidence"])
        self.assertTrue(any("added tc_light_001" in action for action in merged["traffic_control_merge_actions"]))


if __name__ == "__main__":
    unittest.main()
