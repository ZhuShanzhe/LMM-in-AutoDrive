import copy
import json
import tempfile
import unittest
from pathlib import Path

from scene_understanding.core.reprocess_scene_results import reprocess_file, reprocess_record
from scene_understanding.tests.core.test_run_qwen_scene_inference import VALID_OUTPUT


class SceneResultReprocessingTests(unittest.TestCase):
    def test_reprocesses_raw_output_without_model_inference(self):
        raw = copy.deepcopy(VALID_OUTPUT)
        raw["objects"] = [
            {
                "object_id": "vml_obj_001",
                "category": "vehicle",
                "subtype": "car",
                "color": "unknown",
                "bbox_2d": [100, 100, 200, 200],
                "relative_position": "front",
                "lane_relation": "ego_lane",
                "motion_state": "unknown",
                "distance_level": "far",
                "occlusion": "none",
                "confidence": 0.8,
            }
        ]
        raw["potential_hazards"] = [{"type": "unknown", "description": "Unknown hazard."}]
        record = {
            "frame_id": "frame_001",
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
            "status": "invalid",
            "processed_image_size": {"width": 1000, "height": 500},
            "raw_parsed_output": raw,
        }
        updated = reprocess_record(record)
        self.assertEqual(updated["previous_status"], "invalid")
        self.assertEqual(updated["status"], "valid")
        self.assertEqual(updated["parsed_output"]["objects"][0]["object_id"], "vlm_obj_001")
        self.assertEqual(updated["parsed_output"]["objects"][0]["bbox_2d"], [0.1, 0.2, 0.2, 0.4])
        self.assertEqual(updated["parsed_output"]["potential_hazards"], [])

    def test_reprocess_file_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            source.write_text("", encoding="utf-8")
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reprocess_file(source, output)


if __name__ == "__main__":
    unittest.main()
