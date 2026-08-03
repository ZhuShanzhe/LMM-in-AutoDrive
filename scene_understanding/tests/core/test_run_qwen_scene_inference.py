import json
import tempfile
import unittest
from pathlib import Path

from scene_understanding.core.run_qwen_scene_inference import (
    append_jsonl,
    build_conversation,
    completed_frame_ids,
    evaluate_output,
    read_jsonl,
    select_frame_indices,
    select_records,
)
from scene_understanding.core.normalize_scene_output import normalize_scene_output


VALID_OUTPUT = {
    "schema_version": "1.0",
    "frame_id": "frame_001",
    "source": "nuscenes",
    "camera_name": "CAM_FRONT",
    "scene": {
        "summary": "A clear urban road.",
        "road_type": "urban",
        "is_intersection": False,
        "weather": "clear",
        "visibility": "good",
        "traffic_light_state": "not_visible",
        "left_lane_marking": "dashed",
        "right_lane_marking": "solid",
    },
    "objects": [],
    "potential_hazards": [],
}


class QwenSceneInferenceTests(unittest.TestCase):
    def make_record(self, image_path: Path, frame_id: str = "frame_001"):
        return {
            "frame_id": frame_id,
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
            "image_path": str(image_path),
            "prompt": "Return JSON.",
        }

    def test_read_manifest_and_build_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "front.jpg"
            image.write_bytes(b"test-image-placeholder")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(self.make_record(image)) + "\n",
                encoding="utf-8",
            )

            records = read_jsonl(manifest)
            conversation = build_conversation(records[0])
            self.assertEqual(conversation[0]["content"][0], {"type": "image", "path": str(image)})
            self.assertEqual(conversation[0]["content"][1]["text"], "Return JSON.")

    def test_select_records_skips_completed_and_applies_limit(self):
        records = [
            {"frame_id": "a"},
            {"frame_id": "b"},
            {"frame_id": "c"},
        ]
        selected = select_records(records, completed={"a"}, limit=1)
        self.assertEqual([record["frame_id"] for record in selected], ["b"])

    def test_selects_one_based_frame_indices(self):
        records = [{"frame_id": "a"}, {"frame_id": "b"}, {"frame_id": "c"}]
        selected = select_frame_indices(records, [3, 1])
        self.assertEqual([record["frame_id"] for record in selected], ["c", "a"])
        with self.assertRaises(ValueError):
            select_frame_indices(records, [4])

    def test_evaluate_valid_fenced_json(self):
        record = {
            "frame_id": "frame_001",
            "source": "nuscenes",
            "camera_name": "CAM_FRONT",
        }
        raw_output = "```json\n" + json.dumps(VALID_OUTPUT) + "\n```"
        parsed, errors = evaluate_output(record, raw_output)
        self.assertEqual(parsed, VALID_OUTPUT)
        self.assertEqual(errors, [])

    def test_append_and_resume_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            append_jsonl(output, {"frame_id": "frame_001", "status": "valid"})
            append_jsonl(output, {"frame_id": "frame_002", "status": "invalid"})
            self.assertEqual(completed_frame_ids(output), {"frame_001", "frame_002"})

    def test_normalizes_compact_qwen_objects_with_audit_trail(self):
        compact = dict(VALID_OUTPUT)
        compact["objects"] = [
            {
                "bbox_2d": [494, 0, 534, 80],
                "label": "traffic_light",
                "motion_state": "moving_away",
                "confidence": 1.0,
            }
        ]
        compact["potential_hazards"] = [
            {"type": "wet_pavement", "description": "Wet road surface."}
        ]

        normalized, actions = normalize_scene_output(
            compact,
            processed_width=1176,
            processed_height=672,
        )

        obj = normalized["objects"][0]
        self.assertEqual(obj["object_id"], "vlm_obj_001")
        self.assertEqual(obj["category"], "traffic_light")
        self.assertNotIn("label", obj)
        self.assertEqual(obj["bbox_2d"], [0.420068, 0.0, 0.454082, 0.119048])
        self.assertEqual(obj["motion_state"], "unknown")
        self.assertTrue(any("fixed-infrastructure" in action for action in actions))
        self.assertTrue(any("normalized bbox" in action for action in actions))
        self.assertEqual(normalized["potential_hazards"], [])
        self.assertTrue(any("dropped incomplete" in action for action in actions))
        self.assertEqual(
            evaluate_output(
                {"frame_id": "frame_001", "source": "nuscenes", "camera_name": "CAM_FRONT"},
                json.dumps(normalized),
            )[1],
            [],
        )

    def test_corrects_misspelled_object_id(self):
        compact = dict(VALID_OUTPUT)
        compact["objects"] = [
            {
                "object_id": "vml_obj_001",
                "category": "vehicle",
                "subtype": "car",
                "color": "unknown",
                "bbox_2d": [0.1, 0.1, 0.2, 0.2],
                "relative_position": "front",
                "lane_relation": "ego_lane",
                "motion_state": "unknown",
                "distance_level": "far",
                "occlusion": "none",
                "confidence": 0.8,
            }
        ]
        normalized, actions = normalize_scene_output(compact)
        self.assertEqual(normalized["objects"][0]["object_id"], "vlm_obj_001")
        self.assertTrue(any("deterministic object_id" in action for action in actions))

    def test_downgrades_ungrounded_traffic_light_state(self):
        compact = json.loads(json.dumps(VALID_OUTPUT))
        compact["scene"]["traffic_light_state"] = "red"

        normalized, actions = normalize_scene_output(compact)

        self.assertEqual(normalized["scene"]["traffic_light_state"], "unknown")
        self.assertTrue(any("ungrounded traffic_light_state" in action for action in actions))
        self.assertEqual(
            evaluate_output(
                {"frame_id": "frame_001", "source": "nuscenes", "camera_name": "CAM_FRONT"},
                json.dumps(normalized),
            )[1],
            [],
        )


if __name__ == "__main__":
    unittest.main()
