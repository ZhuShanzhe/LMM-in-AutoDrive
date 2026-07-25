import json
import tempfile
import unittest
from pathlib import Path

from scene_understanding.core.prepare_nuscenes_samples import (
    build_manifest,
    normalize_category,
    render_prompt,
    resolve_drivelm_image_path,
    write_jsonl,
)


ANNOTATIONS = {
    "scene_a": {
        "scene_description": "The ego vehicle follows an urban road.",
        "key_frames": {
            "frame_a": {
                "key_object_infos": {
                    "<c1,CAM_FRONT,800.0,450.0>": {
                        "Category": "Vehicle",
                        "Status": "Moving",
                        "Visual_description": "Red car.",
                        "2d_bbox": [640.0, 300.0, 960.0, 700.0],
                    },
                    "<c2,CAM_FRONT_LEFT,800.0,450.0>": {
                        "Category": "Vehicle",
                        "Status": "Moving",
                        "Visual_description": "Blue car.",
                        "2d_bbox": [640.0, 300.0, 960.0, 700.0],
                    },
                },
                "QA": {
                    "perception": [{"Q": "q", "A": "a"}],
                    "prediction": [],
                    "planning": [],
                    "behavior": [],
                },
                "image_paths": {
                    "CAM_FRONT": "../nuscenes/samples/CAM_FRONT/front.jpg",
                    "CAM_FRONT_LEFT": "../nuscenes/samples/CAM_FRONT_LEFT/left.jpg",
                },
            }
        },
    }
}


class NuScenesSamplePreparationTests(unittest.TestCase):
    def test_render_prompt_replaces_only_metadata(self):
        template = '{"frame": "{frame_id}", "source": "{source}", "camera": "{camera_name}"}'
        rendered = render_prompt(
            template,
            frame_id="scene_a_frame_a",
            source="nuscenes",
            camera_name="CAM_FRONT",
        )
        self.assertIn('"frame": "scene_a_frame_a"', rendered)
        self.assertNotIn("{frame_id}", rendered)

    def test_build_manifest_filters_to_front_camera(self):
        records = build_manifest(
            ANNOTATIONS,
            image_root=Path("/tmp/drivelm"),
            prompt_template="frame={frame_id};source={source};camera={camera_name}",
            require_images=False,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame_id"], "scene_a_frame_a")
        self.assertEqual(len(records[0]["ground_truth_objects"]), 1)
        truth_object = records[0]["ground_truth_objects"][0]
        self.assertEqual(truth_object["object_tag"], "<c1,CAM_FRONT,800.0,450.0>")
        self.assertEqual(truth_object["bbox_2d"], [0.4, 0.333333, 0.6, 0.777778])
        self.assertEqual(records[0]["qa_counts"]["perception"], 1)
        self.assertEqual(len(records[0]["prompt_sha256"]), 64)

    def test_resolves_official_relative_image_path(self):
        resolved = resolve_drivelm_image_path(
            Path("/tmp/llama_adapter"),
            "../nuscenes/samples/CAM_FRONT/front.jpg",
        )
        self.assertEqual(
            resolved,
            Path("/tmp/llama_adapter/data/nuscenes/samples/CAM_FRONT/front.jpg"),
        )

    def test_normalizes_traffic_light_category(self):
        self.assertEqual(normalize_category("Traffic element", "Green light."), "traffic_light")

    def test_normalizes_no_entry_category(self):
        self.assertEqual(normalize_category("Traffic element", "No entry."), "traffic_sign")

    def test_write_jsonl_round_trip(self):
        records = build_manifest(
            ANNOTATIONS,
            image_root=Path("/tmp/drivelm"),
            prompt_template="{frame_id}",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.jsonl"
            self.assertEqual(write_jsonl(records, output), 1)
            loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
