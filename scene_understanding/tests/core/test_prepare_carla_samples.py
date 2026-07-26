from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scene_understanding.core.prepare_carla_samples import (
    build_manifest,
    build_manifest_record,
    validate_projection_record,
    write_capture_bundle,
)


EXAMPLE = Path("scene_understanding/schemas/examples/world_state.example.json")


def _world_state():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _projection():
    return {
        "schema_version": "1.0",
        "frame_id": "carla_000123",
        "camera_name": "front_rgb",
        "image_width": 800,
        "image_height": 600,
        "objects": [
            {
                "world_object_id": "carla_actor_42",
                "source_object_id": "42",
                "category": "vehicle",
                "bbox_2d": [0.4, 0.35, 0.6, 0.7],
            }
        ],
    }


class PrepareCarlaSamplesTests(unittest.TestCase):
    def test_projection_validation_rejects_frame_mismatch(self):
        projection = _projection()
        errors = validate_projection_record(
            projection,
            frame_id="another_frame",
            camera_name="front_rgb",
        )
        self.assertTrue(any("frame_id" in error for error in errors))

    def test_capture_bundle_and_manifest_are_frame_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            image.write_bytes(b"image")
            capture = write_capture_bundle(
                root / "capture",
                camera_record={
                    "frame": 123,
                    "camera_name": "front_rgb",
                    "image_path": str(image),
                },
                world_state=_world_state(),
                projection_record=_projection(),
            )
            record = build_manifest_record(
                capture,
                base_dir=root,
                prompt_template="frame={frame_id}; source={source}; camera={camera_name}",
            )
            self.assertEqual(record["frame_id"], "carla_000123")
            self.assertEqual(record["source"], "carla")
            self.assertIn("frame=carla_000123", record["prompt"])
            self.assertEqual(
                record["ground_truth_objects"][0]["world_object_id"],
                "carla_actor_42",
            )

    def test_capture_bundle_rejects_unsynchronized_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.png"
            image.write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "simulation_frame"):
                write_capture_bundle(
                    Path(temporary) / "capture",
                    camera_record={
                        "frame": 124,
                        "camera_name": "front_rgb",
                        "image_path": str(image),
                    },
                    world_state=_world_state(),
                    projection_record=_projection(),
                )

    def test_manifest_rejects_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            image.write_bytes(b"image")
            capture = write_capture_bundle(
                root / "capture",
                camera_record={
                    "frame": 123,
                    "camera_name": "front_rgb",
                    "image_path": str(image),
                },
                world_state=deepcopy(_world_state()),
                projection_record=deepcopy(_projection()),
            )
            with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
                build_manifest(
                    [capture, capture],
                    base_dir=root,
                    prompt_template="{frame_id} {source} {camera_name}",
                )


if __name__ == "__main__":
    unittest.main()
