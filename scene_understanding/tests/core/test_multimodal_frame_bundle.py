from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scene_understanding.core.multimodal_frame_bundle import (
    CAMERA_KEYS,
    IMAGE_SIZE_KEYS,
    INSTRUCTION_KEYS,
    LIDAR_KEYS,
    PROVENANCE_KEYS,
    SCHEMA_VERSION,
    SYNCHRONIZATION_KEYS,
    SYNCHRONIZATION_STATUSES,
    TOP_LEVEL_KEYS,
    WORLD_STATE_KEYS,
    validate_multimodal_frame_bundle,
)


IDENTITY_4X4 = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]

CAMERA_INTRINSICS = [
    400.0, 0.0, 400.0,
    0.0, 400.0, 300.0,
    0.0, 0.0, 1.0,
]


def camera_record(name: str) -> dict:
    return {
        "sensor_name": name,
        "frame": 123,
        "timestamp_s": 6.15,
        "image_path": f"sensors/{name}/00000123.png",
        "image_size": {
            "width": 800,
            "height": 600,
        },
        "intrinsic_matrix": list(CAMERA_INTRINSICS),
        "sensor_to_ego": list(IDENTITY_4X4),
    }


def valid_bundle() -> dict:
    return {
        "schema_version": "1.0.0",
        "bundle_id": "bundle_carla_00000123_req_1",
        "request_id": "req-1",
        "source": "carla",
        "frame_id": "carla_00000123",
        "simulation_frame": 123,
        "timestamp_s": 6.15,
        "synchronization": {
            "status": "EXACT",
            "reference_frame": 123,
            "reference_timestamp_s": 6.15,
            "tolerance_ms": 50.0,
            "max_skew_ms": 0.0,
            "required_modalities": [
                "instruction",
                "front_rgb",
                "left_rgb",
                "right_rgb",
                "rear_rgb",
                "lidar",
                "world_state",
            ],
            "missing_modalities": [],
        },
        "instruction": {
            "source": "asr",
            "text": "看到前方行人后减速避让，然后向左变道",
            "language": "zh-CN",
            "confidence": 0.98,
            "timestamp_s": 6.10,
        },
        "cameras": [
            camera_record("front_rgb"),
            camera_record("left_rgb"),
            camera_record("right_rgb"),
            camera_record("rear_rgb"),
        ],
        "lidar": {
            "sensor_name": "roof_lidar",
            "frame": 123,
            "timestamp_s": 6.15,
            "point_cloud_path": (
                "sensors/roof_lidar/00000123.ply"
            ),
            "point_count": 64000,
            "coordinate_frame": "sensor",
            "sensor_to_ego": list(IDENTITY_4X4),
        },
        "world_state": {
            "frame_id": "carla_00000123",
            "simulation_frame": 123,
            "timestamp_s": 6.15,
            "path": "world_states/carla_00000123.json",
        },
        "provenance": {
            "capture_module": (
                "carla_multimodal_sensor_manager"
            ),
            "metric_source": "carla_actor_api",
        },
    }


class MultimodalFrameBundleTests(unittest.TestCase):
    def test_accepts_exact_scene_two_bundle(self):
        errors = validate_multimodal_frame_bundle(
            valid_bundle()
        )
        self.assertEqual(errors, [])

    def test_rejects_exact_camera_frame_mismatch(self):
        bundle = valid_bundle()
        bundle["cameras"][0]["frame"] = 122

        errors = validate_multimodal_frame_bundle(bundle)

        self.assertIn(
            (
                "$.cameras[0].frame: must equal "
                "synchronization.reference_frame"
            ),
            errors,
        )

    def test_rejects_timestamp_skew_over_tolerance(self):
        bundle = valid_bundle()
        bundle["cameras"][1]["timestamp_s"] = 6.30

        errors = validate_multimodal_frame_bundle(bundle)

        self.assertTrue(
            any(
                error.startswith(
                    "$.synchronization.max_skew_ms:"
                )
                for error in errors
            ),
            errors,
        )

    def test_rejects_duplicate_camera_names(self):
        bundle = valid_bundle()
        bundle["cameras"][1]["sensor_name"] = (
            "front_rgb"
        )

        errors = validate_multimodal_frame_bundle(bundle)

        self.assertIn(
            (
                "$.cameras: duplicate sensor_name "
                "'front_rgb'"
            ),
            errors,
        )

    def test_incomplete_bundle_declares_missing_lidar(self):
        bundle = valid_bundle()
        bundle["lidar"] = None
        bundle["synchronization"]["status"] = (
            "INCOMPLETE"
        )
        bundle["synchronization"][
            "missing_modalities"
        ] = ["lidar"]

        errors = validate_multimodal_frame_bundle(bundle)

        self.assertEqual(errors, [])

    def test_rejects_undeclared_missing_lidar(self):
        bundle = valid_bundle()
        bundle["lidar"] = None
        bundle["synchronization"]["status"] = (
            "INCOMPLETE"
        )

        errors = validate_multimodal_frame_bundle(bundle)

        self.assertIn(
            (
                "$.synchronization.missing_modalities: "
                "must contain 'lidar' when the "
                "required modality is unavailable"
            ),
            errors,
        )


    def test_checked_in_example_is_valid(self):
        schema_dir = (
            Path(__file__).resolve().parents[2]
            / "schemas"
        )
        example = json.loads(
            (
                schema_dir
                / "examples"
                / "multimodal_frame_bundle.example.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            validate_multimodal_frame_bundle(example),
            [],
        )

    def test_checked_in_example_is_fresh(self):
        schema_dir = (
            Path(__file__).resolve().parents[2]
            / "schemas"
        )
        example = json.loads(
            (
                schema_dir
                / "examples"
                / "multimodal_frame_bundle.example.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            example,
            valid_bundle(),
        )

    def test_json_schema_tracks_python_contract(self):
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "multimodal_frame_bundle.schema.json"
        )
        schema = json.loads(
            schema_path.read_text(encoding="utf-8")
        )

        self.assertEqual(
            schema["properties"]["schema_version"][
                "const"
            ],
            SCHEMA_VERSION,
        )
        self.assertEqual(
            set(schema["required"]),
            TOP_LEVEL_KEYS,
        )
        self.assertEqual(
            set(schema["properties"]),
            TOP_LEVEL_KEYS,
        )

        definitions = schema["$defs"]

        expected_required = {
            "synchronization": (
                SYNCHRONIZATION_KEYS
            ),
            "instruction": INSTRUCTION_KEYS,
            "camera": CAMERA_KEYS,
            "imageSize": IMAGE_SIZE_KEYS,
            "lidar": LIDAR_KEYS,
            "worldStateReference": (
                WORLD_STATE_KEYS
            ),
            "provenance": PROVENANCE_KEYS,
        }

        for name, expected in expected_required.items():
            with self.subTest(definition=name):
                self.assertEqual(
                    set(
                        definitions[name]["required"]
                    ),
                    expected,
                )
                self.assertEqual(
                    set(
                        definitions[name][
                            "properties"
                        ]
                    ),
                    expected,
                )

        self.assertEqual(
            set(
                definitions["synchronization"][
                    "properties"
                ]["status"]["enum"]
            ),
            SYNCHRONIZATION_STATUSES,
        )


class MultimodalFrameBundleBuilderTests(
    unittest.TestCase
):
    def _example(self):
        import json
        from pathlib import Path

        scene_root = Path(__file__).resolve().parents[2]
        path = (
            scene_root
            / "schemas"
            / "examples"
            / "multimodal_frame_bundle.example.json"
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _snapshot(self, example):
        from copy import deepcopy

        return {
            "simulation_frame": example[
                "simulation_frame"
            ],
            "complete": True,
            "missing_modalities": [],
            "cameras": deepcopy(example["cameras"]),
            "lidar": deepcopy(example["lidar"]),
        }

    def _build(self, example, snapshot):
        from copy import deepcopy

        from scene_understanding.core.multimodal_frame_bundle import (
            build_multimodal_frame_bundle,
        )

        return build_multimodal_frame_bundle(
            bundle_id=example["bundle_id"],
            request_id=example["request_id"],
            source=example["source"],
            frame_id=example["frame_id"],
            simulation_frame=example[
                "simulation_frame"
            ],
            timestamp_s=example["timestamp_s"],
            instruction=deepcopy(
                example["instruction"]
            ),
            sensor_snapshot=snapshot,
            world_state=deepcopy(
                example["world_state"]
            ),
            provenance=deepcopy(
                example["provenance"]
            ),
            required_modalities=example[
                "synchronization"
            ]["required_modalities"],
            tolerance_ms=example[
                "synchronization"
            ]["tolerance_ms"],
        )

    def test_builds_checked_in_exact_bundle(self):
        example = self._example()
        snapshot = self._snapshot(example)

        bundle = self._build(example, snapshot)

        self.assertEqual(bundle, example)

    def test_builds_within_tolerance_bundle(self):
        example = self._example()
        snapshot = self._snapshot(example)

        snapshot["cameras"][0]["timestamp_s"] = (
            example["timestamp_s"] + 0.02
        )

        bundle = self._build(example, snapshot)

        self.assertEqual(
            bundle["synchronization"]["status"],
            "WITHIN_TOLERANCE",
        )
        self.assertAlmostEqual(
            bundle["synchronization"]["max_skew_ms"],
            20.0,
        )

        errors = validate_multimodal_frame_bundle(
            bundle
        )
        self.assertEqual(errors, [])

    def test_builds_incomplete_bundle(self):
        example = self._example()
        snapshot = self._snapshot(example)

        snapshot["cameras"] = [
            camera
            for camera in snapshot["cameras"]
            if camera["sensor_name"] != "left_rgb"
        ]
        snapshot["lidar"] = None
        snapshot["complete"] = False
        snapshot["missing_modalities"] = [
            "left_rgb",
            "lidar",
        ]

        bundle = self._build(example, snapshot)

        self.assertEqual(
            bundle["synchronization"]["status"],
            "INCOMPLETE",
        )
        self.assertEqual(
            bundle["synchronization"][
                "missing_modalities"
            ],
            ["left_rgb", "lidar"],
        )

        errors = validate_multimodal_frame_bundle(
            bundle
        )
        self.assertEqual(errors, [])

    def test_rejects_timestamp_skew_over_tolerance(self):
        example = self._example()
        snapshot = self._snapshot(example)

        snapshot["cameras"][0]["timestamp_s"] = (
            example["timestamp_s"] + 0.10
        )

        with self.assertRaisesRegex(
            ValueError,
            "timestamp skew",
        ):
            self._build(example, snapshot)

    def test_rejects_sensor_from_different_frame(self):
        example = self._example()
        snapshot = self._snapshot(example)

        snapshot["cameras"][0]["frame"] = 124

        with self.assertRaisesRegex(
            ValueError,
            "frame",
        ):
            self._build(example, snapshot)


if __name__ == "__main__":
    unittest.main()
