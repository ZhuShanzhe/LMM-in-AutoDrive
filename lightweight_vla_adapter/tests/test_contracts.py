from __future__ import annotations

import unittest

import torch

from lightweight_vla_adapter.src.contracts import (
    SensorTensorBatch,
    validate_sensor_bundle,
    validate_vla_proposal,
)
from lightweight_vla_adapter.tests.fixtures import proposal


class ContractTest(unittest.TestCase):
    def test_sensor_bundle_accepts_synchronized_multimodal_input(self):
        bundle = {
            "schema_version": "1.0.0",
            "frame_id": "carla_100",
            "timestamp_s": 5.0,
            "cameras": [
                {
                    "name": "front",
                    "timestamp_s": 5.01,
                    "image_path": "front/000100.png",
                },
                {
                    "name": "left",
                    "timestamp_s": 4.99,
                    "image_path": "left/000100.png",
                },
            ],
            "lidar": {
                "timestamp_s": 5.0,
                "points_path": "lidar/000100.bin",
            },
            "ego_state": {
                "speed_mps": 5.0,
                "acceleration_mps2": 0.0,
                "yaw_rate_rps": 0.0,
            },
            "candidate_entities": [{"entity_id": "vehicle_front"}],
            "feature_refs": {"camera_bev": None, "lidar_bev": None},
        }
        self.assertEqual(validate_sensor_bundle(bundle), [])

    def test_sensor_bundle_rejects_unsynchronized_camera(self):
        bundle = {
            "schema_version": "1.0.0",
            "frame_id": "carla_100",
            "timestamp_s": 5.0,
            "cameras": [
                {
                    "name": "front",
                    "timestamp_s": 5.5,
                    "image_path": "front.png",
                }
            ],
            "lidar": {"timestamp_s": 5.0, "points_path": "points.bin"},
            "ego_state": {
                "speed_mps": 0.0,
                "acceleration_mps2": 0.0,
                "yaw_rate_rps": 0.0,
            },
            "candidate_entities": [],
            "feature_refs": {"camera_bev": None, "lidar_bev": None},
        }
        errors = validate_sensor_bundle(bundle)
        self.assertTrue(any("max_skew" in error for error in errors))

    def test_tensor_batch_validates_shapes(self):
        batch = SensorTensorBatch(
            camera_bev=torch.zeros(1, 8, 32, 32),
            lidar_bev=torch.zeros(1, 4, 32, 32),
            ego_features=torch.zeros(1, 8),
            candidate_features=torch.zeros(1, 4, 12),
            candidate_mask=torch.ones(1, 4, dtype=torch.bool),
            intent_tokens=torch.zeros(1, 6, 256),
            intent_mask=torch.ones(1, 6, dtype=torch.bool),
        )
        batch.validate()

    def test_proposal_contract(self):
        self.assertEqual(validate_vla_proposal(proposal()), [])


if __name__ == "__main__":
    unittest.main()
