from __future__ import annotations

import unittest

import torch

from lightweight_vla_adapter.src.contracts import SensorTensorBatch
from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter


def build_batch(with_images: bool = True) -> SensorTensorBatch:
    return SensorTensorBatch(
        camera_bev=torch.zeros(1, 8, 16, 16),
        lidar_bev=torch.zeros(1, 4, 16, 16),
        ego_features=torch.zeros(1, 8),
        candidate_features=torch.zeros(1, 2, 12),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        intent_tokens=torch.zeros(1, 5, 16),
        intent_mask=torch.ones(1, 5, dtype=torch.bool),
        camera_images=(
            torch.zeros(1, 4, 3, 64, 64, dtype=torch.uint8)
            if with_images
            else None
        ),
        camera_view_mask=(
            torch.ones(1, 4, dtype=torch.bool) if with_images else None
        ),
        environment_features=torch.zeros(1, 12),
    )


class RawMultimodalAdapterTests(unittest.TestCase):
    def test_sensor_contract_accepts_exact_four_view_batch(self):
        build_batch().validate()

    def test_required_raw_camera_cannot_fall_back_to_proxy(self):
        model = LightweightDecisionAdapter(
            camera_channels=8,
            lidar_channels=4,
            candidate_dim=12,
            ego_dim=8,
            intent_dim=16,
            hidden_size=32,
            num_layers=4,
            num_heads=4,
            require_raw_camera=True,
            use_structured_bev=False,
        )
        batch = build_batch(with_images=False)
        with self.assertRaisesRegex(ValueError, "requires synchronized raw camera"):
            model(
                camera_bev=batch.camera_bev,
                lidar_bev=batch.lidar_bev,
                ego_features=batch.ego_features,
                candidate_features=batch.candidate_features,
                candidate_mask=batch.candidate_mask,
                intent_tokens=batch.intent_tokens,
                intent_mask=batch.intent_mask,
                camera_images=batch.camera_images,
                camera_view_mask=batch.camera_view_mask,
                environment_features=batch.environment_features,
            )

    def test_raw_camera_forward_exposes_action_and_visual_risk_heads(self):
        model = LightweightDecisionAdapter(
            camera_channels=8,
            lidar_channels=4,
            candidate_dim=12,
            ego_dim=8,
            intent_dim=16,
            hidden_size=32,
            num_layers=4,
            num_heads=4,
            require_raw_camera=True,
            use_structured_bev=False,
        ).eval()
        batch = build_batch()
        with torch.inference_mode():
            output = model(
                camera_bev=batch.camera_bev,
                lidar_bev=batch.lidar_bev,
                ego_features=batch.ego_features,
                candidate_features=batch.candidate_features,
                candidate_mask=batch.candidate_mask,
                intent_tokens=batch.intent_tokens,
                intent_mask=batch.intent_mask,
                camera_images=batch.camera_images,
                camera_view_mask=batch.camera_view_mask,
                environment_features=batch.environment_features,
            )
        self.assertEqual(tuple(output.action_logits.shape), (1, 9))
        self.assertEqual(tuple(output.visual_risk_logits.shape), (1, 3))

    def test_disabled_candidate_entities_cannot_change_driving_outputs(self):
        torch.manual_seed(7)
        model = LightweightDecisionAdapter(
            camera_channels=8,
            lidar_channels=4,
            candidate_dim=12,
            ego_dim=8,
            intent_dim=16,
            hidden_size=32,
            num_layers=4,
            num_heads=4,
            dropout=0.0,
            require_raw_camera=True,
            use_candidate_entities=False,
            use_structured_bev=False,
        ).eval()
        first = build_batch()
        second = build_batch()
        second.candidate_features.random_(-100, 100)
        with torch.inference_mode():
            outputs = []
            for batch in (first, second):
                outputs.append(
                    model(
                        camera_bev=batch.camera_bev,
                        lidar_bev=batch.lidar_bev,
                        ego_features=batch.ego_features,
                        candidate_features=batch.candidate_features,
                        candidate_mask=batch.candidate_mask,
                        intent_tokens=batch.intent_tokens,
                        intent_mask=batch.intent_mask,
                        camera_images=batch.camera_images,
                        camera_view_mask=batch.camera_view_mask,
                        environment_features=batch.environment_features,
                    )
                )
        for field in (
            "action_logits",
            "target_speed_kmh",
            "target_lane_logits",
            "target_pointer_logits",
            "confidence_logits",
            "visual_risk_logits",
        ):
            self.assertTrue(
                torch.equal(getattr(outputs[0], field), getattr(outputs[1], field)),
                field,
            )


if __name__ == "__main__":
    unittest.main()
