from __future__ import annotations

import unittest

import torch

from lightweight_vla_adapter.src.contracts import SensorTensorBatch
from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter
from lightweight_vla_adapter.src.pipeline import (
    LightweightVLAPipeline,
    decode_visual_risk_assessment,
)
from lightweight_vla_adapter.scripts.train_scene3_multimodal import model_kwargs


def build_batch(
    with_images: bool = True, environment_dim: int = 12
) -> SensorTensorBatch:
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
        environment_features=torch.zeros(1, environment_dim),
    )


class RawMultimodalAdapterTests(unittest.TestCase):
    def test_training_contract_masks_truth_features_by_source(self):
        base = build_batch()
        batch = {
            "camera_bev": torch.ones(2, 8, 16, 16),
            "lidar_bev": torch.ones(2, 4, 16, 16),
            "ego_features": base.ego_features.repeat(2, 1),
            "candidate_features": torch.ones(2, 2, 12),
            "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
            "intent_tokens": base.intent_tokens.repeat(2, 1, 1),
            "intent_mask": base.intent_mask.repeat(2, 1),
            "camera_images": base.camera_images.repeat(2, 1, 1, 1, 1),
            "camera_view_mask": base.camera_view_mask.repeat(2, 1),
            "environment_features": base.environment_features.repeat(2, 1),
            "source_dataset": ["CARLA", "nuScenes"],
        }
        inputs = model_kwargs(
            batch,
            torch.device("cpu"),
            {
                "use_candidate_entities": False,
                "structured_sensor_sources": ["nuScenes"],
            },
        )
        self.assertEqual(int(torch.count_nonzero(inputs["camera_bev"][0])), 0)
        self.assertEqual(int(torch.count_nonzero(inputs["lidar_bev"][0])), 0)
        self.assertGreater(int(torch.count_nonzero(inputs["lidar_bev"][1])), 0)
        self.assertEqual(int(torch.count_nonzero(inputs["candidate_features"])), 0)

    def test_visual_risk_decoder_produces_sensor_safety_contract(self):
        risk = decode_visual_risk_assessment(
            torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float32)
        )
        self.assertEqual(risk["risk_level"], "high")
        self.assertEqual(risk["recommended_action"], "emergency_brake")
        self.assertFalse(risk["lane_change"]["left"]["is_safe"])
        self.assertIsNone(risk["matched_entity_id"])
        self.assertEqual(risk["source"], "learned_raw_camera_visual_risk_head")
        self.assertAlmostEqual(sum(risk["probabilities"].values()), 1.0, places=5)

    def test_ambiguous_high_argmax_is_calibrated_to_caution(self):
        probabilities = torch.tensor([[0.349096, 0.166114, 0.484790]])
        risk = decode_visual_risk_assessment(probabilities.log())
        self.assertEqual(risk["raw_argmax_level"], "high")
        self.assertEqual(risk["risk_level"], "medium")
        self.assertEqual(risk["recommended_action"], "decelerate")
        self.assertEqual(risk["high_confidence_threshold"], 0.55)

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

    def test_masked_view_risk_probe_does_not_replace_primary_risk_state(self):
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
        pipeline = LightweightVLAPipeline(
            model,
            device="cpu",
            checkpoint_loaded=True,
        )
        batch = build_batch()
        batch.camera_view_mask[:] = torch.tensor(
            [[False, True, False, False]]
        )

        risk = pipeline.predict_visual_risk(batch)

        self.assertIn(risk["risk_level"], {"low", "medium", "high"})
        with self.assertRaisesRegex(RuntimeError, "before model inference"):
            _ = pipeline.last_visual_risk_assessment

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

    def test_environment_speed_cap_is_a_hard_model_bound(self):
        model = LightweightDecisionAdapter(
            camera_channels=8,
            lidar_channels=4,
            candidate_dim=12,
            ego_dim=8,
            intent_dim=16,
            environment_dim=14,
            hidden_size=32,
            num_layers=4,
            num_heads=4,
            dropout=0.0,
            require_raw_camera=True,
            use_structured_bev=False,
            speed_cap_environment_index=13,
        ).eval()
        batch = build_batch(environment_dim=14)
        batch.environment_features[:, 13] = 0.25
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
        self.assertGreaterEqual(float(output.target_speed_kmh), 0.0)
        self.assertLessEqual(float(output.target_speed_kmh), 25.0)

    def test_visual_risk_probabilities_condition_decision_token(self):
        torch.manual_seed(11)
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
            use_structured_bev=False,
            condition_decision_on_visual_risk=True,
        ).eval()
        batch = build_batch()
        with torch.inference_mode():
            model.visual_risk_head.weight.zero_()
            model.visual_risk_head.bias.copy_(torch.tensor([12.0, 0.0, 0.0]))
            low_risk = model(
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
            model.visual_risk_head.bias.copy_(torch.tensor([0.0, 0.0, 12.0]))
            high_risk = model(
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
        self.assertFalse(torch.equal(low_risk.action_logits, high_risk.action_logits))


if __name__ == "__main__":
    unittest.main()
