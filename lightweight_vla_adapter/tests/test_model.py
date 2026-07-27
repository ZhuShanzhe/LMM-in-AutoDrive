from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from lightweight_vla_adapter.src.decision_adapter import (
    ACTION_LABELS,
    LightweightDecisionAdapter,
    decode_proposal,
)
from lightweight_vla_adapter.src.intent_encoder import ModernBertIntentEncoder
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer
from lightweight_vla_adapter.tests.fixtures import integration_documents


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ModelTest(unittest.TestCase):
    def test_modernbert_intent_encoder_projects_token_features(self):
        encoder = ModernBertIntentEncoder(
            TinyBackbone(),
            input_hidden_size=32,
            output_hidden_size=16,
            freeze_backbone=True,
        )
        tokens, mask = encoder(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 0]]),
        )
        self.assertEqual(tokens.shape, (1, 3, 16))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(next(encoder.backbone.parameters()).requires_grad)

    def test_four_layer_adapter_shapes_and_decoding(self):
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
            bev_grid=(4, 4),
        ).eval()
        output = model(
            camera_bev=torch.randn(2, 8, 16, 16),
            lidar_bev=torch.randn(2, 4, 16, 16),
            ego_features=torch.randn(2, 8),
            candidate_features=torch.randn(2, 3, 12),
            candidate_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
            intent_tokens=torch.randn(2, 5, 16),
            intent_mask=torch.tensor(
                [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
            ),
        )
        self.assertEqual(output.action_logits.shape, (2, len(ACTION_LABELS)))
        self.assertEqual(output.target_pointer_logits.shape, (2, 3))
        proposals = decode_proposal(
            output,
            request_id="request",
            frame_id="frame",
            candidate_entity_ids=[["a", "b"], ["c", "d", "e"]],
            model_name="student-test",
            latency_ms=1.2,
        )
        self.assertEqual(len(proposals), 2)

    def test_structured_carla_proxy_builds_model_inputs(self):
        _, world_state, _, _ = integration_documents()
        rasterizer = StructuredBEVRasterizer(height=16, width=16)
        batch, entity_ids = rasterizer.build(
            world_state,
            intent_tokens=torch.randn(1, 5, 16),
            intent_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        self.assertEqual(batch.camera_bev.shape, (1, 8, 16, 16))
        self.assertEqual(batch.lidar_bev.shape, (1, 4, 16, 16))
        self.assertEqual(entity_ids, [["vehicle_front"]])

    def test_adapter_rejects_non_deployment_depth(self):
        with self.assertRaisesRegex(ValueError, "between 4 and 6"):
            LightweightDecisionAdapter(
                camera_channels=8,
                lidar_channels=4,
                candidate_dim=12,
                ego_dim=8,
                intent_dim=16,
                hidden_size=32,
                num_layers=3,
                num_heads=4,
            )

    def test_pipeline_rejects_random_initialization_for_decisions(self):
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
            bev_grid=(4, 4),
        )
        pipeline = LightweightVLAPipeline(model, device="cpu")
        _, world_state, _, _ = integration_documents()
        batch, entity_ids = StructuredBEVRasterizer(height=16, width=16).build(
            world_state,
            intent_tokens=torch.randn(1, 5, 16),
            intent_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint is not loaded"):
            pipeline.predict_proposal(
                batch,
                request_id="request",
                frame_id=world_state["frame_id"],
                candidate_entity_ids=entity_ids,
            )


if __name__ == "__main__":
    unittest.main()
