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
from lightweight_vla_adapter.src.distillation import DistillationLoss
from lightweight_vla_adapter.src.intent_encoder import ModernBertIntentEncoder
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer
from lightweight_vla_adapter.src.temporal_supervisor import (
    TemporalProposalSupervisor,
    TemporalSupervisorConfig,
)
from lightweight_vla_adapter.tests.fixtures import integration_documents


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ModelTest(unittest.TestCase):
    @staticmethod
    def _proposal(
        frame: int,
        *,
        action: str = "keep_lane",
        speed: float = 30.0,
        target: str | None = "vehicle_front",
    ) -> dict:
        return {
            "schema_version": "1.0.0",
            "request_id": "request",
            "frame_id": f"frame_{frame}",
            "action": action,
            "target_speed_kmh": speed,
            "target_lane": None,
            "target_location": None,
            "target_entity_id": target,
            "confidence": 0.9,
            "model": "student-test",
            "latency_ms": 1.0,
        }

    @staticmethod
    def _world(
        frame: int,
        *,
        distance_m: float = 20.0,
        relative_vx_mps: float = 0.0,
        second_vehicle: bool = False,
    ) -> dict:
        objects = [
            {
                "entity_id": "vehicle_front",
                "category": "vehicle",
                "relative_position_m": {"x": distance_m, "y": 0.0, "z": 0.0},
                "relative_velocity_mps": {
                    "x": relative_vx_mps,
                    "y": 0.0,
                },
                "lane_relation": "same",
            }
        ]
        if second_vehicle:
            objects.append(
                {
                    "entity_id": "vehicle_second",
                    "category": "vehicle",
                    "relative_position_m": {"x": 24.0, "y": 0.0, "z": 0.0},
                    "relative_velocity_mps": {"x": 0.0, "y": 0.0},
                    "lane_relation": "same",
                }
            )
        return {
            "frame_id": f"frame_{frame}",
            "timestamp_s": frame * 0.05,
            "ego": {"speed_mps": 10.0},
            "objects": objects,
        }

    def test_temporal_supervisor_suppresses_acceleration_oscillation(self):
        supervisor = TemporalProposalSupervisor(
            TemporalSupervisorConfig(accelerate_confirm_frames=3)
        )
        safe = {"recommended_action": "keep_lane"}
        first = supervisor.stabilize(
            self._proposal(0, action="decelerate", speed=28.0),
            self._world(0),
            safe,
        )
        self.assertEqual(first["action"], "decelerate")
        for frame in (1, 2):
            held = supervisor.stabilize(
                self._proposal(frame, action="accelerate", speed=40.0),
                self._world(frame),
                safe,
            )
            self.assertEqual(held["action"], "decelerate")
            self.assertLessEqual(held["target_speed_kmh"], 28.0)
        confirmed = supervisor.stabilize(
            self._proposal(3, action="accelerate", speed=40.0),
            self._world(3),
            safe,
        )
        self.assertEqual(confirmed["action"], "accelerate")
        self.assertLess(confirmed["target_speed_kmh"], 40.0)

    def test_temporal_supervisor_constrains_acceleration_for_braking_lead(self):
        supervisor = TemporalProposalSupervisor()
        result = supervisor.stabilize(
            self._proposal(0, action="accelerate", speed=50.0),
            self._world(0, distance_m=12.0, relative_vx_mps=-4.0),
            {"recommended_action": "keep_lane"},
        )
        self.assertEqual(result["action"], "decelerate")
        self.assertLessEqual(result["target_speed_kmh"], 32.4)
        diagnostics = supervisor.diagnostics("request")
        self.assertIn(
            "closing_lead_vehicle_constraint",
            diagnostics["reasons"],
        )

    def test_temporal_supervisor_preempts_on_deterministic_risk(self):
        supervisor = TemporalProposalSupervisor()
        result = supervisor.stabilize(
            self._proposal(0, action="accelerate", speed=50.0),
            self._world(0),
            {
                "recommended_action": "emergency_brake",
                "risk_level": "high",
            },
        )
        self.assertEqual(result["action"], "emergency_brake")
        self.assertEqual(result["target_speed_kmh"], 0.0)

    def test_conservative_transition_from_lateral_action_is_immediate(self):
        supervisor = TemporalProposalSupervisor()
        safe = {"recommended_action": "keep_lane"}
        first = supervisor.stabilize(
            self._proposal(0, action="lane_change_left"),
            self._world(0),
            safe,
        )
        self.assertEqual(first["action"], "lane_change_left")
        second = supervisor.stabilize(
            self._proposal(1, action="decelerate", speed=24.0),
            self._world(1),
            safe,
        )
        self.assertEqual(second["action"], "decelerate")

    def test_temporal_supervisor_requires_stable_target_switch(self):
        supervisor = TemporalProposalSupervisor(
            TemporalSupervisorConfig(target_switch_confirm_frames=3)
        )
        safe = {"recommended_action": "keep_lane"}
        first = supervisor.stabilize(
            self._proposal(0),
            self._world(0, second_vehicle=True),
            safe,
        )
        self.assertEqual(first["target_entity_id"], "vehicle_front")
        for frame in (1, 2):
            held = supervisor.stabilize(
                self._proposal(frame, target="vehicle_second"),
                self._world(frame, second_vehicle=True),
                safe,
            )
            self.assertEqual(held["target_entity_id"], "vehicle_front")
        switched = supervisor.stabilize(
            self._proposal(3, target="vehicle_second"),
            self._world(3, second_vehicle=True),
            safe,
        )
        self.assertEqual(switched["target_entity_id"], "vehicle_second")

    def test_duplicate_frame_does_not_advance_action_confirmation(self):
        supervisor = TemporalProposalSupervisor(
            TemporalSupervisorConfig(accelerate_confirm_frames=2)
        )
        safe = {"recommended_action": "keep_lane"}
        supervisor.stabilize(
            self._proposal(0, action="decelerate"),
            self._world(0),
            safe,
        )
        duplicate = supervisor.stabilize(
            self._proposal(0, action="accelerate"),
            self._world(0),
            safe,
        )
        self.assertEqual(duplicate["action"], "decelerate")
        next_frame = supervisor.stabilize(
            self._proposal(1, action="accelerate"),
            self._world(1),
            safe,
        )
        self.assertEqual(next_frame["action"], "decelerate")
        confirmed = supervisor.stabilize(
            self._proposal(2, action="accelerate"),
            self._world(2),
            safe,
        )
        self.assertEqual(confirmed["action"], "accelerate")

    def test_large_time_gap_resets_stale_temporal_state(self):
        supervisor = TemporalProposalSupervisor(
            TemporalSupervisorConfig(max_state_gap_s=1.0)
        )
        safe = {"recommended_action": "keep_lane"}
        supervisor.stabilize(
            self._proposal(0, action="decelerate"),
            self._world(0),
            safe,
        )
        resumed_world = self._world(100)
        resumed = supervisor.stabilize(
            self._proposal(100, action="accelerate"),
            resumed_world,
            safe,
        )
        self.assertEqual(resumed["action"], "accelerate")
        self.assertIn(
            "temporal_state_reset_on_time_gap",
            supervisor.diagnostics("request")["reasons"],
        )

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

    def test_equal_width_intent_encoder_preserves_backbone_features(self):
        backbone = TinyBackbone(hidden_size=32)
        encoder = ModernBertIntentEncoder(
            backbone,
            input_hidden_size=32,
            output_hidden_size=32,
            freeze_backbone=True,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        tokens, mask = encoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
        expected = backbone(input_ids).last_hidden_state
        self.assertTrue(torch.equal(tokens, expected))
        self.assertTrue(mask.all())

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

    def test_structured_bev_accepts_live_carla_world_state_fields(self):
        world_state = {
            "ego": {
                "speed_mps": 5.0,
                "acceleration_mps2": 0.0,
                "yaw_rate_rps": 0.0,
                "is_junction": True,
            },
            "objects": [
                {
                    "object_id": "carla_actor_85",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": 13.25,
                        "lateral": -1.5,
                        "vertical": 0.1,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": -2.0,
                        "lateral": 0.25,
                    },
                    "lane_relation": "ego_lane",
                }
            ],
            "environment": {"is_intersection": True},
        }
        batch, entity_ids = StructuredBEVRasterizer(height=16, width=16).build(
            world_state,
            intent_tokens=torch.randn(1, 5, 16),
            intent_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        self.assertEqual(entity_ids, [["carla_actor_85"]])
        self.assertEqual(batch.candidate_features[0, 0, 0].item(), 13.25)
        self.assertEqual(batch.candidate_features[0, 0, 1].item(), -1.5)
        self.assertEqual(batch.candidate_features[0, 0, 4].item(), -2.0)
        self.assertEqual(batch.ego_features[0, 7].item(), 1.0)

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

    def test_training_loss_supervises_lane_pointer_and_confidence(self):
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
        output = model(
            camera_bev=torch.randn(2, 8, 16, 16),
            lidar_bev=torch.randn(2, 4, 16, 16),
            ego_features=torch.randn(2, 8),
            candidate_features=torch.randn(2, 3, 12),
            candidate_mask=torch.ones(2, 3, dtype=torch.bool),
            intent_tokens=torch.randn(2, 5, 16),
            intent_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        loss, components = DistillationLoss()(
            output,
            action_targets=torch.tensor([5, 6]),
            speed_targets=torch.tensor([30.0, 32.0]),
            lane_targets=torch.tensor([1, 2]),
            pointer_targets=torch.tensor([0, -100]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(components["lane"]), 0.0)
        self.assertGreater(float(components["pointer"]), 0.0)
        self.assertGreaterEqual(float(components["confidence"]), 0.0)


if __name__ == "__main__":
    unittest.main()
