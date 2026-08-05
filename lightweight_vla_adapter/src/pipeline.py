from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from .contracts import SensorTensorBatch
from .decision_adapter import LightweightDecisionAdapter, decode_proposal
from .safety_bridge import advance_vla_control_plan
from .temporal_supervisor import TemporalProposalSupervisor


class LightweightVLAPipeline:
    def __init__(
        self,
        model: LightweightDecisionAdapter,
        *,
        model_name: str = "lightweight-vla-adapter",
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        checkpoint_loaded: bool = False,
        temporal_supervisor: TemporalProposalSupervisor | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.model_name = model_name
        self.dtype = next(self.model.parameters()).dtype
        self.checkpoint_loaded = checkpoint_loaded
        self.temporal_supervisor = (
            temporal_supervisor or TemporalProposalSupervisor()
        )

    def _move(self, batch: SensorTensorBatch) -> SensorTensorBatch:
        def floating(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=self.device, dtype=self.dtype)

        def images(tensor: torch.Tensor) -> torch.Tensor:
            moved = tensor.to(device=self.device, dtype=self.dtype)
            return moved.div_(255.0) if not tensor.is_floating_point() else moved

        return SensorTensorBatch(
            camera_bev=floating(batch.camera_bev),
            lidar_bev=floating(batch.lidar_bev),
            ego_features=floating(batch.ego_features),
            candidate_features=floating(batch.candidate_features),
            candidate_mask=batch.candidate_mask.to(self.device),
            intent_tokens=floating(batch.intent_tokens),
            intent_mask=batch.intent_mask.to(self.device),
            camera_images=(
                images(batch.camera_images)
                if batch.camera_images is not None
                else None
            ),
            camera_view_mask=(
                batch.camera_view_mask.to(self.device)
                if batch.camera_view_mask is not None
                else None
            ),
            environment_features=(
                floating(batch.environment_features)
                if batch.environment_features is not None
                else None
            ),
        )

    @staticmethod
    def _model_inputs(batch: SensorTensorBatch) -> dict[str, torch.Tensor | None]:
        return {
            "camera_bev": batch.camera_bev,
            "lidar_bev": batch.lidar_bev,
            "ego_features": batch.ego_features,
            "candidate_features": batch.candidate_features,
            "candidate_mask": batch.candidate_mask,
            "intent_tokens": batch.intent_tokens,
            "intent_mask": batch.intent_mask,
            "camera_images": batch.camera_images,
            "camera_view_mask": batch.camera_view_mask,
            "environment_features": batch.environment_features,
        }

    def predict_proposal(
        self,
        batch: SensorTensorBatch,
        *,
        request_id: str,
        frame_id: str,
        candidate_entity_ids: Sequence[Sequence[str]],
        world_state: dict[str, Any] | None = None,
        risk_assessment: dict[str, Any] | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.checkpoint_loaded:
            raise RuntimeError(
                "student checkpoint is not loaded; random initialization "
                "cannot be used for decision inference"
            )
        batch.validate()
        if batch.camera_bev.shape[0] != 1:
            raise ValueError("runtime predict_proposal currently requires batch size 1")
        moved = self._move(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**self._model_inputs(moved))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        proposal = decode_proposal(
            output,
            request_id=request_id,
            frame_id=frame_id,
            candidate_entity_ids=candidate_entity_ids,
            model_name=self.model_name,
            latency_ms=latency_ms,
        )[0]
        if world_state is not None and risk_assessment is not None:
            return self.temporal_supervisor.stabilize(
                proposal,
                world_state,
                risk_assessment,
                stream_id=stream_id,
            )
        return proposal

    def reset_temporal_state(self, stream_id: str | None = None) -> None:
        self.temporal_supervisor.reset(stream_id)

    def warmup(self, batch: SensorTensorBatch, *, iterations: int = 10) -> None:
        if iterations <= 0:
            raise ValueError("warmup iterations must be positive")
        batch.validate()
        moved = self._move(batch)
        with torch.inference_mode():
            for _ in range(iterations):
                self.model(**self._model_inputs(moved))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def decide(
        self,
        batch: SensorTensorBatch,
        driving_intent: dict[str, Any],
        world_state: dict[str, Any],
        semantic_alignment: dict[str, Any],
        risk_assessment: dict[str, Any],
        *,
        candidate_entity_ids: Sequence[Sequence[str]],
        prior_state: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        proposal = self.predict_proposal(
            batch,
            request_id=driving_intent["request_id"],
            frame_id=world_state["frame_id"],
            candidate_entity_ids=candidate_entity_ids,
            world_state=world_state,
            risk_assessment=risk_assessment,
            stream_id=driving_intent["request_id"],
        )
        state, final_decision = advance_vla_control_plan(
            driving_intent,
            world_state,
            semantic_alignment,
            risk_assessment,
            proposal,
            prior_state=prior_state,
            feedback=feedback,
        )
        return proposal, state, final_decision

    @classmethod
    def from_checkpoint(
        cls,
        model: LightweightDecisionAdapter,
        checkpoint_path: str,
        *,
        model_name: str = "lightweight-vla-adapter",
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        temporal_supervisor: TemporalProposalSupervisor | None = None,
        strict_checkpoint: bool = True,
    ) -> "LightweightVLAPipeline":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=strict_checkpoint)
        if not strict_checkpoint:
            unexpected = list(incompatible.unexpected_keys)
            if unexpected:
                raise RuntimeError(
                    "legacy checkpoint contains unexpected parameters: "
                    + ", ".join(unexpected)
                )
        return cls(
            model,
            model_name=model_name,
            device=device,
            dtype=dtype,
            checkpoint_loaded=True,
            temporal_supervisor=temporal_supervisor,
        )
