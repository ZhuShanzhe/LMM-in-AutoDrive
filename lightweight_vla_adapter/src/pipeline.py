from __future__ import annotations

import copy
import time
from typing import Any, Sequence

import torch

from .contracts import SensorTensorBatch
from .decision_adapter import LightweightDecisionAdapter, decode_proposal
from .safety_bridge import advance_vla_control_plan
from .temporal_supervisor import TemporalProposalSupervisor


VISUAL_RISK_LEVELS = ("low", "medium", "high")
VISUAL_HIGH_CONFIDENCE_THRESHOLD = 0.55


def decode_visual_risk_assessment(
    logits: torch.Tensor,
    *,
    high_confidence_threshold: float = VISUAL_HIGH_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Convert the learned visual-risk head into the safety-gate contract.

    This adapter deliberately exposes no object identity or metric distance:
    those values are unavailable from the raw-camera head.  It is therefore a
    valid sensor-derived safety signal rather than a disguised simulator-truth
    observation.
    """
    if logits.ndim != 2 or logits.shape != (1, len(VISUAL_RISK_LEVELS)):
        raise ValueError("visual risk logits must have shape [1, 3]")
    if not 0.0 < high_confidence_threshold < 1.0:
        raise ValueError("high confidence threshold must be between 0 and 1")
    probabilities = torch.softmax(logits.detach().float(), dim=-1)[0].cpu()
    index = int(probabilities.argmax())
    raw_level = VISUAL_RISK_LEVELS[index]
    level = (
        "medium"
        if raw_level == "high"
        and float(probabilities[2]) < high_confidence_threshold
        else raw_level
    )
    recommended = {
        "low": "keep_lane",
        "medium": "decelerate",
        "high": "emergency_brake",
    }[level]
    lane_safe = level == "low"
    reason_codes = (
        [] if level == "low" else [f"learned_visual_risk_{level}"]
    )
    return {
        "risk_level": level,
        "recommended_action": recommended,
        "reason_codes": reason_codes,
        "matched_entity_id": None,
        "lane_change": {
            direction: {
                "is_safe": lane_safe,
                "reason_codes": (
                    [] if lane_safe else ["visual_risk_blocks_lane_change"]
                ),
            }
            for direction in ("left", "right")
        },
        "source": "learned_raw_camera_visual_risk_head",
        "raw_argmax_level": raw_level,
        "high_confidence_threshold": high_confidence_threshold,
        "probabilities": {
            name: round(float(probabilities[value]), 6)
            for value, name in enumerate(VISUAL_RISK_LEVELS)
        },
    }


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
        high_confidence_threshold: float = VISUAL_HIGH_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.model_name = model_name
        self.dtype = next(self.model.parameters()).dtype
        self.checkpoint_loaded = checkpoint_loaded
        self.temporal_supervisor = (
            temporal_supervisor or TemporalProposalSupervisor()
        )
        self._last_visual_risk_assessment: dict[str, Any] | None = None
        self._risk_history: list[torch.Tensor] = []
        self._risk_history_max = 5
        self.high_confidence_threshold = float(high_confidence_threshold)

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

    def _risk_history_tensor(self) -> torch.Tensor | None:
        if not self.model.use_temporal_risk or not self._risk_history:
            return None
        frames = self._risk_history[-(self._risk_history_max - 1):]
        return torch.stack(frames, dim=1)

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
        use_model_risk_assessment: bool = False,
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
            model_inputs = self._model_inputs(moved)
            if self.model.use_temporal_risk:
                model_inputs["history_risk_features"] = (
                    self._risk_history_tensor()
                )
            output = self.model(**model_inputs)
        self._last_visual_risk_assessment = decode_visual_risk_assessment(
            output.visual_risk_logits,
            high_confidence_threshold=self.high_confidence_threshold,
        )
        if self.model.use_temporal_risk:
            self._risk_history.append(output.risk_input_features.detach())
            if len(self._risk_history) > self._risk_history_max:
                self._risk_history.pop(0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        effective_candidate_ids = (
            candidate_entity_ids
            if self.model.use_candidate_entities
            else [[] for _ in range(batch.camera_bev.shape[0])]
        )
        proposal = decode_proposal(
            output,
            request_id=request_id,
            frame_id=frame_id,
            candidate_entity_ids=effective_candidate_ids,
            model_name=self.model_name,
            latency_ms=latency_ms,
        )[0]
        effective_risk = (
            self._last_visual_risk_assessment
            if use_model_risk_assessment
            else risk_assessment
        )
        if world_state is not None and effective_risk is not None:
            return self.temporal_supervisor.stabilize(
                proposal,
                world_state,
                effective_risk,
                stream_id=stream_id,
            )
        return proposal

    def predict_visual_risk(
        self,
        batch: SensorTensorBatch,
    ) -> dict[str, Any]:
        """Run the learned risk head without changing proposal temporal state.

        Callers may supply a camera-view mask to obtain sensor-only evidence
        for a particular direction.  This performs a normal model forward but
        does not overwrite the primary fused-view risk assessment used by the
        proposal path.
        """

        if not self.checkpoint_loaded:
            raise RuntimeError(
                "student checkpoint is not loaded; random initialization "
                "cannot be used for risk inference"
            )
        batch.validate()
        if batch.camera_bev.shape[0] != 1:
            raise ValueError("runtime predict_visual_risk requires batch size 1")
        moved = self._move(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            output = self.model(**self._model_inputs(moved))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return decode_visual_risk_assessment(
            output.visual_risk_logits,
            high_confidence_threshold=self.high_confidence_threshold,
        )

    @property
    def last_visual_risk_assessment(self) -> dict[str, Any]:
        if self._last_visual_risk_assessment is None:
            raise RuntimeError("visual risk is unavailable before model inference")
        return copy.deepcopy(self._last_visual_risk_assessment)

    def reset_temporal_state(self, stream_id: str | None = None) -> None:
        self._risk_history.clear()
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
        high_confidence_threshold: float = VISUAL_HIGH_CONFIDENCE_THRESHOLD,
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
            high_confidence_threshold=high_confidence_threshold,
        )
