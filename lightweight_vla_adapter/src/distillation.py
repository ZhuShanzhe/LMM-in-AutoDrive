from __future__ import annotations

import torch
from torch import nn

from .decision_adapter import AdapterOutput


class DistillationLoss(nn.Module):
    def __init__(
        self,
        *,
        hard_action_weight: float = 1.0,
        teacher_action_weight: float = 0.5,
        speed_weight: float = 0.2,
        lane_weight: float = 0.2,
        pointer_weight: float = 0.2,
        confidence_weight: float = 0.05,
        temperature: float = 2.0,
        action_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hard_action_weight = float(hard_action_weight)
        self.teacher_action_weight = float(teacher_action_weight)
        self.speed_weight = float(speed_weight)
        self.lane_weight = float(lane_weight)
        self.pointer_weight = float(pointer_weight)
        self.confidence_weight = float(confidence_weight)
        self.temperature = float(temperature)
        self.register_buffer("action_class_weights", action_class_weights)

    def forward(
        self,
        output: AdapterOutput,
        *,
        action_targets: torch.Tensor,
        speed_targets: torch.Tensor,
        teacher_action_logits: torch.Tensor | None = None,
        lane_targets: torch.Tensor | None = None,
        pointer_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hard_action = nn.functional.cross_entropy(
            output.action_logits,
            action_targets.long(),
            weight=self.action_class_weights,
        )
        speed = nn.functional.smooth_l1_loss(
            output.target_speed_kmh, speed_targets.float()
        )
        teacher_action = output.action_logits.new_zeros(())
        if teacher_action_logits is not None:
            temperature = self.temperature
            teacher_action = nn.functional.kl_div(
                nn.functional.log_softmax(
                    output.action_logits / temperature, dim=-1
                ),
                nn.functional.softmax(
                    teacher_action_logits / temperature, dim=-1
                ),
                reduction="batchmean",
            ) * (temperature**2)
        lane = output.action_logits.new_zeros(())
        if lane_targets is not None:
            lane = nn.functional.cross_entropy(
                output.target_lane_logits,
                lane_targets.long(),
            )
        pointer = output.action_logits.new_zeros(())
        if pointer_targets is not None and (pointer_targets >= 0).any():
            pointer = nn.functional.cross_entropy(
                output.target_pointer_logits,
                pointer_targets.long(),
                ignore_index=-100,
            )
        confidence_target = (
            output.action_logits.detach().argmax(dim=-1) == action_targets.long()
        ).to(output.confidence.dtype)
        confidence = nn.functional.binary_cross_entropy_with_logits(
            output.confidence_logits,
            confidence_target,
        )
        total = (
            self.hard_action_weight * hard_action
            + self.teacher_action_weight * teacher_action
            + self.speed_weight * speed
            + self.lane_weight * lane
            + self.pointer_weight * pointer
            + self.confidence_weight * confidence
        )
        return total, {
            "hard_action": hard_action.detach(),
            "teacher_action": teacher_action.detach(),
            "speed": speed.detach(),
            "lane": lane.detach(),
            "pointer": pointer.detach(),
            "confidence": confidence.detach(),
        }
