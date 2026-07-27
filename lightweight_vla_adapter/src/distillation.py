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
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.hard_action_weight = float(hard_action_weight)
        self.teacher_action_weight = float(teacher_action_weight)
        self.speed_weight = float(speed_weight)
        self.temperature = float(temperature)

    def forward(
        self,
        output: AdapterOutput,
        *,
        action_targets: torch.Tensor,
        speed_targets: torch.Tensor,
        teacher_action_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hard_action = nn.functional.cross_entropy(
            output.action_logits, action_targets.long()
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
        total = (
            self.hard_action_weight * hard_action
            + self.teacher_action_weight * teacher_action
            + self.speed_weight * speed
        )
        return total, {
            "hard_action": hard_action.detach(),
            "teacher_action": teacher_action.detach(),
            "speed": speed.detach(),
        }
