"""Action-chunk-inspired longitudinal policy. Not a reproduction of ACT."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .event_memory import EventMemoryHead, EventMemoryRuntime
from .contracts import ACTION_LABELS

STEPS = 30
DT = .1
SEQUENCE_SCHEMA = 'longitudinal_sequence/1.0'


def integrate_acceleration(command, initial_speed, speed_ceiling):
    current = initial_speed.clamp_min(0)
    ceiling = torch.maximum(current, speed_ceiling.clamp_min(0))
    speeds, accelerations = [], []
    for index in range(command.shape[1]):
        following = torch.minimum((current + DT * command[:, index]).clamp_min(0), ceiling)
        speeds.append(following)
        accelerations.append((following - current) / DT)
        current = following
    return torch.stack(speeds, 1), torch.stack(accelerations, 1)


def first_action(speed, acceleration):
    action = torch.full_like(speed, ACTION_LABELS.index('keep_lane'), dtype=torch.long)
    action = torch.where(acceleration > .3, ACTION_LABELS.index('accelerate'), action)
    action = torch.where(acceleration < -.3, ACTION_LABELS.index('decelerate'), action)
    action = torch.where(acceleration <= -6., ACTION_LABELS.index('emergency_brake'), action)
    return torch.where((speed <= .05) & (acceleration <= 0), ACTION_LABELS.index('stop'), action)


class SequenceEventHead(EventMemoryHead):
    schema_version = 'event_memory_sequence/1.0'

    def __init__(self, context_dim=256, hidden=96):
        super().__init__(context_dim, hidden)
        self.acceleration_sequence_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, STEPS))
        nn.init.normal_(self.acceleration_sequence_head[-1].weight, std=.01)
        nn.init.constant_(self.acceleration_sequence_head[-1].bias, .4904146)
        self.action_head.requires_grad_(False)
        self.speed_head.requires_grad_(False)

    def forward(self, memory, valid, context=None):
        result = super().forward(memory, valid, context, return_features=True)
        command = 5.5 * self.acceleration_sequence_head(result.pop('_features')).tanh() - 2.5
        speed, acceleration = integrate_acceleration(command, memory[:, -1, 40] * 40,
                                                     memory[:, -1, 42] * (100 / 3.6))
        action = first_action(speed[:, 0], acceleration[:, 0])
        result.update(speed_sequence_mps=speed, acceleration_sequence_mps2=acceleration,
                      commanded_acceleration_mps2=command, target_speed_mps=speed[:, 0],
                      action_logits=F.one_hot(action, len(ACTION_LABELS)).to(speed.dtype) * 12)
        return result

    @staticmethod
    def apply(base, learned, authorized):
        result = EventMemoryHead.apply(base, learned, authorized)
        # The categorical action is a unit/semantic adapter, not classifier confidence.
        return replace(result, confidence=base.confidence) if result is not base else base


class SequenceMemoryRuntime(EventMemoryRuntime):
    def forward(self, base_output, batch, *, longitudinal_authorized):
        result = super().forward(base_output, batch, longitudinal_authorized=longitudinal_authorized)
        output = self.last_output
        self.diagnostics['longitudinal_sequence'] = dict(
            schema_version=SEQUENCE_SCHEMA, dt_s=DT, horizon_s=DT * STEPS,
            speed_mps=output['speed_sequence_mps'][0].detach().cpu().tolist(),
            acceleration_mps2=output['acceleration_sequence_mps2'][0].detach().cpu().tolist(),
            execution='replan on each new observation; track first speed and acceleration samples',
            action_source='deterministic encoding of first predicted speed/acceleration, not an independent classifier')
        return result
