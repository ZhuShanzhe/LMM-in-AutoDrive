"""Bounded, causal summaries of completed longitudinal behavior segments."""

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .event_memory import EventMemoryHead
from .sequence_policy import SequenceEventHead, integrate_acceleration, first_action
from .contracts import ACTION_LABELS

SUMMARY_DIM = 164
SUMMARY_SLOTS = 33
BEHAVIOR_SCHEMA = 'behavior_segment_sequence/1.0'
STATE_SLOTS = 6
STATE_WIDTH = 32


class RecursiveStateTransfer(nn.Module):
    """Learned slot-wise inheritance and replacement at a decision boundary."""
    def __init__(self):
        super().__init__()
        self.observation = nn.Sequential(nn.Linear(SUMMARY_DIM, 128), nn.ReLU())
        self.transition = nn.Sequential(nn.Linear(128 + STATE_SLOTS * STATE_WIDTH + 256, 192), nn.ReLU())
        self.inherit_head = nn.Linear(192, STATE_SLOTS)
        self.write_head = nn.Linear(192, STATE_SLOTS)
        self.content_head = nn.Linear(192, STATE_SLOTS * STATE_WIDTH)
        self.norm = nn.LayerNorm(STATE_WIDTH)
        self.reconstruct = nn.Linear(STATE_SLOTS * STATE_WIDTH, 52)
        self.motion_state_head = nn.Linear(STATE_SLOTS * STATE_WIDTH, 3)

    def forward(self, previous, summary, context, reset_mask=None):
        if previous.ndim!=3 or previous.shape[1:]!=(STATE_SLOTS,STATE_WIDTH):
            raise ValueError('Previous recursive state must have shape [B,6,32]')
        if summary.shape!=(previous.shape[0],SUMMARY_DIM) or context.shape!=(previous.shape[0],256):
            raise ValueError('Summary/context dimensions do not match the recursive state')
        if reset_mask is not None:
            if reset_mask.shape!=previous.shape[:2] or reset_mask.dtype!=torch.bool:
                raise ValueError('Reset mask must be a boolean [B,6] tensor')
            previous = previous * (~reset_mask)[..., None]
        features = self.transition(torch.cat((self.observation(summary), previous.flatten(1), context), -1))
        inherit = self.inherit_head(features).sigmoid()
        write = self.write_head(features).sigmoid()
        candidate = self.content_head(features).tanh().reshape(-1, STATE_SLOTS, STATE_WIDTH)
        state = self.norm(inherit[..., None] * previous + write[..., None] * candidate)
        return state, inherit, write


def recover_segment_prefixes(summaries, valid):
    """Recover full causal prefixes from the bounded rolling collection snapshots."""
    completed=[];counts=[];last=None
    for values, mask in zip(np.asarray(summaries),np.asarray(valid)):
        existing=values[:-1][mask[:-1]]
        if not len(existing) and last is not None:
            raise ValueError('Episode contains a memory reset; split it before causal prefix replay')
        if len(existing):
            newest=existing[-1]
            if last is None:
                completed.extend(x.copy() for x in existing)
            elif not np.array_equal(newest,last):
                # At 10Hz and a 0.5s behavior dwell, no more than one new segment is expected.
                completed.append(newest.copy())
            last=newest.copy()
        counts.append(len(completed))
    array=np.stack(completed) if completed else np.zeros((0,SUMMARY_DIM),np.float32)
    return array,np.asarray(counts,dtype=np.int64)


def decision_behavior(decision, speed_kmh):
    if decision.get('action') in ('emergency_brake', 'stop'):
        return 0
    acceleration = decision.get('target_acceleration_mps2')
    if acceleration is None:
        acceleration = (float(decision.get('target_speed_kmh', speed_kmh)) - speed_kmh) / 3.6
    if not np.isfinite(acceleration):
        raise ValueError('Nonfinite executed acceleration')
    return 2 if acceleration > .3 else 0 if acceleration < -.3 else 1


class BehaviorMemoryBuffer:
    def __init__(self, minimum_interval_s=.5):
        self.minimum_interval_s = minimum_interval_s
        self.reset()

    def reset(self):
        self.completed = deque(maxlen=SUMMARY_SLOTS - 1)
        self.current = None
        self.behavior = 1
        self.last_switch = -float('inf')
        self.last_time = None
        self.episode = None
        self.switch_count = 0
        self.boundary_reason = 0

    def boundary(self, reason):
        if reason not in (0, 1, 2, 3):
            raise ValueError('Boundary must be behavior, event, instruction, or observation reset')
        if self.current is not None:
            self.completed.append(self._summary(self.current))
        self.current = None
        self.boundary_reason = reason

    def select(self, decision, speed_kmh, timestamp):
        behavior = decision_behavior(decision, speed_kmh)
        urgent = decision.get('action') == 'emergency_brake'
        if behavior == self.behavior:
            return False
        if not urgent and timestamp - self.last_switch < self.minimum_interval_s - 1e-6:
            return False
        self.boundary(0)
        self.behavior = behavior
        self.last_switch = timestamp
        self.switch_count += 1
        return True

    def push(self, feature, timestamp, episode):
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (52,) or not np.isfinite(feature).all() or not np.isfinite(timestamp):
            raise ValueError('Invalid causal behavior observation')
        if episode != self.episode or (self.last_time is not None and
                (timestamp < self.last_time or timestamp - self.last_time > .35)):
            self.reset()
            self.episode = episode
        if self.last_time == timestamp:
            return False
        self.last_time = timestamp
        if self.current is None:
            self.current = dict(first=feature.copy(), last=feature.copy(), total=feature.astype(np.float64),
                                count=1, start=timestamp, end=timestamp, behavior=self.behavior, reason=self.boundary_reason)
        else:
            self.current.update(last=feature.copy(), count=self.current['count'] + 1, end=timestamp)
            self.current['total'] += feature
        return True

    @staticmethod
    def _summary(segment):
        identity = np.eye(3, dtype=np.float32)[segment['behavior']]
        duration = min(1., (segment['end'] - segment['start'] + .1) / 60.)
        return np.concatenate((segment['first'], segment['last'],
            segment['total'] / segment['count'], identity, [duration],
            np.eye(4,dtype=np.float32)[segment['reason']])).astype(np.float32)

    def tensors(self):
        summaries = list(self.completed)
        values = np.zeros((SUMMARY_SLOTS, SUMMARY_DIM), np.float32)
        valid = np.zeros(SUMMARY_SLOTS, bool)
        for index, item in enumerate(summaries, SUMMARY_SLOTS - 1 - len(summaries)):
            values[index] = item
            valid[index] = True
        if self.current is not None:
            values[-1] = self._summary(self.current)
            valid[-1] = True
        return values, valid


class BehaviorSequenceHead(SequenceEventHead):
    schema_version = BEHAVIOR_SCHEMA

    def __init__(self, context_dim=256, hidden=96):
        super().__init__(context_dim, hidden)
        self.recursive_transfer = RecursiveStateTransfer()
        self.state_projection = nn.Linear(STATE_SLOTS * STATE_WIDTH, 128)
        self.extend_memory_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.segment_residual = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.force_current_segment_only = False
        nn.init.zeros_(self.segment_residual[0].weight)
        nn.init.zeros_(self.segment_residual[0].bias)

    def forward(self, memory, valid, context=None, *, behavior_memory=None, behavior_valid=None,
                recursive_state=None):
        if behavior_memory is None or behavior_valid is None:
            raise ValueError('Behavior checkpoint requires causal segment summaries')
        if behavior_memory.ndim!=3 or behavior_memory.shape[0]!=memory.shape[0] or behavior_memory.shape[-1]!=SUMMARY_DIM:
            raise ValueError('Unexpected behavior summary dimensions')
        if behavior_valid.shape!=behavior_memory.shape[:2]:
            raise ValueError('Unexpected behavior validity dimensions')
        result = EventMemoryHead.forward(self, memory, valid, context, return_features=True)
        features = result.pop('_features')
        if context is None:
            context = memory.new_zeros((memory.shape[0], 256))
        transition_context = torch.zeros_like(context)
        state = memory.new_zeros((memory.shape[0], STATE_SLOTS, STATE_WIDTH)) if recursive_state is None else recursive_state
        inherits=[]
        # Training replays causal segment prefixes; runtime passes a cached compressed state.
        if recursive_state is None:
            for index in range(behavior_memory.shape[1] - 1):
                if not bool(behavior_valid[:, index].any()):
                    continue
                update, inherit, _ = self.recursive_transfer(state, behavior_memory[:, index], transition_context)
                state = torch.where(behavior_valid[:, index, None, None], update, state)
                inherits.append(inherit)
        current_state, inherit, write = self.recursive_transfer(state, behavior_memory[:, -1], transition_context)
        local_state, _, _ = self.recursive_transfer(torch.zeros_like(state), behavior_memory[:, -1], transition_context)
        previous = self.state_projection(current_state.flatten(1))
        current = self.state_projection(local_state.flatten(1))
        logits = self.extend_memory_head(torch.cat((features, current), -1)).squeeze(-1)
        weights = logits.sigmoid()
        selected = (weights >= .5).to(weights)
        gate = selected + weights - weights.detach() if self.training else selected
        if self.force_current_segment_only:
            gate = torch.zeros_like(gate)
            selected = torch.zeros_like(selected)
        mask = behavior_valid[:, :-1]
        fused = features + self.segment_residual(torch.cat((current, gate[:, None] * previous), -1))
        command = 5.5 * self.acceleration_sequence_head(fused).tanh() - 2.5
        speed, acceleration = integrate_acceleration(command, memory[:, -1, 40] * 40,
                                                      memory[:, -1, 42] * (100 / 3.6))
        action = first_action(speed[:, 0], acceleration[:, 0])
        result.update(speed_sequence_mps=speed, acceleration_sequence_mps2=acceleration,
            commanded_acceleration_mps2=command, target_speed_mps=speed[:, 0],
            action_logits=F.one_hot(action, len(ACTION_LABELS)).to(speed) * 12,
            extend_memory_logits=logits, extended_memory_used=selected.bool() & mask.any(1),
            behavior_history_segments=mask.sum(1),inherit_weights=inherit,write_weights=write,
            state_reconstruction=self.recursive_transfer.reconstruct(current_state.flatten(1)),
            relative_motion_state=self.recursive_transfer.motion_state_head(current_state.flatten(1)),
            recursive_state=current_state)
        return result
