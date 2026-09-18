"""Causal event-selected 0.4/2/8-second memory for the driving policy."""

from collections import deque
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ACTION_LABELS

FEATURE_DIM = 52
MEMORY_STEPS = 80
EVENT_NAMES = ('closing', 'deceleration', 'observation_loss', 'recovery', 'rear_approach')


class EventMemoryBuffer:
    def __init__(self, length=MEMORY_STEPS):
        self.rows = deque(maxlen=length)
        self.length = length
        self.episode = None

    def push(self, motion, ego, environment, *, episode_id):
        if not episode_id or motion.get('schema_version') != 'risk_motion_observation/1.0':
            raise ValueError('Explicit episode and physical motion schema required')
        values = np.asarray(motion['values'], dtype=np.float32)
        mask = np.asarray(motion['valid_mask'], dtype=np.float32)
        if values.shape != (20,) or mask.shape != (20,):
            raise ValueError('Unexpected motion dimensions')
        timestamp, frame = float(motion['timestamp_s']), int(motion['frame'])
        if not np.isfinite(values).all() or not np.isfinite(timestamp):
            raise ValueError('Nonfinite motion')
        if episode_id != self.episode:
            self.rows.clear()
            self.episode = episode_id
        previous = self.rows[-1] if self.rows else None
        if previous and frame == previous['frame']:
            return False
        if previous and (frame < previous['frame'] or not 0 < timestamp - previous['time'] <= .35):
            self.rows.clear()
            previous = None
        if previous and timestamp - previous['time'] < .095:
            return False
        derivative = np.zeros(4, dtype=np.float32)
        if previous:
            dt = timestamp - previous['time']
            for j, (distance, velocity) in enumerate(((2, 3), (12, 13))):
                valid = mask[velocity] * previous['mask'][velocity]
                same_return = abs(values[distance] - previous['values'][distance]) < .10
                derivative[j] = np.clip((values[velocity] - previous['values'][velocity]) * 40 / dt / 12, -1, 1) * valid * same_return
                derivative[j + 2] = valid * same_return
        ego = np.asarray(ego, dtype=np.float32)
        env = np.asarray(environment, dtype=np.float32)
        # No throttle/brake/steer: avoid copying the data-collection controller.
        signed_acceleration = ((ego[0] - previous['ego_speed']) / (timestamp - previous['time'])) if previous else 0.
        state = np.array([ego[0] / 40, np.clip(signed_acceleration / 12, -1, 1),
                          env[13], env[1], env[2], env[6], env[9], ego[7]], dtype=np.float32)
        feature = np.concatenate((values * mask, mask, state, derivative))
        if not np.isfinite(feature).all():
            raise ValueError('Nonfinite ego/environment state')
        self.rows.append(dict(feature=feature, values=values, mask=mask, time=timestamp, frame=frame,
                              ego_speed=float(ego[0])))
        return True

    def tensors(self):
        x = np.zeros((self.length, FEATURE_DIM), dtype=np.float32)
        valid = np.zeros(self.length, dtype=np.bool_)
        for i, row in enumerate(self.rows, self.length - len(self.rows)):
            x[i], valid[i] = row['feature'], True
        return x, valid


class EventMemoryHead(nn.Module):
    schema_version = 'event_memory_vla/3.0'

    def __init__(self, context_dim=256, hidden=96):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.event_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        self.duration_head = nn.Linear(hidden * 2 + 5, 3)
        self.temporal_conv = nn.Conv1d(hidden, hidden, 3, groups=hidden, padding=0)
        self.context_projection = nn.Linear(context_dim, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3 + 5, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU())
        self.action_head = nn.Linear(128, len(ACTION_LABELS))
        self.risk_head = nn.Linear(128, 3)
        self.horizon_head = nn.Linear(128, 6)
        self.severity_head = nn.Linear(128, 2)
        self.decel_head = nn.Linear(128, 1)
        self.speed_head = nn.Linear(128, 1)
        self.uncertainty_head = nn.Linear(128, 1)

    def forward(self, memory, valid, context=None, *, return_features=False):
        if memory.ndim != 3 or memory.shape[-1] != FEATURE_DIM or memory.shape[1] != MEMORY_STEPS:
            raise ValueError('Expected [batch,80,52] event memory')
        if valid.shape != memory.shape[:2]:
            raise ValueError('Memory validity shape mismatch')
        masked = memory * valid[..., None]
        tokens = self.token(masked) * valid[..., None]
        # Left padding makes the convolution causal, including during training.
        temporal = self.temporal_conv(F.pad(tokens.transpose(1, 2), (2, 0))).transpose(1, 2)
        temporal = (temporal + tokens) * valid[..., None]
        def pool(length):
            m = valid[:, -length:, None]
            return (temporal[:, -length:] * m).sum(1) / m.sum(1).clamp_min(1)
        short, medium, long = [pool(k) for k in (4, 20, 80)]
        current = tokens[:, -1]
        if context is None:
            context = memory.new_zeros((memory.shape[0], self.context_projection.in_features))
        context_features = self.context_projection(context)
        conditioned_current = current + context_features
        event = self.event_head(torch.cat((conditioned_current, short), -1))
        duration = self.duration_head(torch.cat((conditioned_current, short, event.sigmoid()), -1))
        weights = duration.softmax(-1)
        hard_weights = F.one_hot(duration.argmax(-1), 3).to(weights.dtype)
        read_weights = hard_weights + weights - weights.detach() if self.training else hard_weights
        selected = (torch.stack((short, medium, long), 1) * read_weights[..., None]).sum(1)
        fused = self.fusion(torch.cat((current, selected, context_features, event.sigmoid()), -1))
        available = valid[:, -1]
        result = dict(action_logits=self.action_head(fused), risk_logits=self.risk_head(fused),
                    horizon_logits=self.horizon_head(fused).reshape(-1, 2, 3),
                    severity_logits=self.severity_head(fused), required_decel=F.softplus(self.decel_head(fused)).squeeze(-1),
                    target_speed_mps=self.speed_head(fused).sigmoid().squeeze(-1) * masked[:, -1, 42] * (100 / 3.6),
                    uncertainty_logits=self.uncertainty_head(fused).squeeze(-1),
                    event_logits=event, duration_logits=duration, duration_weights=weights,
                    available=available)
        if return_features:
            result['_features'] = fused
        return result

    @staticmethod
    def apply(base, learned, authorized):
        active = learned['available'] & torch.as_tensor(authorized, device=base.action_logits.device, dtype=torch.bool)
        if active.shape != (base.action_logits.shape[0],):
            raise ValueError('Authorization must be a batch vector')
        if not bool(active.any()):
            return base
        action = torch.where(active[:, None], learned['action_logits'], base.action_logits)
        speed = torch.where(active, learned['target_speed_mps'] * 3.6, base.target_speed_kmh.reshape(-1))
        risk = torch.where(active[:, None], learned['risk_logits'], base.visual_risk_logits)
        return replace(base, action_logits=action, target_speed_kmh=speed,
                       visual_risk_logits=risk,
                       confidence=torch.where(active, action.softmax(-1).amax(-1), base.confidence.reshape(-1)),
                       risk_score=None, ordinal_risk_logits=None, risk_horizon_logits=None,
                       risk_uncertainty=learned['uncertainty_logits'].sigmoid(), temporal_used=True)


def configure_finetuning(base, layers=2, train_sensor_backbones=False):
    if not 1 <= layers <= len(base.layers):
        raise ValueError('Choose at least one actual VLA fusion layer')
    base.requires_grad_(False)
    modules = list(base.layers[-layers:]) + [base.ego_projection, base.environment_projection,
              base.intent_projection, base.action_head, base.speed_head, base.visual_risk_head]
    if train_sensor_backbones:
        modules += [base.raw_camera_encoder, base.bev_encoder]
    for module in modules:
        module.requires_grad_(True)
    base.query_tokens.requires_grad_(True)
    return {name: p.numel() for name, p in base.named_parameters() if p.requires_grad}


class EventMemoryRuntime(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.diagnostics = {}

    def forward(self, base_output, batch, *, longitudinal_authorized):
        extra = {}
        if self.head.schema_version in ('behavior_segment_sequence/1.0','layered_behavior_sequence/1.0','layered_behavior_sequence/2.0'):
            extra = dict(behavior_memory=batch['behavior_memory'].float(), behavior_valid=batch['behavior_memory_valid'])
            if 'recursive_state' in batch:
                extra['recursive_state'] = batch['recursive_state'].float()
        if self.head.schema_version in ('layered_behavior_sequence/1.0','layered_behavior_sequence/2.0'):
            extra['layered_context']=batch['layered_context'].float()
        output = self.head(batch['event_memory'].float(), batch['event_memory_valid'],
                           base_output.decision_embedding.float(), **extra)
        self.last_output = output
        self.diagnostics = dict(schema_version=self.head.schema_version,
            event_probabilities=output['event_logits'][0].sigmoid().detach().cpu().tolist(),
            event_names=list(EVENT_NAMES),
            memory_weights=output['duration_weights'][0].detach().cpu().tolist(),
            selected_memory_s=(.4, 2., 8.)[int(output['duration_logits'][0].argmax())],
            valid_history_steps=int(batch['event_memory_valid'][0].sum()),
            horizon_seconds=[1, 2, 4],
            front_rear_horizon_scores=output['horizon_logits'][0].sigmoid().detach().cpu().tolist(),
            severity_scores=output['severity_logits'][0].sigmoid().detach().cpu().tolist(),
            required_decel_mps2=float(output['required_decel'][0]),
            uncertainty_score=float(output['uncertainty_logits'][0].sigmoid()),
            raw_action=ACTION_LABELS[int(output['action_logits'][0].argmax())],
            raw_target_speed_kmh=float(output['target_speed_mps'][0] * 3.6),
            raw_risk_probabilities=output['risk_logits'][0].softmax(-1).detach().cpu().tolist(),
            score_semantics='learned kinematic surrogate; not calibrated real-world probability',
            applied=bool(longitudinal_authorized[0]))
        if extra:
            self.diagnostics['behavior_memory'] = dict(
                extension_probability=float(output['extend_memory_logits'][0].sigmoid()),
                extended_memory_used=bool(output['extended_memory_used'][0]),
                completed_segments=int(output['behavior_history_segments'][0]),
                inheritance_weights=output['inherit_weights'][0].detach().cpu().tolist(),
                write_weights=output['write_weights'][0].detach().cpu().tolist(),
                relative_motion_state=output['relative_motion_state'][0].detach().cpu().tolist(),
                semantics='causal summaries of executed behavior; gate is not a calibrated necessity probability')
        if 'layered_gates' in output:
            self.diagnostics['layered_context']=dict(consumed=True,
                observation_event_gates=output['layered_gates'][0].detach().cpu().tolist(),
                context_ablated=bool(self.head.force_no_layered_context),
                events_ablated=bool(self.head.force_no_event_context),
                long_memory_ablated=bool(self.head.force_current_segment_only),
                feature_version='layered_observation_event/1.0')
        return self.head.apply(base_output, output, longitudinal_authorized)
