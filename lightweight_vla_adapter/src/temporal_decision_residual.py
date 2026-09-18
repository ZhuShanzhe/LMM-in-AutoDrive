"""Opt-in longitudinal residual over a frozen VLA context and causal radar."""

from dataclasses import replace

import torch
from torch import nn

from .contracts import ACTION_LABELS
from .risk_motion_probe import motion_feature_tensor


class TemporalDecisionResidual(nn.Module):
    schema_version = 'temporal_decision_residual/1.0'

    def __init__(self, mean, std, mode='history', context_dim=256, context_type='decision_embedding'):
        super().__init__()
        self.mode = mode
        self.context_dim = context_dim
        if context_type not in ('decision_embedding', 'sensor_intent_context'):
            raise ValueError('Unsupported context type')
        self.context_type = context_type
        expected = context_dim + {'current': 41, 'history': 165}[mode]
        if mean.shape != (expected,) or std.shape != mean.shape:
            raise ValueError('Unexpected feature dimensions')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid normalization statistics')
        self.register_buffer('mean', mean.detach().clone())
        self.register_buffer('std', std.detach().clone())
        self.fusion = nn.Sequential(nn.Linear(expected, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, len(ACTION_LABELS) + 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        supported = {'keep_lane', 'accelerate', 'decelerate', 'stop', 'emergency_brake'}
        self.register_buffer('action_mask', torch.tensor([float(a in supported) for a in ACTION_LABELS]))

    def features(self, base_output, batch):
        context = getattr(base_output, self.context_type)
        if context is None or context.shape[-1] != self.context_dim:
            raise ValueError('Required frozen context unavailable or incompatible')
        return torch.cat((context.detach(), motion_feature_tensor(batch, self.mode)), -1).float()

    def predict_tensors(self, features, base_action, base_risk, base_speed, available):
        # An experimental residual must not weaken an existing emergency brake.
        available = available * (base_action.argmax(-1) != ACTION_LABELS.index('emergency_brake'))[:, None]
        delta = self.head(self.fusion((features - self.mean) / self.std)) * available
        count = len(ACTION_LABELS)
        action = base_action + delta[:, :count] * self.action_mask
        risk = base_risk + delta[:, count:count + 3]
        speed = (base_speed.reshape(-1) + 10. * delta[:, -1]).clamp(0., 60.)
        return action, risk, speed

    def forward(self, base_output, batch, *, longitudinal_authorized):
        """Caller must authorize an active longitudinal-only instruction.

        Not for lane changes/turns. Sensor loss retains the base output, and
        the existing FSM and safety gate must still process the proposal.
        """
        features = self.features(base_output, batch)
        authorized = torch.as_tensor(longitudinal_authorized, device=features.device, dtype=torch.bool)
        if authorized.shape != (features.shape[0],):
            raise ValueError('Authorization must have one boolean per sample')
        available = (batch['motion_values'][:, :1] * batch['motion_valid_mask'][:, :1]
                     * authorized[:, None])
        action, risk, speed = self.predict_tensors(features, base_output.action_logits,
                                                 base_output.visual_risk_logits, base_output.target_speed_kmh, available)
        active = available[:, 0].bool()
        confidence = action.softmax(-1).amax(-1)
        return replace(base_output, action_logits=action, visual_risk_logits=risk,
                       target_speed_kmh=torch.where(active, speed, base_output.target_speed_kmh.reshape(-1)),
                       confidence=torch.where(active, confidence, base_output.confidence.reshape(-1)),
                       temporal_used=base_output.temporal_used or (self.mode == 'history' and bool(active.any())))

    @classmethod
    def from_checkpoint(cls, path):
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if artifact.get('schema_version') != cls.schema_version:
            raise ValueError('Unsupported temporal decision checkpoint')
        model = cls(artifact['mean'], artifact['std'], artifact['mode'], artifact['context_dim'],
                    artifact.get('context_type', 'decision_embedding'))
        model.load_state_dict(artifact['model'], strict=True)
        model.baseline_sha256 = artifact.get('baseline_sha256')
        return model.eval()
