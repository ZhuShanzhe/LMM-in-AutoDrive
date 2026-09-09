"""Experimental learned risk correction; not a default decision controller."""

import torch
from torch import nn


def motion_feature_tensor(batch, mode):
    if mode == 'current':
        value, mask = batch['motion_values'], batch['motion_valid_mask']
        return torch.cat((value * mask, mask.float(), batch['ego_features'][:, :1]), -1)
    if mode != 'history':
        raise ValueError('Unknown motion feature mode')
    value, mask = batch['motion_history'], batch['motion_history_valid_mask']
    return torch.cat(((value * mask).flatten(1), mask.float().flatten(1),
                      batch['motion_history_step_mask'].float(), batch['ego_features'][:, :1]), -1)


def build_motion_residual(dim):
    model = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 3))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    return model


class MotionRiskProbe(nn.Module):
    """Correct three risk logits; missing front measurements retain the prior."""

    def __init__(self, mode, mean, std):
        super().__init__()
        expected = {'current': 41, 'history': 165}.get(mode)
        if expected is None or tuple(mean.shape) != (expected,) or std.shape != mean.shape:
            raise ValueError('Unexpected fixed motion feature dimensions')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid training normalization statistics')
        self.mode = mode
        self.residual = build_motion_residual(expected)
        self.register_buffer('mean', mean.detach().clone().float())
        self.register_buffer('std', std.detach().clone().float())

    @classmethod
    def from_checkpoint(cls, path):
        artifact = torch.load(path, weights_only=True, map_location='cpu')
        if artifact.get('schema_version') != 'motion_risk_probe/2.0':
            raise ValueError('Unsupported motion probe checkpoint')
        model = cls(artifact['mode'], artifact['mean'], artifact['std'])
        model.residual.load_state_dict(artifact['model'], strict=True)
        return model.eval()

    def forward(self, base_risk_logits, batch):
        features = motion_feature_tensor(batch, self.mode).to(self.mean.dtype)
        if base_risk_logits.shape != (features.shape[0], 3):
            raise ValueError('Base risk logits must have shape [B,3]')
        available = batch['motion_values'][:, :1] * batch['motion_valid_mask'][:, :1]
        delta = self.residual((features - self.mean) / self.std) * available
        return base_risk_logits + delta
