"""Fixed-shape tensor boundary for x86 ONNX checks, not a J6 compiler."""

import copy
import torch
from torch import nn


INPUT_NAMES = (
    "camera_bev", "lidar_bev", "ego_features", "candidate_features",
    "candidate_mask", "intent_tokens", "intent_mask", "camera_images",
    "camera_view_mask", "environment_features",
)
OUTPUT_NAMES = ("action_logits", "speed_kmh", "lane_logits", "visual_risk_logits", "confidence")


class ExportHardSwish(nn.Module):
    """Equivalent basic-operator form for the Horizon floating-point importer."""

    def forward(self, value):
        middle = value * (value + 3.0) / 6.0
        # A Clip/Mul form is fused back to unsupported floating HardSwish by HMCT.
        return torch.where(value <= -3.0, torch.zeros_like(value),
                           torch.where(value >= 3.0, value, middle))


class ExportHardSigmoid(nn.Module):
    def forward(self, value):
        return torch.where(value <= -3.0, torch.zeros_like(value),
                           torch.where(value >= 3.0, torch.ones_like(value), (value + 3.0) / 6.0))


def prepare_fixed_shape_export(model):
    """Copy the model and lower its known pooling operations exactly.

    For a 7x7 feature map, adaptive 2x2 pooling uses overlapping 4x4 bins
    with stride 3. This is not equivalent to resizing images or dropping bins.
    """
    if model.training:
        raise ValueError("Export requires eval mode")
    if model.use_temporal_risk:
        raise ValueError("Temporal history export needs a separate explicit input contract")
    if model.raw_camera_encoder.token_grid != (2, 2):
        raise ValueError("This export profile requires a 2x2 visual token grid")
    if model.bev_encoder.output_grid != (8, 8):
        raise ValueError("This export profile requires an 8x8 BEV token grid")
    result = copy.deepcopy(model)
    result.raw_camera_encoder.pool = nn.AvgPool2d(kernel_size=4, stride=3)
    result.bev_encoder.pool = nn.AvgPool2d(kernel_size=8, stride=8)
    for name, module in list(result.named_modules()):
        if isinstance(module, (nn.Hardswish, nn.Hardsigmoid)):
            parent_name, _, child_name = name.rpartition('.')
            parent = result.get_submodule(parent_name) if parent_name else result
            replacement = ExportHardSwish() if isinstance(module, nn.Hardswish) else ExportHardSigmoid()
            setattr(parent, child_name, replacement)
    return result


class FixedShapePolicy(nn.Module):
    """224x224 RGB, 64x64 BEV; omit inactive pointer/history outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, camera_bev, lidar_bev, ego_features, candidate_features,
                candidate_mask, intent_tokens, intent_mask, camera_images,
                camera_view_mask, environment_features):
        if camera_images.shape[-2:] != (224, 224):
            raise ValueError("camera_images must use the fixed 224x224 profile")
        if camera_bev.shape[-2:] != (64, 64) or lidar_bev.shape[-2:] != (64, 64):
            raise ValueError("BEV inputs must use the fixed 64x64 profile")
        out = self.model(**dict(zip(INPUT_NAMES, (
            camera_bev, lidar_bev, ego_features, candidate_features, candidate_mask,
            intent_tokens, intent_mask, camera_images, camera_view_mask, environment_features,
        ))))
        return (out.action_logits, out.target_speed_kmh, out.target_lane_logits,
                out.visual_risk_logits, out.confidence)


def initialize_state_risk_branch(model, checkpoint):
    """Warm-start only the new zero residual; reject unrelated missing weights."""
    if not model.use_state_conditioned_risk:
        raise ValueError("The state-conditioned risk branch must be enabled")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {key for key in model.state_dict() if key.startswith("state_risk_residual.")}
    missing = set(model.state_dict()) - set(state)
    unexpected = set(state) - set(model.state_dict())
    if missing != expected or unexpected:
        raise ValueError(f"Incompatible baseline: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(state, strict=False)
    return model
