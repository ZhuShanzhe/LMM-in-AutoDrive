import torch
import pytest
from torch import nn

from lightweight_vla_adapter.src.deployment_export import prepare_fixed_shape_export
from lightweight_vla_adapter.scripts.run_offline_inference import build_model


@pytest.mark.parametrize('length', [0, 1, 4, 16, 32])
def test_bounded_mask_count_preserves_empty_partial_and_full_masks(length):
    from lightweight_vla_adapter.src.decision_adapter import _valid_token_count
    mask = torch.zeros(3, length, dtype=torch.bool)
    mask[1, ::2] = True
    mask[2] = True
    expected = mask.sum(1, keepdim=True).clamp_min(1)
    actual = _valid_token_count(mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.int64


def test_hardswish_lowering_matches_reference():
    from lightweight_vla_adapter.src.deployment_export import ExportHardSwish
    value = torch.cat([torch.linspace(-100, 100, 10001), torch.tensor([-3., 0., 3.])])
    torch.testing.assert_close(ExportHardSwish()(value), nn.Hardswish()(value), rtol=1e-6, atol=1e-6)


def test_hardsigmoid_lowering_matches_reference():
    from lightweight_vla_adapter.src.deployment_export import ExportHardSigmoid
    value = torch.cat([torch.linspace(-100, 100, 10001), torch.tensor([-3., 0., 3.])])
    torch.testing.assert_close(ExportHardSigmoid()(value), nn.Hardsigmoid()(value), rtol=1e-6, atol=1e-6)


def test_static_pool_preserves_adaptive_bins():
    for size, output, kernel, stride in ((7, 2, 4, 3), (64, 8, 8, 8)):
        x = torch.randn(2, 8, size, size)
        torch.testing.assert_close(nn.AdaptiveAvgPool2d(output)(x), nn.AvgPool2d(kernel, stride)(x))


def test_export_refuses_unprofiled_shapes_and_training():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="eval"):
        prepare_fixed_shape_export(SimpleNamespace(training=True))
    with pytest.raises(ValueError, match="Temporal"):
        prepare_fixed_shape_export(SimpleNamespace(training=False, use_temporal_risk=True))


def test_state_residual_initially_zero_and_can_learn():
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1]/"configs/universal_three_scene_v6_sensor_policy.json").read_text())
    config["use_state_conditioned_risk"] = True
    model = build_model(config).eval()
    x = torch.randn(2, config["hidden_size"] + config["ego_dim"] + config["environment_dim"])
    output = model.state_risk_residual(x)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert model.state_risk_residual[-1].weight.grad.abs().sum() > 0
