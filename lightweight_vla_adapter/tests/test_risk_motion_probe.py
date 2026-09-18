import pytest
import torch

from lightweight_vla_adapter.src.risk_motion_probe import MotionRiskProbe, motion_feature_tensor


def batch():
    return {'motion_values': torch.zeros(2, 20), 'motion_valid_mask': torch.ones(2, 20, dtype=torch.bool),
            'ego_features': torch.zeros(2, 8)}


def test_missing_front_keeps_base_logits_even_with_nonzero_residual():
    probe = MotionRiskProbe('current', torch.zeros(41), torch.ones(41))
    with torch.no_grad():
        probe.residual[-1].bias.fill_(1.)
    data = batch()
    data['motion_values'][1, 0] = 1.
    reference = torch.randn(2, 3)
    corrected = probe(reference, data)
    torch.testing.assert_close(corrected[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(corrected[1], reference[1] + 1.)


def test_invalid_values_are_masked_before_normalization():
    data = batch()
    data['motion_valid_mask'][:, 2] = False
    data['motion_values'][:, 2] = .99
    assert motion_feature_tensor(data, 'current')[:, 2].count_nonzero() == 0


@pytest.mark.parametrize('std', [torch.zeros(41), torch.full((41,), float('nan'))])
def test_invalid_normalization_is_rejected(std):
    with pytest.raises(ValueError):
        MotionRiskProbe('current', torch.zeros(41), std)


def test_actuator_controls_and_map_fields_cannot_change_probe_features():
    data = batch()
    expected = motion_feature_tensor(data, 'current')
    data['ego_features'][:, 1:] = 123.
    torch.testing.assert_close(motion_feature_tensor(data, 'current'), expected, rtol=0, atol=0)
