from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightweight_vla_adapter.src.event_memory import EventMemoryBuffer, EventMemoryHead, configure_finetuning
from lightweight_vla_adapter.src.counterfactual_motion_targets import rollout_targets
from lightweight_vla_adapter.src.risk_motion_observation import encode_motion_observation


def observation(frame, gap=25., velocity=-2.):
    front = dict(schema_version='physical_front_radar/1.0', sensor_frame=frame,
                 nearest_distance_m=gap, nearest_relative_velocity_mps=velocity)
    rear = dict(schema_version='physical_rear_radar/1.0', sensor_frame=frame)
    return encode_motion_observation(front, rear, frame=frame, timestamp_s=frame * .1)


def test_memory_resets_on_episode_or_time_gap():
    buffer = EventMemoryBuffer()
    for i in range(100):
        buffer.push(observation(i), [10, 0, 0, 0, 1, 0, 20, 0], [0] * 13 + [.36], episode_id='a')
    assert buffer.tensors()[1].sum() == 80
    buffer.push(observation(100), [10] * 8, [0] * 14, episode_id='b')
    assert buffer.tensors()[1].sum() == 1
    buffer.push(observation(110), [10] * 8, [0] * 14, episode_id='b')
    assert buffer.tensors()[1].sum() == 1


def test_teacher_actuators_do_not_change_memory():
    a, b = EventMemoryBuffer(), EventMemoryBuffer()
    for i in range(5):
        a.push(observation(i), [10, 0, 0, 0, 0, 0, 20, 0], [0] * 14, episode_id='x')
        b.push(observation(i), [10, 0, 0, 1, 1, 1, 20, 0], [0] * 14, episode_id='x')
    assert np.array_equal(a.tensors()[0], b.tensors()[0])


def test_acceleration_is_causal_and_signed():
    buffer = EventMemoryBuffer()
    buffer.push(observation(0), [10, 5, 0, 0, 0, 0, 20, 0], [0] * 14, episode_id='x')
    buffer.push(observation(1), [9.5, 5, 0, 0, 0, 0, 20, 0], [0] * 14, episode_id='x')
    assert buffer.tensors()[0][-1, 41] == pytest.approx(-5 / 12)


def test_masked_memory_has_no_effect():
    torch.manual_seed(1)
    model = EventMemoryHead().eval()
    memory = torch.randn(2, 80, 52)
    valid = torch.ones(2, 80, dtype=torch.bool)
    valid[:, :40] = False
    altered = memory.clone()
    altered[:, :40] = 1000
    with torch.no_grad():
        a, b = model(memory, valid), model(altered, valid)
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_long_history_changes_output_and_event_head_gets_gradients():
    torch.manual_seed(2)
    model = EventMemoryHead()
    with torch.no_grad():
        model.duration_head.bias[2] = 10.
    x = torch.randn(2, 80, 52)
    valid = torch.ones(2, 80, dtype=torch.bool)
    a = model(x, valid)
    y = x.clone()
    y[:, :60] *= -1
    b = model(y, valid)
    assert not torch.allclose(a['action_logits'], b['action_logits'])
    (a['action_logits'].sum() + a['event_logits'].sum()).backward()
    assert model.duration_head.weight.grad.abs().sum() > 0
    assert model.event_head[-1].weight.grad.abs().sum() > 0


def test_unavailable_or_unauthorized_preserves_base():
    base = SimpleNamespace(action_logits=torch.zeros(1, 9))
    learned = dict(available=torch.tensor([False]))
    assert EventMemoryHead.apply(base, learned, torch.tensor([True])) is base


def test_risk_depends_on_relative_motion_not_gap_alone():
    matching = [20, 0, 35, 20, 0, 1000, 20, 0, .85, 20]
    stopped = [20, 0, 35, 0, 0, 1000, 20, 0, .85, 20]
    result = rollout_targets([matching, stopped])
    assert result['required_decel'][1] > result['required_decel'][0]
    assert result['risk'][1] > result['risk'][0]


def test_lead_acceleration_changes_risk():
    rows = [[20, 0, 25, 20, a, 1000, 20, 0, .85, 20] for a in (2, -6)]
    result = rollout_targets(rows)
    assert result['required_decel'][1] > result['required_decel'][0]


def test_horizon_targets_are_monotonic_and_finite():
    rows = [[v, 0, gap, 0, -3, 1000, 10, 0, .4, 20] for v in (0, 10, 25) for gap in (2, 20, 80)]
    result = rollout_targets(rows)
    assert (np.diff(result['horizon'], axis=-1) >= 0).all()
    assert all(np.isfinite(x).all() for x in result.values())
    assert (result['speed'][result['action'] >= 3] == 0).all()


def test_finetuning_really_unfreezes_vla_fusion_layers():
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
            for name in ('ego_projection', 'environment_projection', 'intent_projection', 'action_head', 'speed_head', 'visual_risk_head'):
                setattr(self, name, torch.nn.Linear(4, 4))
            self.query_tokens = torch.nn.Parameter(torch.ones(2, 4))
    base = Base()
    names = configure_finetuning(base, 2)
    assert not base.layers[0].weight.requires_grad
    assert base.layers[-1].weight.requires_grad
    assert 'layers.2.weight' in names
