import torch
import pytest

from lightweight_vla_adapter.src.decision_adapter import AdapterOutput
from lightweight_vla_adapter.src.temporal_decision_residual import TemporalDecisionResidual
from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.src.safety_bridge import enforce_final_lane_policy
from lightweight_vla_adapter.tests.fixtures import integration_documents
from scene_understanding.src.control_decision import build_control_decision


def inputs():
    base = AdapterOutput(action_logits=torch.randn(2, len(ACTION_LABELS)), target_speed_kmh=torch.tensor([20., 30.]),
                         target_lane_logits=torch.zeros(2, 3), target_pointer_logits=torch.zeros(2, 32),
                         confidence_logits=torch.zeros(2), confidence=torch.tensor([.8, .7]),
                         decision_embedding=torch.randn(2, 256), visual_risk_logits=torch.randn(2, 3), risk_input_features=None)
    batch = dict(motion_values=torch.ones(2, 20), motion_valid_mask=torch.ones(2, 20, dtype=torch.bool),
                 motion_history=torch.ones(2, 4, 20), motion_history_valid_mask=torch.ones(2, 4, 20, dtype=torch.bool),
                 motion_history_step_mask=torch.ones(2, 4, dtype=torch.bool), ego_features=torch.randn(2, 8))
    return base, batch


@pytest.mark.parametrize('mode,size', [('current', 297), ('history', 421)])
def test_zero_residual_preserves_predictions(mode, size):
    base, batch = inputs()
    model = TemporalDecisionResidual(torch.zeros(size), torch.ones(size), mode)
    output = model(base, batch, longitudinal_authorized=torch.ones(2, dtype=torch.bool))
    for name in ('action_logits', 'visual_risk_logits', 'target_speed_kmh'):
        assert torch.equal(getattr(output, name), getattr(base, name))


@pytest.mark.parametrize('blocked_by', ['authorization', 'missing_radar'])
def test_nonzero_residual_cannot_override_out_of_scope_or_missing_sensor(blocked_by):
    base, batch = inputs()
    model = TemporalDecisionResidual(torch.zeros(421), torch.ones(421))
    with torch.no_grad():
        model.head.bias.fill_(2.)
    authorized = torch.ones(2, dtype=torch.bool)
    if blocked_by == 'authorization':
        authorized[:] = False
    else:
        batch['motion_valid_mask'][:, 0] = False
    output = model(base, batch, longitudinal_authorized=authorized)
    for name in ('action_logits', 'visual_risk_logits', 'target_speed_kmh', 'confidence'):
        assert torch.equal(getattr(output, name), getattr(base, name))


def test_new_motion_branch_ignores_controls_and_masked_history_values():
    base, batch = inputs()
    model = TemporalDecisionResidual(torch.zeros(421), torch.ones(421))
    batch['motion_history_valid_mask'][:, 0] = False
    expected = model.features(base, batch)
    batch['ego_features'][:, 1:] = 999.
    batch['motion_history'][:, 0] = 999.
    assert torch.equal(expected, model.features(base, batch))


def test_residual_does_not_weaken_baseline_emergency_brake():
    base, batch = inputs()
    base.action_logits.zero_()
    base.action_logits[:, ACTION_LABELS.index('emergency_brake')] = 10.
    model = TemporalDecisionResidual(torch.zeros(421), torch.ones(421))
    with torch.no_grad():
        model.head.bias[ACTION_LABELS.index('accelerate')] = 100.
    result = model(base, batch, longitudinal_authorized=torch.ones(2, dtype=torch.bool))
    assert torch.equal(result.action_logits, base.action_logits)


@pytest.mark.parametrize('unsafe', [False, True])
def test_final_guard_catches_downstream_opposite_lane_override(unsafe):
    intent, world, alignment, risk = integration_documents()
    risk['lane_change']['left']['is_safe'] = not unsafe
    canonical = build_control_decision(intent, world, alignment, risk)
    bypass = dict(canonical, action='lane_change_right', target_lane='right')
    final, reason = enforce_final_lane_policy(bypass, canonical, risk)
    assert reason == 'final_lane_policy_guard'
    assert final['action'] == ('decelerate' if unsafe else 'lane_change_left')


def test_final_guard_keeps_braking_and_safe_authorized_lane():
    intent, world, alignment, risk = integration_documents()
    canonical = build_control_decision(intent, world, alignment, risk)
    for action in ('lane_change_left', 'decelerate', 'stop', 'emergency_brake'):
        candidate = dict(canonical, action=action)
        final, reason = enforce_final_lane_policy(candidate, canonical, risk)
        assert final['action'] == action
        assert reason is None


def test_pipeline_opt_in_changes_action_and_risk_without_changing_default():
    from lightweight_vla_adapter.tests.test_raw_multimodal_adapter import build_batch
    from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
    torch.set_num_threads(2)
    model = LightweightDecisionAdapter(camera_channels=8, lidar_channels=4, candidate_dim=12,
                                       ego_dim=8, intent_dim=16, hidden_size=32, num_heads=4)
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.zero_()
        model.action_head.bias[ACTION_LABELS.index('keep_lane')] = 5.
    pipe = LightweightVLAPipeline(model, device='cpu', checkpoint_loaded=True)
    batch = build_batch()
    _, motion = inputs()
    motion = {k: v[:1] for k, v in motion.items()}
    residual = TemporalDecisionResidual(torch.zeros(197), torch.ones(197), context_dim=32)
    with torch.no_grad():
        residual.head.bias[ACTION_LABELS.index('stop')] = 20.
        residual.head.bias[len(ACTION_LABELS) + 2] = 20.
    kwargs = dict(request_id='test', frame_id='f1', candidate_entity_ids=[[]])
    assert pipe.predict_proposal(batch, **kwargs)['action'] == 'keep_lane'
    changed = pipe.predict_proposal(batch, **kwargs, decision_residual=residual,
                                    motion_inputs=motion, longitudinal_authorized=True)
    assert changed['action'] == 'stop'
    assert pipe.last_visual_risk_assessment['risk_level'] == 'high'
    assert pipe.last_visual_risk_assessment['source'] == 'learned_motion_decision_residual'
    assert pipe.predict_proposal(batch, **kwargs)['action'] == 'keep_lane'


def test_sensor_intent_context_is_invariant_to_actuators():
    from lightweight_vla_adapter.tests.test_raw_multimodal_adapter import build_batch
    from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
    torch.set_num_threads(2)
    model = LightweightDecisionAdapter(camera_channels=8, lidar_channels=4, candidate_dim=12,
                                       ego_dim=8, intent_dim=16, hidden_size=32, num_heads=4).eval()
    pipe = LightweightVLAPipeline(model, device='cpu', checkpoint_loaded=True)
    batch = pipe._move(build_batch())
    with torch.inference_mode():
        before = model(**pipe._model_inputs(batch))
        batch.ego_features[:, 1:] = 3.
        after = model(**pipe._model_inputs(batch))
    assert torch.equal(before.sensor_intent_context, after.sensor_intent_context)
    assert not torch.equal(before.decision_embedding, after.decision_embedding)
