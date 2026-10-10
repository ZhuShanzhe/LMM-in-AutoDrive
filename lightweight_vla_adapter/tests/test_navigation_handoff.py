from copy import deepcopy

import pytest

from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.fixtures import integration_documents
from scene_understanding.src.control_decision import validate_control_decision


@pytest.mark.parametrize('action', ['TURN', 'CHANGE_LANE', 'SET_SPEED', 'KEEP_LANE', 'PROCEED'])
def test_completed_maneuver_retires_old_target_and_restores_cruise(action):
    doc, world, _, risk = integration_documents(parser_action='ADJUST_SPEED')
    doc['intent']['steps'][-1]['action'] = action
    runtime = DrivingPlanRuntime()
    runtime.document = doc
    runtime.frame_id = world['frame_id']
    runtime.state = {'plan_status': 'COMPLETED'}
    original = dict(action='keep_lane', reason='plan_completed', decision_status='READY',
                    target_speed_kmh=0., target_location={'x': 1., 'y': 2.},
                    source_step_action=action)
    result = runtime.navigation_handoff(original, risk, cruise_speed_kmh=30.)
    assert result['target_speed_kmh'] == 30.
    assert result['target_location'] is None
    assert result['source_step_action'] == 'KEEP_LANE'
    assert runtime.completed_navigation_step()['action'] == 'KEEP_LANE'
    assert original['target_speed_kmh'] == 0.


def test_handoff_preserves_strict_control_decision_contract():
    doc, world, _, risk = integration_documents(parser_action='ADJUST_SPEED')
    runtime = DrivingPlanRuntime()
    runtime.document = doc
    runtime.frame_id = world['frame_id']
    runtime.state = {'plan_status': 'COMPLETED'}
    decision = dict(schema_version='1.0.0', request_id=doc['request_id'],
        frame_id=world['frame_id'], decision_status='READY', action='keep_lane',
        target_speed_kmh=0., target_lane=None, target_location=None, emergency=False,
        reason='plan_completed', parse_status='VALID', parse_confidence=1.,
        source_step_id='step_1', source_step_action='ADJUST_SPEED', source_step_count=1,
        matched_entity_id=None, risk_level='low', risk_reason_codes=[], blocked_reason_codes=[])
    result = runtime.navigation_handoff(decision, risk, cruise_speed_kmh=30.)
    assert validate_control_decision(result) == []
    assert result['request_id'] == doc['request_id']
    assert runtime.navigation_handoff(decision, risk, cruise_speed_kmh=110.) == decision


@pytest.mark.parametrize('fault', ['stop', 'emergency', 'cancel', 'active', 'high', 'stale', 'invalid_speed'])
def test_handoff_never_restarts_explicit_stop_or_overrides_risk(fault):
    doc, world, _, risk = integration_documents(parser_action='ADJUST_SPEED')
    doc['intent']['steps'][-1]['action'] = {
        'stop': 'STOP', 'emergency': 'EMERGENCY_BRAKE', 'cancel': 'CANCEL',
    }.get(fault, 'TURN')
    runtime = DrivingPlanRuntime()
    runtime.document = doc
    runtime.frame_id = world['frame_id']
    runtime.state = {'plan_status': 'ACTIVE' if fault == 'active' else 'COMPLETED'}
    if fault == 'high':
        risk.update(risk_level='high', recommended_action='emergency_brake')
    if fault == 'stale':
        risk['frame_id'] = 'old'
    original = dict(action='stop', target_speed_kmh=0., reason='plan_completed', decision_status='READY')
    assert runtime.navigation_handoff(deepcopy(original), risk,
        cruise_speed_kmh=float('nan') if fault == 'invalid_speed' else 30.) == original
