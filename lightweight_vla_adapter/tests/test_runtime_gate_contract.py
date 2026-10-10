from copy import deepcopy

import pytest

from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.src.lane_risk_contract import enforce_requested_lane
from lightweight_vla_adapter.tests.fixtures import integration_documents, proposal
from scene_understanding.src.control_decision import build_control_decision


def setup():
    intent, world, alignment, risk = integration_documents()
    canonical = build_control_decision(intent, world, alignment, risk)
    canonical['target_speed_kmh'] = 0.
    runtime = DrivingPlanRuntime()
    runtime.frame_id = canonical['frame_id']
    runtime.timestamp_s = 5.
    runtime.document = intent
    runtime.state = dict(plan_status='ACTIVE', active_step_index=0,
                         step_states=[dict(status='WAITING')])
    risk.update(target_lane_contract=dict(direction='left', can_execute=True),
                behavior_policy=dict(disposition='CLEAR', physical_stop=False, max_speed_kmh=30.),
                risk_evidence=dict(decision_timestamp_s=5.),
                task_feasibility=dict(CHANGE_LANE_LEFT=dict(can_execute=True),
                                      PATH_CLEAR=dict(can_execute=True)))
    risk['lane_change']['left']['speed_cap_kmh'] = 8.
    return runtime, canonical, risk


def test_real_lane_departure_output_passes_bridge_without_losing_authority():
    runtime, canonical, risk = setup()
    result = runtime.start_authorized_lane_motion(canonical, risk)
    assert result['target_speed_kmh'] == 5.
    gated = gate_vla_proposal(proposal(action='accelerate'), result, risk)
    assert gated['action'] == 'lane_change_left'
    assert gated['target_speed_kmh'] == 5.


def deferred():
    runtime, canonical, risk = setup()
    canonical.update(decision_status='BLOCKED', action='stop', target_lane=None,
                     blocked_reason_codes=['physical_target_lane_occupied'])
    evidence = dict(request_id=canonical['request_id'], frame_id=canonical['frame_id'],
                    condition='PATH_CLEAR', valid=True, satisfied=True,
                    source='fresh_lidar_radar_driving_corridor', observed_at_s=5., valid_until_s=5.25)
    return runtime.deferred_lane_navigation(canonical, risk, evidence), risk


def test_deferred_navigation_bridge_never_initiates_lateral_motion():
    canonical, risk = deferred()
    assert canonical['deferred_lane_change'] is True
    result = gate_vla_proposal(proposal(action='lane_change_right'), canonical, risk)
    assert result['action'] == 'keep_lane'
    assert result['target_lane'] is None and result['target_speed_kmh'] == 10.


@pytest.mark.parametrize('action', ['stop', 'emergency_brake'])
def test_deferred_navigation_retains_model_braking_priority(action):
    canonical, risk = deferred()
    result = gate_vla_proposal(proposal(action=action, target_speed_kmh=0.), canonical, risk)
    assert result['action'] == action and result['target_speed_kmh'] == 0.
    assert result['allow_positive_acceleration'] is False


@pytest.mark.parametrize('bad', ['unknown', 'boolean', 'mode', 'frame', 'expiry', 'request', 'permission'])
def test_runtime_extensions_remain_strict(bad):
    canonical, risk = deferred()
    if bad == 'unknown': canonical['permit_anything'] = True
    if bad == 'boolean': canonical['allow_positive_acceleration'] = 'yes'
    if bad == 'mode': canonical['longitudinal_control_mode'] = 'IGNORE_OBSTACLES'
    if bad == 'frame': canonical['deferred_current_lane_evidence']['frame_id'] = 'old'
    if bad == 'expiry': canonical['deferred_current_lane_evidence']['valid_until_s'] = 4.9
    if bad == 'request': canonical['deferred_current_lane_evidence']['request_id'] = 'other'
    if bad == 'permission': canonical['risk_motion_permission'] = {'max_speed_kmh': 100.}
    with pytest.raises(ValueError):
        gate_vla_proposal(proposal(action='keep_lane'), canonical, risk)


def test_passing_metadata_is_scoped_to_the_active_overtake_step():
    runtime, canonical, risk = setup()
    canonical.update(source_step_action='OVERTAKE', action='accelerate', target_lane=None,
                     passing_lane_established=True, target_speed_kmh=10.)
    result = gate_vla_proposal(proposal(action='accelerate'), canonical, risk)
    assert result['passing_lane_established'] is True
    canonical['source_step_action'] = 'KEEP_LANE'
    with pytest.raises(ValueError):
        gate_vla_proposal(proposal(), canonical, risk)


def test_emergency_risk_never_starts_authorized_lane_motion():
    runtime, canonical, risk = setup()
    risk.update(recommended_action='emergency_brake', risk_level='high')
    assert runtime.start_authorized_lane_motion(canonical, risk) == canonical


def test_deferred_motion_survives_every_gate_without_entering_occupied_target_lane():
    canonical, risk = deferred()
    risk['lane_change']['left'].update(is_safe=False,reason_codes=['physical_target_lane_occupied'])
    gated=gate_vla_proposal(proposal(action='accelerate'),canonical,risk)
    bounded,reason=enforce_requested_lane(gated,risk,'left')
    final,_=DrivingPlanRuntime().enforce_execution(bounded,canonical,risk=risk)
    assert reason=='deferred_current_lane_only'
    assert final['action']=='keep_lane' and final['target_lane'] is None
    assert final['target_location'] is None and final['target_speed_kmh']==10.
    assert risk['lane_change']['left']['is_safe'] is False


@pytest.mark.parametrize('bad',['stop','expired','wrong_frame','path_blocked','physical_stop','low_cap'])
def test_deferred_target_gate_rejects_invalid_or_superseded_current_lane_permission(bad):
    canonical,risk=deferred()
    risk['lane_change']['left'].update(is_safe=False,reason_codes=['physical_target_lane_occupied'])
    if bad=='stop':canonical.update(action='stop',target_speed_kmh=0.,allow_positive_acceleration=False)
    if bad=='expired':canonical['deferred_current_lane_evidence']['valid_until_s']=4.9
    if bad=='wrong_frame':canonical['deferred_current_lane_evidence']['frame_id']='old'
    if bad=='path_blocked':risk['task_feasibility']['PATH_CLEAR']['can_execute']=False
    if bad=='physical_stop':risk['behavior_policy']['physical_stop']=True
    if bad=='low_cap':risk['behavior_policy']['max_speed_kmh']=5.
    result,_=enforce_requested_lane(canonical,risk,'left')
    assert result['action']=='stop' and result['target_speed_kmh']==0.
