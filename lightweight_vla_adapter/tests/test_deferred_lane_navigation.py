from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


def context():
    runtime=DrivingPlanRuntime()
    runtime.frame_id='f20';runtime.timestamp_s=1.
    runtime.document=dict(request_id='change',intent=dict(steps=[dict(step_id='lane',
        action='CHANGE_LANE',parameters=dict(direction='LEFT'))]))
    runtime.state=dict(plan_status='ACTIVE',active_step_index=0,
        step_states=[dict(status='WAITING')])
    decision=dict(decision_status='BLOCKED',action='stop',target_speed_kmh=0.,
        emergency=False,blocked_reason_codes=['physical_target_lane_occupied'])
    risk=dict(frame_id='f20',risk_level='low',behavior_policy=dict(
        disposition='CLEAR',physical_stop=False,max_speed_kmh=30.))
    evidence=dict(request_id='change',frame_id='f20',condition='PATH_CLEAR',
        valid=True,satisfied=True,source='fresh_lidar_radar_driving_corridor',
        observed_at_s=1.,valid_until_s=1.25)
    return runtime,decision,risk,evidence


@pytest.mark.parametrize('failure',[None,'missing','expired','wrong_frame','wrong_request',
    'unknown_corridor','occupied_corridor','physical_stop','insufficient','unresolved_direction',
    'emergency','active_change','low_cap'])
def test_deferred_motion_requires_fresh_current_lane_clearance(failure):
    runtime,decision,risk,evidence=context()
    if failure=='missing':evidence={}
    if failure=='expired':evidence['observed_at_s']=0.
    if failure=='wrong_frame':evidence['frame_id']='old'
    if failure=='wrong_request':evidence['request_id']='other'
    if failure=='unknown_corridor':evidence['valid']=False
    if failure=='occupied_corridor':evidence['satisfied']=False
    if failure=='physical_stop':risk['behavior_policy']['physical_stop']=True
    if failure=='insufficient':risk['behavior_policy']['disposition']='OBSERVATION_INSUFFICIENT'
    if failure=='unresolved_direction':runtime.document['intent']['steps'][0]['parameters']={}
    if failure=='emergency':decision['emergency']=True
    if failure=='active_change':runtime.state['step_states'][0]['status']='ACTIVE'
    if failure=='low_cap':risk['behavior_policy']['max_speed_kmh']=3.
    before=deepcopy(runtime.state)
    result=runtime.deferred_lane_navigation(decision,risk,evidence)
    assert (result.get('deferred_lane_change') is True)==(failure is None)
    assert runtime.state==before
    if failure is None:
        assert result['action']=='keep_lane' and result['target_lane'] is None
        assert result['target_speed_kmh']==10.
    else:assert result==decision


def test_later_model_or_sequence_cannot_turn_deferred_navigation_into_a_lane_change():
    runtime,decision,risk,evidence=context()
    canonical=runtime.deferred_lane_navigation(decision,risk,evidence)
    proposal=dict(action='lane_change_left',target_lane='left',target_location=dict(x=1.,y=2.),
        target_speed_kmh=40.,target_acceleration_mps2=3.,longitudinal_sequence_schema='sequence')
    result,_=runtime.enforce_execution(proposal,canonical,risk=risk)
    assert result['action']=='keep_lane' and result['target_lane'] is None
    assert result['target_location'] is None and result['target_speed_kmh']==10.
    assert 'target_acceleration_mps2' not in result


def test_deferred_navigation_never_overrides_a_later_emergency_stop():
    runtime,decision,risk,evidence=context()
    canonical=runtime.deferred_lane_navigation(decision,risk,evidence)
    result,_=runtime.enforce_execution(dict(action='emergency_brake',emergency=True,target_speed_kmh=0.),canonical)
    assert result['action']=='emergency_brake' and result['target_speed_kmh']==0.


def test_partially_executed_lane_change_cannot_be_reclassified_as_current_lane_cruise():
    runtime,decision,risk,evidence=context()
    assert runtime.deferred_lane_navigation(decision,risk,evidence,maneuver_started=True)==decision
