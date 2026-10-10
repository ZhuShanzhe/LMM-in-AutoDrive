from copy import deepcopy
import pytest

from lightweight_vla_adapter.src.risk_behavior import (
    RiskBehaviorPolicy, cautious_permission, stopping_clearance,
)
from lightweight_vla_adapter.src.environment_speed_policy import environment_speed_bound, apply_environment_speed_bound
from lightweight_vla_adapter.tests.test_risk_behavior import risk, clearance


def stream(level='medium', *, distance=40., fault=None):
    policy=RiskBehaviorPolicy()
    for i in range(16):
        now=1.+i*.1; r=risk('a'+str(i),now,level)
        r.update(source='learned_raw_camera_visual_risk_head',
            prediction_provenance=dict(residual_applied=False))
        if level=='high':r['probabilities']=dict(low=.001,medium=.001,high=.998)
        if fault=='state_high':r['prediction_provenance']['residual_applied']=True
        e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=distance)
        r['physical_forward_radar']['sensor_frame']=i
        r['risk_evidence']['sensor_frame']=i
        if fault=='close_obstacle':e['nearest_obstacle_m']=2.
        if fault=='stale':e['observed_at_s']=now-.4
        if fault=='ttc':r['physical_forward_radar'].update(nearest_closing_velocity_mps=5.,nearest_closing_distance_m=15.)
        r=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,ego_speed_kmh=0.,
            parsed_intent='STOP' if fault=='stop' else 'CHANGE_LANE' if fault=='lane' else 'TURN_RIGHT',
            crossing_hazard=fault=='crossing')
    return r


def test_medium_can_travel_at_30_with_sufficient_clearance():
    r=stream();p=cautious_permission(r)
    assert p['max_speed_kmh']==30. and r['risk_level']=='medium'


def test_medium_speed_is_bounded_by_measured_space():
    r=stream(distance=14.);p=cautious_permission(r)
    assert 20. <= p['max_speed_kmh'] < 30.
    assert stopping_clearance(p['max_speed_kmh']) <= 14.


def test_persistent_visual_only_high_can_use_fresh_clearance_at_10():
    r=stream('high');p=cautious_permission(r)
    assert p['max_speed_kmh']==10. and r['risk_level']=='high'
    assert r['unmodified_recommended_action']=='emergency_brake'


@pytest.mark.parametrize('fault',['state_high','close_obstacle','stale','ttc','stop','lane','crossing'])
def test_physical_or_unverified_high_cannot_get_motion_permission(fault):
    assert cautious_permission(stream('high',fault=fault)) is None


def test_high_permission_is_revoked_by_new_physical_radar_obstacle():
    r=stream('high');r['physical_forward_radar'].update(nearest_distance_m=2.,obstacle_candidate_count=1)
    assert cautious_permission(r) is None


def test_high_permission_needs_full_confirmation_not_only_a_boolean():
    r=stream('high');r['behavior_policy']['high_clear_since_s']=r['risk_evidence']['decision_timestamp_s']
    assert cautious_permission(r) is None


def test_temporally_held_visual_high_preserves_verified_clearance():
    from lightweight_vla_adapter.src.risk_behavior import visual_only_high
    r=stream('high')
    r['source']+='\x2brisk_continuity'
    r['reason_codes']=['temporal_visual_risk_high']
    assert visual_only_high(r)
    assert cautious_permission(r)['max_speed_kmh']==10.
    r['reason_codes'].append('temporal_physical_closing_ttc')
    assert not visual_only_high(r) and cautious_permission(r) is None


def test_visual_label_alternation_cannot_reset_physical_clearance_history():
    policy=RiskBehaviorPolicy()
    for i in range(25):
        now=1.+i*.1; level='medium' if i%2==0 else 'high'
        r=risk('a'+str(i),now,level)
        r.update(source='learned_raw_camera_visual_risk_head',prediction_provenance={'residual_applied':False})
        r['physical_forward_radar']['sensor_frame']=i;r['risk_evidence']['sensor_frame']=i
        e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=40.)
        output=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,ego_speed_kmh=0.,parsed_intent='KEEP_LANE')
        if i>=11:
            permission=cautious_permission(output)
            assert permission and permission['max_speed_kmh']==(10. if level=='high' else 30.)
            assert output['behavior_policy']['clear_since_s']==1.


@pytest.mark.parametrize('distance,closing,expected',[(12.,0.,'BRAKE'),(2.,0.,'EMERGENCY_STOP'),(12.,10.,'EMERGENCY_STOP')])
def test_clearance_and_closing_speed_select_physical_braking_tier(distance,closing,expected):
    from lightweight_vla_adapter.src.risk_behavior import apply_behavior_decision
    policy=RiskBehaviorPolicy();r=stream('high');now=r['risk_evidence']['decision_timestamp_s']+.1
    r['risk_evidence'].update(decision_timestamp_s=now,sensor_frame=100)
    r['physical_forward_radar'].update(sensor_frame=100,nearest_distance_m=distance,obstacle_candidate_count=1,
        nearest_closing_velocity_mps=closing,nearest_closing_distance_m=distance)
    e=clearance();e.update(sensor_frame=100,observed_at_s=now,clear_distance_m=40.,nearest_obstacle_m=distance)
    r=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,ego_speed_kmh=25.,parsed_intent='KEEP_LANE')
    assert r['behavior_policy']['response_tier']==expected and cautious_permission(r) is None
    result,_=apply_behavior_decision(dict(frame_id=r['frame_id'],action='accelerate',target_speed_kmh=30.,
        source_step_action='KEEP_LANE'),r)
    assert result['target_speed_kmh']==0. and not result['allow_positive_acceleration']
    assert result['emergency']==(expected=='EMERGENCY_STOP')


def test_observation_hold_does_not_invent_emergency_braking():
    from lightweight_vla_adapter.src.risk_behavior import apply_behavior_decision
    r=stream('high',fault='stale')
    result,_=apply_behavior_decision(dict(action='emergency_brake',emergency=True,source_step_action='KEEP_LANE'),r)
    assert result['action']=='stop' and not result['emergency'] and result['target_speed_kmh']==0.
    result,_=apply_behavior_decision(dict(action='emergency_brake',emergency=True,source_step_action='EMERGENCY_BRAKE'),r)
    assert result['emergency']


@pytest.mark.parametrize('rain,wet,sun,expected', [(60,80,-10,50),(0,0,-10,60),(60,80,30,60)])
def test_only_rainy_night_changes_environment_speed(rain,wet,sun,expected):
    bound=environment_speed_bound(60,60,precipitation=rain,wetness=wet,sun_altitude=sun)
    assert bound['max_speed_kmh']==expected and not bound['collision_risk_added']


def test_weather_cap_cannot_undo_stop_or_legal_limit():
    bound=environment_speed_bound(40,60,precipitation=60,wetness=80,sun_altitude=-10)
    result,_=apply_environment_speed_bound(dict(action='stop',target_speed_kmh=0.,allow_positive_acceleration=False),bound,0.)
    assert result['target_speed_kmh']==0. and result['action']=='stop'
    assert bound['max_speed_kmh'] < 40.


def test_visual_advisory_execution_preserves_turn_despite_model_stop():
    from lightweight_vla_adapter.tests.fixtures import integration_documents
    from scene_understanding.src.control_decision import build_control_decision
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
    document,world,alignment,_=integration_documents(parser_action='TURN')
    r=stream('high');world['frame_id']=r['frame_id'];alignment['world_state_frame_id']=r['frame_id']
    document['intent']['steps'][0]['parameters']={'direction':'RIGHT','target_location':{'x':10.,'y':10.}}
    canonical=build_control_decision(document,world,alignment,r)
    proposal=dict(schema_version='1.0.0',request_id=canonical['request_id'],frame_id=canonical['frame_id'],
        action='emergency_brake',target_speed_kmh=0.,target_lane=None,target_location=None,
        target_entity_id=None,confidence=.9,model='fixture',latency_ms=1.)
    output=gate_vla_proposal(proposal,canonical,r)
    assert output['action']=='turn_right' and output['target_location']==canonical['target_location']
    assert output['target_speed_kmh']<=10. and output['reason']=='risk_bounded_instruction_execution'
    assert proposal['action']=='emergency_brake'


def test_temporal_supervisor_cannot_override_new_bounded_permission():
    from experiment.CARLA.control.generic_temporal_risk_supervisor import GenericTemporalRiskSupervisor
    r=stream('high');decision=dict(action='turn_right',target_speed_kmh=10.,frame_id=r['frame_id'],
        decision_status='READY',blocked_reason_codes=[],emergency=False)
    output,_=GenericTemporalRiskSupervisor().apply(decision,decision,r,
        parsed_intent='TURN_RIGHT',requested_lane_direction=None,target_lane_risk=None,
        stationary_elapsed_s=100.,resume_active=False,resume_speed_kmh=40.,frame=100)
    assert output['action']=='turn_right' and output['target_speed_kmh']==10.


def test_verified_navigation_can_resume_from_zero_with_original_cruise_goal():
    from lightweight_vla_adapter.src.risk_behavior import resume_bounded_navigation
    r=stream()
    decision=dict(action='turn_right',target_speed_kmh=0.,frame_id=r['frame_id'],
        decision_status='READY',source_step_action='TURN',blocked_reason_codes=[],emergency=False)
    output=resume_bounded_navigation(decision,r,25.)
    assert output['target_speed_kmh']==25. and output['action']=='turn_right'
    assert decision['target_speed_kmh']==0.


@pytest.mark.parametrize('source_action',['STOP','WAIT','YIELD','ADJUST_SPEED','CHECK','CONFIRM'])
def test_navigation_resume_cannot_override_other_instruction_steps(source_action):
    from lightweight_vla_adapter.src.risk_behavior import resume_bounded_navigation
    r=stream()
    decision=dict(action='keep_lane',target_speed_kmh=0.,frame_id=r['frame_id'],
        decision_status='READY',source_step_action=source_action,blocked_reason_codes=[],emergency=False)
    assert resume_bounded_navigation(decision,r,30.)['target_speed_kmh']==0.


def test_action_residual_does_not_reclassify_unchanged_base_risk_as_joint_risk():
    r=stream('high')
    r['prediction_provenance'].update(residual_applied=True,risk_residual_applied=False)
    assert cautious_permission(r)['max_speed_kmh']==10.
    r['prediction_provenance']['risk_residual_applied']=True
    assert cautious_permission(r) is None
