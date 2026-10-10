from copy import deepcopy
import numpy as np
import pytest

from lightweight_vla_adapter.src.risk_behavior import (
    RiskBehaviorPolicy, lidar_corridor_evidence, cautious_permission, apply_behavior_decision,
    potential_path_intrusion,
)
from lightweight_vla_adapter.src.longitudinal_contract import enforce_longitudinal_contract
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.tests.fixtures import integration_documents


def road_cloud(slope=0.):
    x=np.arange(2.6,22.,.2)
    return np.array([[a,y,10.+a*slope] for a in x for y in (-.5,0.,.5)])


def clearance(points=None,slope=0.):
    return lidar_corridor_evidence(road_cloud(slope) if points is None else points,
        [(0,0,10),(20,0,10+20*slope)],half_width_m=1.3,ego_front_m=2.,sensor_frame=10,observed_at_s=1.)


def risk(frame='f10',timestamp=1.,level='medium'):
    return dict(frame_id=frame,risk_level=level,recommended_action='emergency_brake' if level=='high' else 'decelerate',
        reason_codes=['learned_visual_risk_'+level],probabilities=dict(low=.001,medium=.998,high=.001),
        lane_change={d:dict(is_safe=False,reason_codes=['not_verified']) for d in ('left','right')},
        physical_forward_radar=dict(schema_version='physical_front_radar/1.0',sensor_frame=10,
            nearest_distance_m=None,obstacle_candidate_count=0),
        risk_evidence=dict(is_fresh=True,sensor_frame=10,decision_timestamp_s=timestamp))


def confirmed(action='TURN_RIGHT'):
    policy=RiskBehaviorPolicy(max_speed_kmh=6.)
    for i in range(8):
        now=1.+i*.1;frame='f'+str(i);r=risk(frame,now);r['risk_evidence']['sensor_frame']=i
        evidence=clearance();evidence.update(sensor_frame=i,observed_at_s=now)
        r['physical_forward_radar']['sensor_frame']=i
        result=policy.apply(r,evidence,frame_id=frame,timestamp_s=now,ego_speed_kmh=0.,parsed_intent=action)
    return result


@pytest.mark.parametrize('slope', [0.,.15,-.15])
def test_measured_road_support_on_flat_and_sloped_roads(slope):
    e=clearance(slope=slope)
    assert e['valid'] and e['clear_distance_m']>10 and e['nearest_obstacle_m'] is None


def test_empty_cloud_is_not_clearance():
    assert not clearance(np.empty((0,3)))['valid']
    e=clearance(np.array([[x,0.,15.] for x in np.arange(3.,23.,.2)]))
    assert e['clear_distance_m']==0.


def test_obstacle_truncates_measured_free_horizon():
    e=clearance(np.vstack([road_cloud(),[5.,0.,11.]]))
    assert e['nearest_obstacle_m']==3. and e['clear_distance_m']<=2.


@pytest.mark.parametrize('fault', ['stale','no_support','mismatched_frame','high','stop_intent','lane_intent','gap','duplicate'])
def test_no_cautious_permission_for_missing_or_invalid_evidence(fault):
    policy=RiskBehaviorPolicy()
    for i in range(10):
        now=1.+i*(.5 if fault=='gap' else .1);frame='f'+str(i)
        r=risk(frame,now,'high' if fault=='high' else 'medium')
        e=clearance();e.update(sensor_frame=0 if fault=='duplicate' else i,observed_at_s=now)
        r['risk_evidence']['sensor_frame']=e['sensor_frame']
        r['physical_forward_radar']['sensor_frame']=e['sensor_frame']
        if fault=='stale':e['observed_at_s']=now-.4
        if fault=='no_support':e['clear_distance_m']=0.
        if fault=='mismatched_frame':r['risk_evidence']['sensor_frame']=-1
        action='STOP' if fault=='stop_intent' else 'CHANGE_LANE' if fault=='lane_intent' else 'TURN_RIGHT'
        result=policy.apply(r,e,frame_id=frame,timestamp_s=now,ego_speed_kmh=0.,parsed_intent=action)
    assert cautious_permission(result) is None


def test_clearance_does_not_relabel_medium_as_low():
    r=confirmed()
    assert r['risk_level']=='medium' and r['recommended_action']=='monitor'
    assert r['behavior_policy']['disposition']=='CAUTIOUS_PROCEED'
    assert r['unmodified_recommended_action']=='decelerate'


def test_close_obstacle_is_must_stop():
    r=risk();e=clearance(np.vstack([road_cloud(),[5.,0.,11.]]))
    result=RiskBehaviorPolicy().apply(r,e,frame_id='f10',timestamp_s=1.,ego_speed_kmh=0.,parsed_intent='TURN_RIGHT')
    assert result['behavior_policy']['disposition']=='MUST_STOP'


@pytest.mark.parametrize('distance,closing,speed,stop', [(20.,5.,20.,False),(5.,5.,20.,True),(15.,4.,60.,True)])
def test_closing_ttc_slows_early_but_preserves_actual_braking_distance(distance,closing,speed,stop):
    policy=RiskBehaviorPolicy()
    for i in range(8):
        now=1.+i*.1;frame='ttc'+str(i);r=risk(frame,now,'low')
        r['risk_evidence']['sensor_frame']=i
        r['physical_forward_radar'].update(sensor_frame=i,nearest_distance_m=distance,
            obstacle_candidate_count=1,nearest_closing_distance_m=distance,nearest_closing_velocity_mps=closing)
        e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=50.)
        result=policy.apply(r,e,frame_id=frame,timestamp_s=now,ego_speed_kmh=speed,parsed_intent='KEEP_LANE')
    p=result['behavior_policy']
    assert p['physical_stop'] is stop
    if not stop:
        permission=cautious_permission(result)
        assert permission is not None and permission['max_speed_kmh']<=10.
        assert permission['allow_positive_acceleration'] is False
    else:
        assert cautious_permission(result) is None


@pytest.mark.parametrize('verified', [False,True])
def test_radar_surface_forecast_requires_measured_transverse_motion(verified):
    entity=dict(kind='UNCLASSIFIED_RADAR_SURFACE',status='TRACKED',age_s=0.,
        position_xy_m=[5.,4.],velocity_xy_mps=[0.,-3.],position_std_m=.1,
        observed_velocity_xy_mps=[0.,-3.],motion_sample_count=4 if verified else 1,
        motion_observation_span_s=.3 if verified else 0.,motion_position_residual_m=.01)
    assert potential_path_intrusion({'entities':[entity]},[[0.,0.],[40.,0.]],
        half_width_m=1.,ego_front_m=2.,speed_mps=5.) is verified


def test_tentative_in_corridor_radar_surface_still_blocks_immediately():
    entity=dict(kind='UNCLASSIFIED_RADAR_SURFACE',status='TENTATIVE',age_s=0.,
        position_xy_m=[5.,0.],velocity_xy_mps=[0.,0.],position_std_m=.1)
    assert potential_path_intrusion({'entities':[entity]},[[0.,0.],[40.,0.]],
        half_width_m=1.,ego_front_m=2.,speed_mps=5.)


def test_unclear_motion_stops_even_after_liveness_override():
    r=risk();r=RiskBehaviorPolicy().apply(r,clearance(np.empty((0,3))),frame_id='f10',timestamp_s=1.,ego_speed_kmh=0.,parsed_intent='TURN_RIGHT')
    d,_=apply_behavior_decision(dict(action='accelerate',target_speed_kmh=40.,frame_id='f10'),r)
    assert d['action']=='stop' and not d['allow_positive_acceleration']


def test_high_risk_never_becomes_crawl():
    r=risk(level='high');r=RiskBehaviorPolicy().apply(r,clearance(),frame_id='f10',timestamp_s=1.,ego_speed_kmh=0.,parsed_intent='TURN_RIGHT')
    d,_=apply_behavior_decision(dict(action='accelerate',target_speed_kmh=40.,frame_id='f10'),r)
    assert d['action']=='emergency_brake' and d['target_speed_kmh']==0.


@pytest.mark.parametrize('action',['turn_right','turn_left','keep_lane','decelerate'])
def test_bounded_motion_can_restart_from_rest_without_erasing_turn(action):
    r=confirmed();permission=cautious_permission(r)
    d=dict(action=action,target_speed_kmh=20.,target_location={'x':10,'y':10},frame_id=r['frame_id'],
           decision_status='READY',blocked_reason_codes=[],source_step_action='TURN')
    d,_=apply_behavior_decision(d,r);d,diag=enforce_longitudinal_contract(d,r,0.)
    assert d['action']==action and d['target_speed_kmh']==6.
    assert d['target_location']=={'x':10,'y':10} and d['risk_motion_permission']==permission
    assert d['allow_positive_acceleration'] and diag['mode']=='CAUTIOUS_SPEED_CAP'


@pytest.mark.parametrize('fault',['expired','frame','emergency','blocked'])
def test_permission_cannot_bypass_stop_or_readiness(fault):
    r=confirmed();d=dict(action='turn_right',target_speed_kmh=15.,frame_id=r['frame_id'],decision_status='READY',blocked_reason_codes=[])
    if fault=='expired':r['risk_evidence']['decision_timestamp_s']+=1.
    if fault=='frame':d['frame_id']='wrong'
    if fault=='emergency':d.update(action='emergency_brake',emergency=True,target_speed_kmh=0.)
    if fault=='blocked':d.update(action='stop',decision_status='BLOCKED',blocked_reason_codes=['target_missing'],target_speed_kmh=0.)
    result,_=apply_behavior_decision(d,r)
    assert result['action'] in ('stop','emergency_brake') and result['target_speed_kmh']==0.


def test_turn_plan_can_recover_under_confirmed_bounded_risk():
    document,world,_,_=integration_documents(parser_action='TURN')
    document['intent']['steps'][0].update(parameters={'direction':'RIGHT','target_location':{'x':10.,'y':10.}},
        completion={'type':'JUNCTION_EXITED'})
    runtime=DrivingPlanRuntime();r=confirmed()
    for i in range(9):
        now=2.+i*.1;frame='plan'+str(i);world['frame_id']=frame
        r['frame_id']=frame;r['risk_evidence']['decision_timestamp_s']=now
        r['physical_forward_radar']['sensor_frame']=r['behavior_policy']['lidar']['sensor_frame']
        r['behavior_policy'].update(frame_id=frame,valid_until_s=now+.25)
        r['behavior_policy']['lidar']['observed_at_s']=now
        runtime.prepare(document,frame_id=frame,timestamp_s=now,speed_mps=1.)
        if i==0:runtime.risk_pause=True
        decision=runtime.advance(world,r)
    assert not runtime.risk_pause and decision['decision_status']=='READY'
    assert decision['action']=='turn_right' and decision['target_location'] is not None


def test_stop_proposal_is_not_overridden_by_cautious_navigation():
    r=confirmed();d=dict(action='stop',target_speed_kmh=0.,frame_id=r['frame_id'],source_step_action='STOP')
    out,_=apply_behavior_decision(d,r)
    assert out['action']=='stop' and out['target_speed_kmh']==0.


def test_final_cautious_cap_does_not_erase_traffic_or_instruction_acceleration_block():
    r=confirmed();d=dict(action='turn_right',target_speed_kmh=2.,frame_id=r['frame_id'],
        decision_status='READY',blocked_reason_codes=[],allow_positive_acceleration=False,
        reason='traffic_signal_stopline_constraint')
    out,_=apply_behavior_decision(d,r)
    assert out['target_speed_kmh']==2. and out['allow_positive_acceleration'] is False


def test_new_close_radar_obstacle_revokes_previously_issued_permission():
    r=confirmed();r['physical_forward_radar'].update(nearest_distance_m=2.,obstacle_candidate_count=1)
    assert cautious_permission(r) is None
    d,_=apply_behavior_decision(dict(action='turn_right',target_speed_kmh=6.,frame_id=r['frame_id']),r)
    assert d['action']=='stop' and d['target_speed_kmh']==0.


def test_gate_retains_canonical_turn_during_cautious_deceleration():
    doc,w,a,_=integration_documents(parser_action='TURN')
    from scene_understanding.src.control_decision import build_control_decision
    r=confirmed();w['frame_id']=r['frame_id'];a['world_state_frame_id']=r['frame_id']
    doc['intent']['steps'][0]['parameters']={'direction':'RIGHT','target_location':{'x':10.,'y':10.}}
    canonical=build_control_decision(doc,w,a,r)
    proposal=dict(schema_version='1.0.0',request_id=canonical['request_id'],frame_id=canonical['frame_id'],
        action='decelerate',target_speed_kmh=10.,target_lane=None,target_location=None,target_entity_id=None,
        confidence=.9,model='fixture',latency_ms=1.)
    out=gate_vla_proposal(proposal,canonical,r)
    assert out['action']=='turn_right' and out['target_location']==canonical['target_location']


def test_downstream_cautious_deceleration_retains_route_target():
    r=confirmed();canonical=dict(action='turn_right',target_speed_kmh=6.,frame_id=r['frame_id'],
        target_location={'x':10,'y':10},decision_status='READY')
    result,_=DrivingPlanRuntime().enforce_execution(dict(canonical,action='decelerate',target_location=None),canonical,risk=r)
    assert result['action']=='turn_right' and result['target_location']==canonical['target_location']


@pytest.mark.parametrize('age,y_velocity,expected',[(.1,-3.,True),(.1,0.,False),(.6,-3.,False)])
def test_fresh_lateral_entrant_is_screened_without_blocking_stable_adjacent_track(age,y_velocity,expected):
    observation=dict(entities=[dict(age_s=age,position_xy_m=[6.,4.],velocity_xy_mps=[0.,y_velocity],position_std_m=.1)])
    assert potential_path_intrusion(observation,[(0,0,10),(20,0,10)],half_width_m=1.,ego_front_m=2.,speed_mps=1.) is expected


@pytest.mark.parametrize('uncertainty',[.1,.5,1.])
@pytest.mark.parametrize('drift',[0.,-.02,.02])
def test_adjacent_static_surface_uncertainty_is_not_path_intrusion(uncertainty,drift):
    observation=dict(entities=[dict(age_s=.1,position_xy_m=[10.6,2.08],
        velocity_xy_mps=[0.,drift],position_std_m=uncertainty)])
    assert not potential_path_intrusion(observation,[(0,0),(20,0)],
        half_width_m=1.268,ego_front_m=2.446,speed_mps=0.)


@pytest.mark.parametrize('velocity',[0.,-1.])
def test_physical_in_corridor_surface_is_still_protected(velocity):
    observation=dict(entities=[dict(age_s=.1,position_xy_m=[5.,1.],
        velocity_xy_mps=[0.,velocity],position_std_m=.8)])
    assert potential_path_intrusion(observation,[(0,0),(20,0)],
        half_width_m=1.268,ego_front_m=2.446,speed_mps=0.)


def test_new_crossing_hazard_immediately_revokes_medium_motion():
    r=confirmed();e=r['behavior_policy']['lidar']
    result=RiskBehaviorPolicy().apply(r,e,frame_id=r['frame_id'],timestamp_s=e['observed_at_s'],
        ego_speed_kmh=0.,parsed_intent='TURN_RIGHT',crossing_hazard=True)
    assert result['behavior_policy']['disposition']=='MUST_STOP'


@pytest.mark.parametrize('level',['medium','high'])
def test_fresh_repeated_bundle_preserves_continuous_clearance(level):
    policy=RiskBehaviorPolicy()
    for i in range(28):
        now=1.+i*.05
        sensor=i//2
        observed_at=1.+sensor*.1
        r=risk('f'+str(i),now,level)
        if level=='high':
            r.update(source='learned_raw_camera_visual_risk_head',
                     prediction_provenance={'residual_applied':False})
        e=clearance();e.update(sensor_frame=sensor,observed_at_s=observed_at)
        r['risk_evidence']['sensor_frame']=sensor
        r['physical_forward_radar']['sensor_frame']=sensor
        result=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,
                            ego_speed_kmh=0.,parsed_intent='KEEP_LANE')
    assert result['behavior_policy']['clear_since_s']==1.
    permission=cautious_permission(result)
    assert permission and permission.get('allow_positive_acceleration',True)


def test_repeated_bundle_expires_without_being_retimestamped():
    policy=RiskBehaviorPolicy()
    for i in range(20):
        now=1.+i*.05;r=risk('f'+str(i),now)
        e=clearance()
        result=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,
                            ego_speed_kmh=0.,parsed_intent='KEEP_LANE')
    assert cautious_permission(result) is None
    assert result['behavior_policy']['clear_since_s'] is None
