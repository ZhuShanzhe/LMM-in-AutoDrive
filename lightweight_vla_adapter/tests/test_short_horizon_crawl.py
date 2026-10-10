from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy,cautious_permission,apply_behavior_decision
from lightweight_vla_adapter.src.longitudinal_contract import enforce_longitudinal_contract
from lightweight_vla_adapter.tests.test_risk_behavior import risk,clearance


def sample(policy,i,*,level='medium',speed=26.,distance=12.,fault=None):
    now=1.+i*.1;frame=f'f{i}';r=risk(frame,now,level)
    r.update(source='learned_raw_camera_visual_risk_head',
        prediction_provenance=dict(risk_residual_applied=False))
    e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=distance)
    r['risk_evidence']['sensor_frame']=r['physical_forward_radar']['sensor_frame']=i
    if fault=='stale':e['observed_at_s']=now-.4
    if fault=='no_support':e['clear_distance_m']=0.
    if fault=='invalid':e['valid']=False
    if fault=='obstacle':e['nearest_obstacle_m']=10.
    if fault=='absolute_stop':e['nearest_obstacle_m']=.4
    if fault=='near_radar':r['physical_forward_radar'].update(nearest_distance_m=.4,obstacle_candidate_count=1)
    if fault=='ttc':r['physical_forward_radar'].update(nearest_closing_velocity_mps=5.,nearest_closing_distance_m=10.)
    return policy.apply(r,e,frame_id=frame,timestamp_s=now,ego_speed_kmh=speed,
        parsed_intent='STOP'if fault=='stop'else'CHANGE_LANE'if fault=='lane'else'SET_SPEED',
        crossing_hazard=fault=='crossing')


@pytest.mark.parametrize('level',['low','medium','high'])
def test_valid_short_horizon_first_decelerates_instead_of_stopping(level):
    r=sample(RiskBehaviorPolicy(),0,level=level)
    permission=cautious_permission(r)
    assert permission and 5.<=permission['max_speed_kmh']<=10.
    assert permission['allow_positive_acceleration']is False
    d=dict(action='decelerate',target_speed_kmh=30.,frame_id=r['frame_id'],
        decision_status='READY',source_step_action='SET_SPEED')
    d,_=apply_behavior_decision(d,r,allow_restart=True)
    d,_=enforce_longitudinal_contract(d,r,26.)
    assert d['action']=='decelerate' and d['target_speed_kmh']==10.
    assert d['allow_positive_acceleration']is False


@pytest.mark.parametrize('level',['low','medium','high'])
def test_confirmed_short_horizon_uses_measured_bound_instead_of_fixed_crawl(level):
    p=RiskBehaviorPolicy()
    for i in range(16):r=sample(p,i,level=level,speed=26. if i==0 else 8.)
    permit=cautious_permission(r)
    assert permit and (permit['max_speed_kmh']==10. if level=='high' else 10.<permit['max_speed_kmh']<30.)
    assert permit.get('allow_positive_acceleration',True)
    assert r['behavior_policy']['bounded_short_horizon']
    r=sample(p,16,level=level,speed=8.,distance=30.)
    assert not r['behavior_policy']['bounded_short_horizon']


@pytest.mark.parametrize('fault',['stale','no_support','invalid','obstacle','absolute_stop','near_radar','ttc','stop','lane','crossing'])
def test_short_horizon_does_not_override_real_hazards_or_invalid_observation(fault):
    r=sample(RiskBehaviorPolicy(),0,fault=fault)
    assert cautious_permission(r)is None
    if fault in {'absolute_stop','near_radar'}:
        assert r['behavior_policy']['absolute_stop_active']
        assert r['behavior_policy']['response_tier']=='EMERGENCY_STOP'


def test_new_obstacle_revokes_low_speed_authority_immediately():
    p=RiskBehaviorPolicy()
    for i in range(8):r=sample(p,i,speed=26. if i==0 else 8.)
    assert cautious_permission(r)
    r=sample(p,8,fault='obstacle')
    assert cautious_permission(r)is None and r['behavior_policy']['physical_stop']


def test_forged_high_unconfirmed_restart_flag_is_rejected():
    r=sample(RiskBehaviorPolicy(),0,level='high')
    r=deepcopy(r);r['behavior_policy']['allow_positive_acceleration']=True
    assert cautious_permission(r)is None


@pytest.mark.parametrize('level',['medium','high'])
def test_initial_visual_advisory_with_long_clear_horizon_brakes_instead_of_stopping(level):
    r=sample(RiskBehaviorPolicy(),0,level=level,distance=40.)
    permission=cautious_permission(r)
    assert permission and permission['max_speed_kmh']==10.
    assert permission['allow_positive_acceleration']is False


def test_sensor_gap_revokes_provisional_slowdown():
    p=RiskBehaviorPolicy();assert cautious_permission(sample(p,0))
    assert cautious_permission(sample(p,5))is None
