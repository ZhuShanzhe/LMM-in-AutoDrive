import pytest
from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy,cautious_permission,apply_behavior_decision
from lightweight_vla_adapter.tests.test_risk_behavior import risk,clearance


def observation(policy,i,*,fault=None):
    now=1.+i*.1;frame=f'f{i}';r=risk(frame,now,'medium');e=clearance()
    e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=25.)
    r['risk_evidence']['sensor_frame']=r['physical_forward_radar']['sensor_frame']=i
    if fault=='stale':e['observed_at_s']=now-.4
    if fault=='near_obstacle':e['nearest_obstacle_m']=.4
    if fault=='no_support':e['clear_distance_m']=0.
    if fault=='ttc':r['physical_forward_radar'].update(nearest_closing_distance_m=5.,nearest_closing_velocity_mps=5.)
    return policy.apply(r,e,frame_id=frame,timestamp_s=now,ego_speed_kmh=0.,
        parsed_intent='YIELD',task_conditioned=True,crossing_hazard=fault=='crossing')


def test_yield_approach_confirms_clearance_without_permanent_mid_risk_stop():
    policy=RiskBehaviorPolicy()
    for i in range(9):r=observation(policy,i)
    permit=cautious_permission(r)
    assert permit and permit.get('allow_positive_acceleration',True)
    d=dict(action='decelerate',target_speed_kmh=10.,frame_id=r['frame_id'],
           source_step_action='ADJUST_SPEED',decision_status='READY')
    final,_=apply_behavior_decision(d,r,allow_restart=True)
    assert final['target_speed_kmh']==10. and final['action']=='decelerate'


@pytest.mark.parametrize('fault',['stale','near_obstacle','no_support','ttc','crossing'])
def test_yield_never_overrides_physical_or_missing_evidence(fault):
    policy=RiskBehaviorPolicy()
    for i in range(9):observation(policy,i)
    r=observation(policy,9,fault=fault)
    assert cautious_permission(r) is None
    d=dict(action='decelerate',target_speed_kmh=10.,frame_id=r['frame_id'],source_step_action='ADJUST_SPEED')
    final,_=apply_behavior_decision(d,r,allow_restart=True)
    assert final['action'] in {'stop','emergency_brake'} and final['target_speed_kmh']==0.
