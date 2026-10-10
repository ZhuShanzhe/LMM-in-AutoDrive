from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.risk_behavior import ground_established_passing, cautious_permission

def fixture():
    r=dict(frame_id='f',risk_level='high',source='learned_raw_camera_visual_risk_head',
        reason_codes=['learned_visual_risk_high'],prediction_provenance={'risk_residual_applied':False},
        recommended_action='monitor',risk_evidence=dict(is_fresh=True,sensor_frame=100,decision_timestamp_s=5.),
        physical_forward_radar=dict(schema_version='physical_front_radar/1.0',sensor_frame=100,
            obstacle_candidate_count=0,nearest_distance_m=None),
        behavior_policy=dict(schema_version='risk_behavior/2.0',frame_id='f',disposition='CAUTIOUS_PROCEED',
            confirmed=True,visual_advisory=True,physical_stop=False,max_speed_kmh=10.,required_clearance_m=5.59,
            valid_until_s=5.2,high_clear_since_s=2.,raw_physical_speed_cap_kmh=40.,recent_physical_speed_cap_kmh=38.,closing_ttc_response='NONE',
            lidar=dict(valid=True,sensor_frame=100,observed_at_s=5.,clear_distance_m=31.,nearest_obstacle_m=None)))
    d=dict(frame_id='f',source_step_action='OVERTAKE',passing_lane_established=True,decision_status='READY')
    p=dict(schema_version='measured_adjacent_passing/1.0',frame_id='f',sensor_frame=100,observed_at_s=5.,
        target_id='body4',lateral_m=3.,transverse_speed_mps=0.,corridor_clear=True,origin_side='RIGHT',policy_truth_access=False)
    return r,d,p

def test_grounded_passing_is_speed_authority_not_class_or_completion():
    r,d,p=fixture();out=ground_established_passing(r,d,p)
    assert out['risk_level']=='high' and r['behavior_policy']['max_speed_kmh']==10.
    assert cautious_permission(out)['max_speed_kmh']==30.
    assert not out.get('task_completed')

@pytest.mark.parametrize('fault', ['stale','frame','cutin','same_lane','no_target','unclear','truth',
    'no_lane','blocked','emergency','short','early','radar','lidar','physical_reason','physical_cap'])
def test_unsafe_or_unknown_conditions_do_not_expand_authority(fault):
    r,d,p=fixture()
    if fault=='stale':p['observed_at_s']=4.
    if fault=='frame':p['sensor_frame']=99
    if fault=='cutin':p['transverse_speed_mps']=-1.
    if fault=='same_lane':p['lateral_m']=1.
    if fault=='no_target':p['target_id']=None
    if fault=='unclear':p['corridor_clear']=False
    if fault=='truth':p['policy_truth_access']=True
    if fault=='no_lane':d['passing_lane_established']=False
    if fault=='blocked':d['blocked_reason_codes']=['unknown']
    if fault=='emergency':r['behavior_policy']['emergency_evidence']=True
    if fault=='short':r['behavior_policy']['bounded_short_horizon']=True
    if fault=='early':r['behavior_policy']['high_clear_since_s']=4.
    if fault=='radar':r['physical_forward_radar'].update(obstacle_candidate_count=1,nearest_distance_m=20.)
    if fault=='lidar':r['behavior_policy']['lidar']['nearest_obstacle_m']=20.
    if fault=='physical_reason':r['reason_codes'].append('physical_forward_radar_emergency_distance')
    if fault=='physical_cap':r['behavior_policy']['raw_physical_speed_cap_kmh']=20.
    assert ground_established_passing(r,d,p)==r

def test_certificate_expiry_revokes_permission():
    r,d,p=fixture();out=ground_established_passing(r,d,p)
    out['risk_evidence']['decision_timestamp_s']=5.4
    assert cautious_permission(out) is None
