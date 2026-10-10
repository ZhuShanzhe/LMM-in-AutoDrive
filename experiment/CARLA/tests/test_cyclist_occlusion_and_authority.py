from collections import deque
import copy
import numpy as np
import pytest

from control.observed_vehicle_reidentification import occlusion_score
from control.passing_conditions import PassingMemory
from lightweight_vla_adapter.src.risk_behavior import ground_established_passing, cautious_permission


def fixture():
    road=np.column_stack([np.arange(-30.,100.),np.zeros(130),np.zeros(130)])
    appearance=np.array([1.,0.])
    track=dict(category='cyclist',position=np.array([10.,3.]),velocity=np.array([3.,0.]),
        last_seen_s=1.,velocity_valid=True,motion_history_established=True,
        history=deque([(i/10,np.array([7+3*i/10,3.])) for i in range(11)],maxlen=80),
        appearances=deque([appearance],maxlen=8))
    measurement=dict(category='pedestrian',position=np.array([13.,3.,1.]),box_cropped=False,
        appearance=appearance,view='rear',local_id='reappeared-rider',
        surface=np.array([[13,2.9,1],[13,3.1,1],[13.1,2.9,1],[13.1,3.1,1]]))
    return road,track,measurement


@pytest.mark.parametrize('category',['cyclist','motorcycle','pedestrian'])
def test_current_body_on_observed_road_recovers_known_cyclist(category):
    road,t,m=fixture();m['category']=category
    assert occlusion_score(t,m,2.,road)==pytest.approx(0.)


def test_recovery_follows_road_curve_not_straight_world_prediction():
    _,t,m=fixture()
    road=np.array([[0,0,0],[10,0,0],[20,0,0],[20,10,0],[20,20,0]])
    t.update(position=np.array([15.,0.]),velocity=np.array([5.,0.]))
    m.update(position=np.array([20.,5.,1.]),surface=np.array([[20.,5.,1.]]*4))
    assert occlusion_score(t,m,3.,road)==pytest.approx(0.)


def test_ambiguous_measured_riders_cannot_recover_a_bound_identity():
    road,t,m=fixture();memory=PassingMemory();memory.tracks['a']=t
    other=copy.deepcopy(t);other['position'][0]+=.1;memory.tracks['b']=other
    memory.update([m],frame=40,timestamp_s=2.,ego_xy=[0,0],heading=[1,0],road=road)
    assert ('rear','reappeared-rider') not in memory.aliases
    assert memory.facts=={}


@pytest.mark.parametrize('fault',['stale','unknown_motion','color','lateral','crop','three_points','nonfinite','wrong_class'])
def test_recovery_rejects_contradictory_or_missing_identity_evidence(fault):
    road,t,m=fixture();now=2.
    if fault=='stale':now=9.01
    if fault=='unknown_motion':t['velocity_valid']=False
    if fault=='color':m['appearance']=np.array([0.,1.])
    if fault=='lateral':m['position'][1]+=2.
    if fault=='crop':m['box_cropped']=True
    if fault=='three_points':m['surface']=m['surface'][:3]
    if fault=='nonfinite':m['surface'][0,0]=np.nan
    if fault=='wrong_class':m['category']='vehicle'
    assert occlusion_score(t,m,now,road) is None


def test_recovered_rider_needs_new_velocity_before_completion():
    road,t,m=fixture();memory=PassingMemory();memory.tracks['known']=t
    memory.bound[('r','p')]=dict(target_id='known',bound_at_s=.5,
        clear_since_s=None,last_frame=None,last_time=None)
    document=dict(request_id='r',intent=dict(steps=[dict(step_id='p',action='PASS_BY',parameters=dict(condition='CYCLIST'))]))
    for frame in range(40,56):
        now=frame*.05;x=13.+3.*(now-2.)
        measured=copy.deepcopy(m);measured['position'][0]=x;measured['surface'][:,0]+=x-13.
        memory.update([measured],frame=frame,timestamp_s=now,ego_xy=[50,0],heading=[1,0],road=road)
        proof=memory.proofs(document,frame_id=f'f{frame}',frame=frame,timestamp_s=now,
            observed_at_s=now,ego_xy=[50,0],heading=[1,0],ego_rear_m=2.4,corridor_clear=True)['p']
        assert memory.aliases[('rear','reappeared-rider')]=='known'
        if frame<43:assert not proof['satisfied']
    assert proof['satisfied']
    assert memory.tracks['known']['velocity_valid']
    assert memory.tracks['known']['category']=='cyclist'


def risk_fixture():
    risk=dict(frame_id='f',risk_level='high',source='learned_raw_camera_visual_risk_head',
        reason_codes=['learned_visual_risk_high'],prediction_provenance={'risk_residual_applied':False},
        recommended_action='monitor',risk_evidence=dict(is_fresh=True,sensor_frame=100,decision_timestamp_s=5.),
        physical_forward_radar=dict(schema_version='physical_front_radar/1.0',sensor_frame=100,
            obstacle_candidate_count=0,nearest_distance_m=None),
        behavior_policy=dict(schema_version='risk_behavior/2.0',frame_id='f',disposition='CAUTIOUS_PROCEED',
            confirmed=True,visual_advisory=True,physical_stop=False,max_speed_kmh=10.,required_clearance_m=5.59,
            valid_until_s=5.2,high_clear_since_s=2.,raw_physical_speed_cap_kmh=24.,
            recent_physical_speed_cap_kmh=22.,closing_ttc_response='NONE',bounded_short_horizon=True,
            lidar=dict(valid=True,sensor_frame=100,observed_at_s=5.,clear_distance_m=18.,nearest_obstacle_m=None)))
    decision=dict(frame_id='f',source_step_action='PASS_BY',passing_lane_established=True,decision_status='READY')
    proof=dict(schema_version='measured_adjacent_passing/1.0',frame_id='f',sensor_frame=100,observed_at_s=5.,
        target_id='bike',target_category='cyclist',coordinate_frame='MEASURED_ROAD_LOCAL',velocity_valid=True,
        lateral_m=3.5,transverse_speed_mps=0.,corridor_clear=True,origin_side='RIGHT',policy_truth_access=False)
    return risk,decision,proof


def test_safe_established_cyclist_pass_is_bounded_by_physical_speed():
    r,d,p=risk_fixture();out=ground_established_passing(r,d,p)
    assert out['risk_level']=='high'
    assert cautious_permission(out)['max_speed_kmh']==22.
    assert r['behavior_policy']['max_speed_kmh']==10.
    assert not out.get('task_completed')


@pytest.mark.parametrize('fault',['unestablished','same_lane','cutin','stale','unknown_motion','no_road',
    'blocked','short_physical','emergency','lidar_obstacle','radar_too_close','physical_high'])
def test_cyclist_authority_cannot_override_uncertainty_or_physical_hazard(fault):
    r,d,p=risk_fixture()
    if fault=='unestablished':d['passing_lane_established']=False
    if fault=='same_lane':p['lateral_m']=1.
    if fault=='cutin':p['transverse_speed_mps']=-1.
    if fault=='stale':p['observed_at_s']=4.
    if fault=='unknown_motion':p['velocity_valid']=False
    if fault=='no_road':p['coordinate_frame']='EGO_TANGENT'
    if fault=='blocked':d['blocked_reason_codes']=['blocked']
    if fault=='short_physical':r['behavior_policy']['recent_physical_speed_cap_kmh']=14.
    if fault=='emergency':r['behavior_policy']['emergency_evidence']=True
    if fault=='lidar_obstacle':r['behavior_policy']['lidar']['nearest_obstacle_m']=14.
    if fault=='radar_too_close':r['physical_forward_radar'].update(obstacle_candidate_count=1,nearest_distance_m=4.)
    if fault=='physical_high':r['reason_codes'].append('physical_forward_radar_emergency_distance')
    assert ground_established_passing(r,d,p)==r
