import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_ground_filter import filter_ground


def observation(height, distance=10.):
    p=dict(distance_m=float(np.hypot(distance,height-1)), relative_velocity_mps=0.,
        closing_speed_mps=0., azimuth_deg=0., altitude_deg=float(np.rad2deg(np.arctan2(height-1,distance))),
        relative_height_m=height-1)
    return dict(_ground_candidates=[p],tracking_points=[p],nearest_distance_m=p['distance_m'],
                obstacle_candidate_count=1,azimuth_obstacle_bins=[p])


def cloud(slope=0.):
    x,y=np.meshgrid(np.linspace(7.6,12.4,17),np.linspace(-2.4,2.4,17))
    return np.column_stack([x.ravel(),y.ravel(),(slope*x).ravel()])


def matrix():
    m=np.eye(4);m[2,3]=1.;return m


@pytest.mark.parametrize('slope',[-.10,-.06,0,.06,.10])
def test_ground_on_different_grades(slope):
    o=observation(10*slope)
    result=filter_ground(o,cloud(slope),matrix())
    assert result['nearest_distance_m'] is None
    assert result['tracking_points']==[]
    assert o['obstacle_candidate_count']==1


@pytest.mark.parametrize('height',[.06,.15,.5,1.5])
def test_low_obstacles_and_vehicles_are_retained(height):
    result=filter_ground(observation(height),cloud(),matrix())
    assert result['obstacle_candidate_count']==1


def test_missing_sparse_and_mixed_evidence_keeps_returns():
    o=observation(0.)
    for c in [None, cloud()[:4],np.concatenate([cloud(),cloud()+[0,0,.3]])]:
        assert filter_ground(o,c,matrix())['nearest_distance_m'] is not None


def test_top_surface_not_accepted_as_ground():
    assert filter_ground(observation(.8),cloud()+[0,0,.8],matrix())['nearest_distance_m'] is not None


def test_pose_invariance():
    theta=.7;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    m=matrix();m[:3,:3]=r;m[:3,3]+=[100,30,2]
    c=cloud(.06)@r.T+[100,30,2]
    assert filter_ground(observation(.6),c,m)['nearest_distance_m'] is None


@pytest.mark.parametrize('mode',['matched','missing','timestamp_mismatch','future_only','expired_only'])
def test_sensor_join_is_causal_and_requires_current_frame(mode):
    from collections import OrderedDict
    import threading
    from carla_multiview_sensor import SynchronizedMultiviewCameraRig
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig.enable_radar=True;rig._condition=threading.Condition()
    obs=observation(.6);obs.update(sensor_frame=10,measurement_timestamp_s=1.,_sensor_matrix=matrix().tolist())
    rig._radar_observations={'front':OrderedDict([(10,obs)])}
    rig._frames=OrderedDict()
    if mode!='missing':
        rig._frames[10]=dict(ground_points_world=cloud(.06) if mode in ('matched','timestamp_mismatch') else cloud(.06)[:4],
                            ground_timestamp_s=.9 if mode=='timestamp_mismatch' else 1.)
    if mode=='future_only':rig._frames[11]=dict(ground_points_world=cloud(.06),ground_timestamp_s=1.05)
    if mode=='expired_only':rig._frames[1]=dict(ground_points_world=cloud(.06),ground_timestamp_s=.5)
    result=rig.latest_radar(maximum_frame=10)
    assert (result['nearest_distance_m'] is None)==(mode=='matched')
    assert '_ground_candidates' not in result and '_sensor_matrix' not in result


@pytest.mark.parametrize('height,distance,expected',[(0.,10.,'low'),(.15,10.,'medium'),(.15,5.,'high')])
def test_ground_filter_does_not_disable_physical_safety(height,distance,expected):
    from universal_vla_controller import fuse_forward_radar_risk
    filtered=filter_ground(observation(height,distance),cloud(),matrix())
    risk=fuse_forward_radar_risk(dict(risk_level='low',recommended_action='maintain_speed',reason_codes=[]),
        filtered,ego_speed_kmh=0.)
    assert risk['risk_level']==expected
    if expected=='high':assert risk['recommended_action']=='emergency_brake'


def test_forward_radar_evidence_marks_only_fresh_physical_returns_confirmed():
    from universal_vla_controller import fuse_forward_radar_risk

    learned = dict(risk_level='low', recommended_action='keep_lane', reason_codes=[])
    confirmed = fuse_forward_radar_risk(
        learned,
        dict(sensor_frame=80, measurement_timestamp_s=10., obstacle_candidate_count=1,
             nearest_distance_m=5.),
        ego_speed_kmh=0., decision_timestamp_s=10.1,
    )
    stale = fuse_forward_radar_risk(
        learned,
        dict(sensor_frame=79, measurement_timestamp_s=9., obstacle_candidate_count=1,
             nearest_distance_m=5.),
        ego_speed_kmh=0., decision_timestamp_s=10.1,
    )

    assert confirmed['risk_evidence']['status'] == 'CONFIRMED_HAZARD'
    assert confirmed['risk_evidence']['is_fresh'] is True
    assert confirmed['risk_evidence']['valid_until_s'] == pytest.approx(10.25)
    assert stale['risk_evidence']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert stale['risk_evidence']['is_fresh'] is False
    # Evidence expiry must not quietly clear a potentially hazardous brake.
    assert stale['recommended_action'] == 'emergency_brake'


def test_forward_radar_empty_snapshot_is_not_a_clearance_claim():
    from universal_vla_controller import fuse_forward_radar_risk

    fused = fuse_forward_radar_risk(
        dict(risk_level='high', recommended_action='decelerate',
             reason_codes=['learned_visual_risk_high'], source='event_memory_vla'),
        dict(sensor_frame=90, measurement_timestamp_s=12., obstacle_candidate_count=0,
             nearest_distance_m=None),
        ego_speed_kmh=20., decision_timestamp_s=12.05,
    )

    assert fused['risk_evidence']['status'] == 'MODEL_ONLY'
    assert fused['risk_evidence']['is_fresh'] is True
    assert fused['risk_level'] == 'high'
    assert fused['recommended_action'] == 'decelerate'
