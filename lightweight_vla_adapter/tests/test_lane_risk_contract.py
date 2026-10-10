import numpy as np
import pytest
from lightweight_vla_adapter.src.lane_risk_contract import LaneRiskContract,lidar_lane_evidence,enforce_requested_lane


def bev(obstacle=False,direction='left'):
    data=np.zeros((4,64,64),dtype=np.float32)
    data[0,0,0]=1.;data[1,0,0]=-.44
    if obstacle:
        row=round((60.-6.)/80.*63);col=round(((-3.5 if direction=='left' else 3.5)+30.)/60.*63)
        data[0,row,col]=1.;data[1,row,col]=-.1
    return data


def visual(direction='left',safe=True):
    return dict(risk_level='low' if safe else 'medium',lane_change={direction:dict(is_safe=safe)})


@pytest.mark.parametrize('direction',['left','right'])
def test_lidar_obstacle_in_requested_corridor(direction):
    assert lidar_lane_evidence(bev(True,direction),direction,5.)['occupied']
    assert not lidar_lane_evidence(bev(True,direction),'right' if direction=='left' else 'left',5.)['occupied']


def test_only_fresh_continuous_clearance_can_release():
    fusion=LaneRiskContract()
    for i in range(8):
        r=fusion.update({},'left',visual(),bev(),timestamp_s=i*.1,sensor_frame=i,frame_age_s=0.,speed_mps=0.)
    assert r['lane_change']['left']['is_safe']
    r=fusion.update({},'left',visual(safe=False),bev(),timestamp_s=.8,sensor_frame=8,frame_age_s=0.,speed_mps=0.)
    assert not r['lane_change']['left']['is_safe']
    r=fusion.update({},'left',visual(),bev(),timestamp_s=.9,sensor_frame=9,frame_age_s=0.,speed_mps=0.)
    assert not r['lane_change']['left']['is_safe']


@pytest.mark.parametrize('age,frame',[(1.,8),(0.,7)])
def test_stale_or_reused_frame_is_not_new_clearance(age,frame):
    fusion=LaneRiskContract()
    for i in range(8):fusion.update({},'left',visual(),bev(),timestamp_s=i*.1,sensor_frame=i,frame_age_s=0.,speed_mps=0.)
    r=fusion.update({},'left',visual(),bev(),timestamp_s=.8,sensor_frame=frame,frame_age_s=age,speed_mps=0.)
    assert not r['lane_change']['left']['is_safe']


def test_global_low_risk_cannot_overwrite_side_unsafe():
    fusion=LaneRiskContract()
    r=fusion.update(dict(risk_level='low',lane_change={'left':dict(is_safe=True)}),'left',visual(),bev(True),timestamp_s=0.,sensor_frame=1,frame_age_s=0.,speed_mps=4.)
    final,override=enforce_requested_lane(dict(action='lane_change_left',target_speed_kmh=20.,reason='resume'),r,'left')
    assert final['action']=='stop' and final['target_speed_kmh']==0. and override


def test_invalid_lidar_does_not_mean_clear():
    assert not lidar_lane_evidence(np.zeros((4,64,64)),'left',0.)['valid']


def test_lane_identity_follows_topology_across_road_boundaries():
    from types import SimpleNamespace as Obj
    from lightweight_vla_adapter.src.lane_risk_contract import continued_lane_waypoint
    target=Obj(road_id=2,section_id=0,lane_id=3,transform=Obj(location=Obj(distance=lambda _:1.)))
    previous=Obj(next=lambda distance:[target])
    assert continued_lane_waypoint(previous,Obj(road_id=2,section_id=0),None) is target
    assert continued_lane_waypoint(previous,Obj(road_id=99,section_id=0),None) is None


@pytest.mark.parametrize('continuing,obstacle,expected',[(False,False,False),(True,False,True),(True,True,False)])
def test_medium_visual_risk_only_allows_clear_low_speed_completion(continuing,obstacle,expected):
    fusion=LaneRiskContract()
    for i in range(8):
        risk=fusion.update({},'left',visual(safe=False),bev(obstacle),timestamp_s=i*.1,
            sensor_frame=i,frame_age_s=0.,speed_mps=1.,continuing=continuing)
    assert risk['lane_change']['left']['is_safe']==expected
    result,_=enforce_requested_lane(dict(action='lane_change_left',target_speed_kmh=20.),risk,'left')
    assert result['target_speed_kmh']==(5. if expected else 0.)


@pytest.mark.parametrize('entered,front,rear_distance,age,expected',[
    (True,0,6.,0.,True),(False,0,6.,0.,False),(True,1,6.,0.,False),
    (True,0,4.,0.,False),(True,0,6.,1.,False),(True,0,float('nan'),0.,False)])
def test_entered_lane_can_finish_only_with_fresh_far_rear_only_occupancy(entered,front,rear_distance,age,expected):
    fusion=LaneRiskContract()
    evidence=dict(valid=True,observed=True,occupied=True,obstacle_points=10,
        front_obstacle_points=front,rear_obstacle_points=10-front,
        rear_obstacle_nearest_m=rear_distance,
        source='same_frame_lidar_target_lane_geometry',front_horizon_m=20.,rear_horizon_m=10.)
    for i in range(8):
        risk=fusion.update({},'right',dict(visual('right'),source='measured_view'),bev(),
            timestamp_s=i*.1,sensor_frame=i,frame_age_s=age,speed_mps=0.,
            continuing=True,target_entered=entered,physical_evidence=evidence,road_legal=True)
    assert risk['lane_change']['right']['is_safe'] is expected
    assert risk['target_lane_contract']['lidar']['occupied'] is True
    if expected:
        assert risk['lane_change']['right']['speed_cap_kmh']==8.


def test_measured_lane_keeps_front_and_rear_obstacle_evidence_separate():
    from lightweight_vla_adapter.src.lane_risk_contract import measured_lane_clearance
    ground=[[x,y,0.] for x in np.arange(-9.5,20.,1.) for y in (-.4,.4)]
    rear=[[-6.,0.,1.]]
    road=[[-10.,0.,0.],[0.,0.,0.],[20.,0.,0.]]
    evidence=measured_lane_clearance(np.asarray(ground+rear),road,[0.,0.,0.])
    assert evidence['occupied'] and evidence['observed']
    assert evidence['front_obstacle_points']==0 and evidence['rear_obstacle_points']==1
    assert evidence['rear_obstacle_nearest_m']==pytest.approx(6.)
    evidence=measured_lane_clearance(np.asarray(ground+rear+[[2.,0.,1.]]),road,[0.,0.,0.])
    assert evidence['front_obstacle_points']==1
