import pytest
from benchmark.traffic_compliance import crossing_evidence


def line():return dict(id='line',xy=[0.,0.],yaw_deg=0.,lane_width_m=3.5,lane_key=(1,0,-1))
def pose(x,y=0.,yaw=0.,lane=(1,0,-1)):
    return dict(front_xy=[x,y],yaw_deg=yaw,lane_key=lane)


@pytest.mark.parametrize('before,after,status',[
    ('Red','Red','RED_LIGHT_VIOLATION'),('Green','Green','PERMITTED_CROSSING'),
    ('Yellow','Yellow','PERMITTED_CROSSING'),('Red','Green','SIGNAL_TRANSITION_CROSSING'),
    ('UNAVAILABLE','Red','UNAVAILABLE'),('CONFLICTING_STATES','Red','UNAVAILABLE')])
def test_signal_crossing_status(before,after,status):
    assert crossing_evidence(pose(-.1),pose(.1),line(),before,after)['status']==status


@pytest.mark.parametrize('previous,current',[
    (None,pose(.1)),(pose(-.1),pose(-.01)),(pose(.1),pose(.2)),
    (pose(-.1,3.5),pose(.1,3.5)),(pose(-.1,yaw=180),pose(.1,yaw=180)),
    (pose(-.1,lane=(2,0,-1)),pose(.1,lane=(2,0,-1)))])
def test_unrelated_or_uncrossed_stop_plane_does_not_claim_violation(previous,current):
    assert crossing_evidence(previous,current,line(),'Red','Red') is None


def test_large_displacement_is_not_counted_as_normal_traffic_crossing():
    assert crossing_evidence(pose(-3.),pose(3.),line(),'Red','Red')['status']=='INVALID_DISCONTINUITY'
