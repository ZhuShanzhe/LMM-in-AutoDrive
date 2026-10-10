from types import SimpleNamespace as NS
import numpy as np
import pytest
from control.generic_route_pid import GenericRoutePID
from lightweight_vla_adapter.src.risk_behavior import lidar_corridor_evidence


@pytest.mark.parametrize('height',[0.,10.,35.])
@pytest.mark.parametrize('translated',[False,True])
def test_selected_route_level_is_retained(height,translated):
    controller=GenericRoutePID.__new__(GenericRoutePID)
    location=NS(x=0.,y=0.,z=height+.1)
    controller.ego=NS(get_location=lambda:location)
    calls=[]
    def current_waypoint(location,**kwargs):
        calls.append(location)
        return NS(transform=NS(location=NS(z=height)))
    controller.world=NS(get_map=lambda:NS(get_waypoint=current_waypoint))
    controller.progress_m=lambda:0.
    controller.route_context=NS(distances_m=[0.,2.,4.,6.])
    controller.route_plan=[(NS(transform=NS(location=NS(x=i*2.,y=0.,z=height+i*.1))),None)
        for i in range(4)]
    controller.logical_lane_adapter=object() if translated else None
    controller._waypoint_target=lambda offset,**snapshot:dict(x=offset,y=0.,z=height+offset*.05)
    points=controller.radar_corridor_polyline(20.,include_height=True)
    np.testing.assert_allclose(points,[(0.,0.,height),(2.,0.,height+.1),
        (4.,0.,height+.2),(6.,0.,height+.3)],rtol=0.,atol=1e-10)
    assert calls[0] is location and len(calls)==2
    assert calls[1].z==pytest.approx(height+.3)


def test_static_map_is_cached_per_controller():
    controller=GenericRoutePID.__new__(GenericRoutePID)
    calls=[]
    road_map=object()
    controller.world=NS(get_map=lambda:calls.append('map') or road_map)
    assert controller._sensor_road_map() is road_map
    assert controller._sensor_road_map() is road_map
    assert calls==['map']


def test_corridor_reuses_one_ego_snapshot_for_all_targets():
    controller=GenericRoutePID.__new__(GenericRoutePID)
    calls=[]
    location=NS(x=0.,y=0.)
    controller.ego=NS(get_location=lambda:calls.append('ego') or location)
    controller.progress_m=lambda:0.
    controller.route_context=NS(distances_m=[0.,2.,4.,6.])
    controller.route_plan=[(None,None)]*4
    controller.logical_lane_adapter=object()
    def target(offset,**snapshot):
        assert snapshot['progress_m']==0.
        assert snapshot['ego_location'] is location
        return dict(x=offset,y=0.)
    controller._waypoint_target=target
    controller.radar_corridor_polyline(20.)
    assert calls==['ego']


def evidence(road,points):
    return lidar_corridor_evidence(points,road,half_width_m=1.,ego_front_m=2.,
        sensor_frame=1,observed_at_s=1.)


def test_cross_level_reprojection_is_invalid_not_an_obstacle():
    ground=np.array([[x,y,10.] for x in np.arange(3.,25.,.1) for y in [-.2,.2]])
    result=evidence([[0.,0.,10.],[5.,0.,10.],[7.,0.,0.],[25.,0.,0.]],ground)
    assert not result['valid']
    assert result['reason']=='implausible_route_surface_grade'
    assert result['nearest_obstacle_m'] is None


@pytest.mark.parametrize('grade',[-.15,0.,.15])
def test_real_slopes_keep_ground_and_obstacle_evidence(grade):
    ground=np.array([[x,y,10.+grade*x] for x in np.arange(3.,25.,.1) for y in [-.2,.2]])
    obstacle=np.array([[12.,y,11.+grade*12.] for y in np.linspace(-.2,.2,20)])
    result=evidence([[0.,0.,10.],[25.,0.,10.+grade*25.]],np.concatenate([ground,obstacle]))
    assert result['valid']
    assert result['nearest_obstacle_m']==10.
    assert result['obstacle_points']==20
