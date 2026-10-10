from types import SimpleNamespace as NS
import pytest
from control.generic_route_pid import GenericRoutePID


@pytest.mark.parametrize('backend',['route_context','route_manager'])
@pytest.mark.parametrize('height',[True,False])
def test_control_and_sensor_corridor_follow_the_same_completed_lane(backend,height):
    def waypoint(x,y):
        return NS(transform=NS(location=NS(x=x,y=y,z=.1*x),rotation=NS(yaw=0.)))
    current=waypoint(0.,3.5);original=waypoint(30.,0.)
    c=GenericRoutePID.__new__(GenericRoutePID)
    c._completed_lane_navigation=True
    c.ego=NS(get_location=lambda:NS(x=0.,y=3.5,z=.6),
             get_transform=lambda:NS(rotation=NS(yaw=0.)))
    c._sensor_road_map=lambda:NS(get_waypoint=lambda *a,**kw:current)
    c._current_lane_navigation_point=lambda distance,**kw:waypoint(distance,3.5)
    c.progress_m=lambda:0.
    c._extend_observation_after_goal=lambda points,*a,**kw:points
    c.route_context=NS(distances_m=[30.]) if backend=='route_context' else None
    c.route_plan=[(original,None)] if backend=='route_context' else None
    c.route_manager=NS(route=[dict(distance_m=30.,x=30.,y=0.,z=3.)]) if backend=='route_manager' else None
    result=c.radar_corridor_polyline(30.,include_height=height)
    assert len(result)>=6 and all(point[1]==3.5 for point in result)
    if height:assert all(point[2]==.1*point[0] for point in result)


def test_ambiguous_completed_lane_cannot_fall_back_to_a_different_sensor_lane():
    c=GenericRoutePID.__new__(GenericRoutePID)
    c._completed_lane_navigation=True
    c.ego=NS(get_location=lambda:NS(x=0.,y=3.5,z=.6),
             get_transform=lambda:NS(rotation=NS(yaw=0.)))
    c._current_lane_navigation_point=lambda *a,**kw:None
    c._route_target=lambda:None
    c.progress_m=lambda:0.
    c._extend_observation_after_goal=lambda points,*a,**kw:points
    assert c.radar_corridor_polyline(30.)==[(0.,3.5)]
