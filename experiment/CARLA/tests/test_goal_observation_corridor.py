"""Mission boundaries cannot truncate a physically measurable road corridor."""
from types import SimpleNamespace as NS
import carla
from control.generic_route_pid import GenericRoutePID


class Waypoint:
    lane_type=carla.LaneType.Driving
    def __init__(self,x,z=0.,yaw=0.,fork=False):
        self.transform=NS(location=NS(x=x,y=0.,z=z),rotation=NS(yaw=yaw))
        self.fork=fork
    def next(self,distance):
        point=Waypoint(self.transform.location.x+distance)
        return [point,Waypoint(point.transform.location.x)] if self.fork else [point]


def controller(waypoint=None,end=4.):
    value=GenericRoutePID.__new__(GenericRoutePID)
    value.route_context=NS(distances_m=[0.,end])
    value.route_plan=[(Waypoint(end),None)]
    value.route_manager=None
    value._sensor_road_map=lambda:NS(get_waypoint=lambda *args,**kwargs:waypoint or Waypoint(4.))
    return value


def test_sensor_geometry_extends_but_execution_goal_is_unchanged():
    c=controller();points=[(0.,0.,0.),(4.,0.,0.)]
    result=c._extend_observation_after_goal(points,30.,progress_m=0.,include_height=True)
    assert result[-1][0]>=30. and result[:2]==points
    assert points==[(0.,0.,0.),(4.,0.,0.)]
    assert c.route_context.distances_m[-1]==4. and c.route_plan[-1][0].transform.location.x==4.


def test_unknown_branch_is_never_selected_to_invent_free_space():
    points=[(0.,0.,0.),(4.,0.,0.)]
    assert controller(Waypoint(4.,fork=True))._extend_observation_after_goal(
        points,30.,progress_m=0.,include_height=True)==points


def test_wrong_road_level_and_reverse_geometry_are_rejected():
    points=[(0.,0.,0.),(4.,0.,0.)]
    for waypoint in (Waypoint(4.,z=8.),Waypoint(4.,yaw=180.)):
        assert controller(waypoint)._extend_observation_after_goal(
            points,30.,progress_m=0.,include_height=True)==points


def test_unfinished_mission_keeps_planned_sensor_geometry():
    points=[(0.,0.,0.),(4.,0.,0.)]
    assert controller(end=1000.)._extend_observation_after_goal(
        points,30.,progress_m=0.,include_height=True)==points
