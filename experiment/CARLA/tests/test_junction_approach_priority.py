from types import SimpleNamespace as NS

import carla
import pytest
from control.generic_route_pid import GenericRoutePID
from control.pid_controller import EgoPIDController


def waypoint(road, lane, *, junction=False, z=0., yaw=0.):
    return NS(road_id=road, lane_id=lane, section_id=0, is_junction=junction,
        lane_type=carla.LaneType.Driving,
        transform=NS(location=NS(x=0., y=float(lane)*3.5, z=z), rotation=NS(yaw=yaw)))


def controller(current, near, entry, adjacent, direction, legal=True):
    adjacent.transform.location.y = current.transform.location.y + (
        3.5 if direction == 'right' else -3.5)
    c=GenericRoutePID.__new__(GenericRoutePID)
    c._completed_lane_navigation=True
    c._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    c.ego=NS(get_location=lambda:current.transform.location)
    c.route_manager=None
    c.route_context=NS(distances_m=[0.,15.,30.,100.])
    c.route_plan=[(w,None) for w in (current,near,near,entry)]
    c.progress_m=lambda:0.
    c._pid=EgoPIDController.__new__(EgoPIDController)
    c.autonomous_lane_change_legal=lambda d:legal and d==direction
    current.get_left_lane=lambda:adjacent if direction=='left' else None
    current.get_right_lane=lambda:adjacent if direction=='right' else None
    return c


@pytest.mark.parametrize('direction',['left','right'])
def test_reachable_junction_prevents_near_lane_round_trip(direction):
    current=waypoint(1,-1)
    other=waypoint(1,-2)
    entry=waypoint(7,-1,junction=True)
    current.next=lambda d:[entry if d>=92 else current]
    other.next=lambda d:[other]
    c=controller(current,other,entry,other,direction)
    assert c.route_lane_recovery_direction() is None


@pytest.mark.parametrize('direction',['left','right'])
@pytest.mark.parametrize('legal',[False,True])
def test_imminent_junction_overrides_near_lane_only_when_legal(direction,legal):
    current=waypoint(1,-1)
    other=waypoint(1,-2)
    entry=waypoint(7,-1,junction=True)
    current.next=lambda d:[current]
    other.next=lambda d:[entry if d>=92 else other]
    c=controller(current,current,entry,other,direction,legal)
    assert c.route_lane_recovery_direction()==(direction if legal else None)


@pytest.mark.parametrize('z,yaw,section',[(2.,0.,0),(0.,180.,0),(0.,0.,1)])
def test_wrong_level_heading_or_section_does_not_prove_reachability(z,yaw,section):
    current=waypoint(1,-1)
    other=waypoint(1,-2)
    entry=waypoint(7,-1,junction=True)
    wrong=waypoint(7,-1,junction=True,z=z,yaw=yaw)
    wrong.section_id=section
    current.next=lambda d:[current]
    other.next=lambda d:[wrong]
    c=controller(current,current,entry,other,'right')
    assert c._junction_approach_lane(current)==(False,None)


def test_nonjunction_route_keeps_existing_local_alignment():
    current=waypoint(1,-1)
    other=waypoint(1,-2)
    current.next=lambda d:[current]
    other.next=lambda d:[other]
    c=controller(current,other,other,other,'right')
    assert c.route_lane_recovery_direction()=='right'
