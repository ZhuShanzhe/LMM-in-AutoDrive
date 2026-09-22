from continuous.route_manager import RouteManager
from types import SimpleNamespace as NS
import pytest


def test_target_yaw_follows_polyline_tangent_not_jumping_waypoint_rotation():
    manager = RouteManager.__new__(RouteManager)
    manager.route = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "distance_m": 0.0},
        {"x": 10.0, "y": 2.5, "z": 0.0, "yaw": 65.0, "distance_m": 10.5},
    ]
    manager.progress_m = 0.0
    manager.current_index = 0
    target = manager.target_point(10.0)
    assert 13.0 < target["yaw"] < 15.0


def straight_manager(end=10000):
    class Waypoint:
        road_id=1
        section_id=0
        lane_id=-1
        is_junction=False
        def __init__(self,x):
            self.transform=NS(location=NS(x=x,y=0,z=0),rotation=NS(yaw=0))
        def next(self,distance):
            x=self.transform.location.x+distance
            return [Waypoint(x)] if x<=end else []
    world_map=NS(get_waypoint=lambda location,**kwargs:Waypoint(location.x))
    return RouteManager(NS(get_map=lambda:world_map))


@pytest.mark.parametrize('length',[5000,12,3])
def test_route_includes_exact_terminal_sample(length):
    manager=straight_manager()
    route=manager.build_route(NS(x=0),length_m=length,step_m=5)
    assert route[-1]['distance_m']==length
    assert route[-1]['x']==length
    assert manager.route_length_m==length
    assert all(b['distance_m']>a['distance_m'] for a,b in zip(route,route[1:]))


def test_dead_end_does_not_fabricate_route_length():
    manager=straight_manager(end=10)
    route=manager.build_route(NS(x=0),length_m=20,step_m=5)
    assert route[-1]['distance_m']==10


@pytest.mark.parametrize('progress,xyz,expected', [
    (82, (82,0,0), False),
    (97, (97,0,0), True),
    (100, (100,4,0), False),
    (100, (100,0,4), False),
    (90, (100,0,0), False),
    (100, (100,0,0), True),
    (100, (float('nan'),0,0), False),
    (float('nan'), (100,0,0), False),
])
def test_route_end_requires_progress_and_spatial_arrival(progress, xyz, expected):
    manager=straight_manager()
    manager.build_route(NS(x=0), length_m=100)
    manager.progress_m=progress
    assert manager.is_finished(3, NS(x=xyz[0],y=xyz[1],z=xyz[2])) is expected


def test_empty_route_is_not_complete():
    assert not straight_manager().is_finished(3)


@pytest.mark.parametrize('tolerance', [-1, float('nan'), float('inf')])
def test_invalid_terminal_tolerance(tolerance):
    with pytest.raises(ValueError):
        straight_manager().is_finished(tolerance)


@pytest.mark.parametrize('length,step',[(0,5),(float('inf'),5),(10,float('nan'))])
def test_route_rejects_nonfinite_or_empty_bounds(length,step):
    with pytest.raises(ValueError): straight_manager().build_route(NS(x=0),length_m=length,step_m=step)


def turn_point(road,yaw,junction=False):
    return NS(road_id=road,lane_id=-1,lane_type='Driving',is_junction=junction,
              transform=NS(rotation=NS(yaw=yaw)),next=lambda step:[])


@pytest.mark.parametrize('yaw,action', [(-90,'turn_left'),(90,'turn_right'),(180,'u_turn')])
def test_turn_requires_junction_and_correct_exit(yaw,action):
    manager=RouteManager.__new__(RouteManager)
    entry=turn_point(1,0)
    connector=turn_point(100,0,True)
    exit_point=turn_point(2,yaw)
    connector.next=lambda step:[exit_point]
    assert manager._choose_turn(entry,[connector],action) is connector
    bend=turn_point(1,yaw)
    bend.next=lambda step:[exit_point]
    assert manager._choose_turn(entry,[bend],action) is None


@pytest.mark.parametrize('road,yaw',[(1,-90),(2,-20),(2,90)])
def test_left_turn_rejects_same_road_shallow_and_wrong_direction(road,yaw):
    manager=RouteManager.__new__(RouteManager)
    connector=turn_point(100,0,True)
    connector.next=lambda step:[turn_point(road,yaw)]
    assert manager._choose_turn(turn_point(1,0),[connector],'turn_left') is None


def test_turn_rejects_unfinished_junction_and_mid_junction_entry():
    manager=RouteManager.__new__(RouteManager)
    connector=turn_point(100,-90,True)
    connector.next=lambda step:[connector]
    assert manager._choose_turn(turn_point(1,0),[connector],'turn_left') is None
    assert manager._choose_turn(turn_point(1,0,True),[connector],'turn_left') is None
