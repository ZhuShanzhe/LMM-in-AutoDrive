import math
from types import SimpleNamespace as NS
import carla
import pytest
from scenarios.complex.town05_scene2 import route_crosswalk_index


def rectangle(x,y,z,yaw):
    angle=math.radians(yaw)
    def point(a,b):
        return carla.Location(x=x+a*math.cos(angle)-b*math.sin(angle),
            y=y+a*math.sin(angle)+b*math.cos(angle),z=z)
    points=[point(-1,-8),point(1,-8),point(1,8),point(-1,8)]
    return points+[points[0]]


@pytest.mark.parametrize('yaw',[0.,35.,90.,180.,-90.])
def test_old_crosswalk_index_is_rebound_to_current_route(yaw):
    points=rectangle(250.,250.,0.,yaw)+rectangle(10.,10.,0.,yaw)
    road_map=NS(get_crosswalks=lambda:points)
    waypoint=NS(transform=NS(location=carla.Location(x=10.,y=10.,z=0.),rotation=NS(yaw=yaw)))
    assert route_crosswalk_index(road_map,waypoint,0)==1


def test_other_road_level_is_not_used():
    road_map=NS(get_crosswalks=lambda:rectangle(10.,10.,20.,0.))
    waypoint=NS(transform=NS(location=carla.Location(x=10.,y=10.,z=0.),rotation=NS(yaw=0.)))
    with pytest.raises(ValueError,match='same-level'):
        route_crosswalk_index(road_map,waypoint,0)


def test_parallel_crosswalk_is_not_a_route_crossing():
    road_map=NS(get_crosswalks=lambda:rectangle(10.,10.,0.,90.))
    waypoint=NS(transform=NS(location=carla.Location(x=10.,y=10.,z=0.),rotation=NS(yaw=0.)))
    with pytest.raises(ValueError,match='transverse'):
        route_crosswalk_index(road_map,waypoint,0)
