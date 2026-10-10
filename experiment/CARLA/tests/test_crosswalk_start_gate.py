import math
from types import SimpleNamespace as NS
import pytest
from scenarios.crosswalk_start_gate import crossing_start_observation


def point(x,y,z=0.):return NS(x=x,y=y,z=z)


def vehicle(x,y,yaw=0.,z=0.,identity=1):
    angle=math.radians(yaw);c,s=math.cos(angle),math.sin(angle)
    corners=[point(x+c*dx-s*dy,y+s*dx+c*dy,z+dz)
        for dx in (-2.,2.) for dy in (-1.,1.) for dz in (0.,1.5)]
    transform=NS(rotation=NS(yaw=yaw))
    return NS(id=identity,get_transform=lambda:transform,
        bounding_box=NS(get_world_vertices=lambda t:corners))


def test_body_on_crossing_defers_start():
    result=crossing_start_observation(point(0,-5),point(0,5),[vehicle(0,0)])
    assert not result['clear'] and result['blocked_actor_ids']==[1] and not result['policy_input']


def test_rotated_body_detected():
    assert not crossing_start_observation(point(-5,0),point(5,0),[vehicle(0,0,90)])['clear']


def test_parallel_nearby_car_does_not_block():
    assert crossing_start_observation(point(0,-5),point(0,5),[vehicle(4,0)])['clear']


def test_other_level_is_not_crossing_traffic():
    assert crossing_start_observation(point(0,-5),point(0,5),[vehicle(0,0,z=10)])['clear']


def test_clear_path_is_not_a_completion_fact():
    result=crossing_start_observation(point(0,-5),point(0,5),[])
    assert result['clear'] and 'completed' not in result


@pytest.mark.parametrize('start,target,margin',[(point(0,0),point(0,0),.5),
    (point(float('nan'),0),point(0,3),.5),(point(0,0),point(0,3),-1.)])
def test_invalid_geometry_rejected(start,target,margin):
    with pytest.raises(ValueError):crossing_start_observation(start,target,[],margin_m=margin)
