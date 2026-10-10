import sys
from types import SimpleNamespace

import pytest
from scenarios.crosswalk_exit import sidewalk_exit_targets


class Point:
    def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z


def fake_carla(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',SimpleNamespace(Location=Point,
        LaneType=SimpleNamespace(Sidewalk='sidewalk',Driving='driving')))


class Road:
    def get_waypoint(self,p,project_to_road=True,lane_type=None):
        if lane_type=='sidewalk':
            return SimpleNamespace(is_junction=p.x>=-5.,transform=SimpleNamespace(location=Point(p.x,4.,0.)))
        return object() if p.x>-3. and p.y<2. else None


def test_exit_uses_nonjunction_sidewalk_and_keeps_physical_waypoints(monkeypatch):
    fake_carla(monkeypatch)
    points=sidewalk_exit_targets(Road(),Point(),Point(1.,0.))
    assert points[0].x==-8.
    assert points[1].x==-8. and points[1].y==4.


def test_rotated_road_selects_equivalent_geometry(monkeypatch):
    fake_carla(monkeypatch)
    class Rotated:
        def get_waypoint(self,p,project_to_road=True,lane_type=None):
            result=Road().get_waypoint(Point(p.y,-p.x,p.z),project_to_road,lane_type)
            if lane_type=='sidewalk':
                at=result.transform.location
                result.transform.location=Point(-at.y,at.x,at.z)
            return result
    points=sidewalk_exit_targets(Rotated(),Point(),Point(0.,1.))
    assert points[-1].x==-4. and points[-1].y==-8.


def test_missing_safe_exit_is_rejected(monkeypatch):
    fake_carla(monkeypatch)
    class NoExit:
        def get_waypoint(self,*args,**kwargs):return None
    with pytest.raises(ValueError,match='No lane-free'):
        sidewalk_exit_targets(NoExit(),Point(),Point(1.,0.))


@pytest.mark.parametrize('tangent',[Point(),Point(float('nan'),0.)])
def test_invalid_tangent_rejected(monkeypatch,tangent):
    fake_carla(monkeypatch)
    with pytest.raises(ValueError):sidewalk_exit_targets(Road(),Point(),tangent)
