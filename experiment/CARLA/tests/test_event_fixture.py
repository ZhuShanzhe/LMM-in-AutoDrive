from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.event_fixture import bind_task_roles, closed_crosswalks, crosswalk_fixture


def actor(identity=1,role='ped',type_id='walker.pedestrian.0001'):
    return NS(id=identity,attributes={'role_name':role},type_id=type_id,is_alive=True,
              bounding_box=NS(extent=NS(x=.3,y=.3),location=NS(x=0,y=0),rotation=NS(yaw=0)))


SPEC={'steps':[{'kind':'yield_pedestrian','target_role':'ped'},
               {'kind':'overtake','target_role':'lead'}]}


def test_role_binding_requires_actual_matching_actors():
    result=bind_task_roles(SPEC,[actor(),actor(2,'lead','vehicle.audi.tt')],{'ped':20,'lead':40})
    assert result['ped'].actor_id==1 and result['lead'].initial_route_s_m==40


@pytest.mark.parametrize('actors',[
    [],[actor(),actor()],[actor(type_id='vehicle.audi.tt')]])
def test_missing_duplicate_or_incompatible_role_is_invalid(actors):
    with pytest.raises(ConfigError):
        bind_task_roles({'steps':SPEC['steps'][:1]},actors,{'ped':20})


def points(y=0):
    return [NS(x=x,y=v+y) for x,v in [(18,-6),(22,-6),(22,6),(18,6),(18,-6)]]


ROUTE=[dict(x=x,y=0,z=0,yaw=0,distance_m=x) for x in range(0,101,5)]


def test_crosswalk_uses_official_geometry_and_bumper_stop_line():
    result=crosswalk_fixture(NS(get_crosswalks=points),ROUTE,0,20)
    assert result['stop_line_route_s_m']==16
    assert len(result['conflict_polygon_xy'])==4


def test_crosswalk_on_other_road_is_not_usable():
    with pytest.raises(ConfigError,match='does not intersect'):
        crosswalk_fixture(NS(get_crosswalks=lambda:points(40)),ROUTE,0,20)


def test_unclosed_polygon_and_missing_index_rejected():
    with pytest.raises(ConfigError,match='unterminated'):
        closed_crosswalks(points()[:-1])
    with pytest.raises(ConfigError,match='unavailable'):
        crosswalk_fixture(NS(get_crosswalks=points),ROUTE,2,20)
