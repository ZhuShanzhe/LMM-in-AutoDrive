from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError,load_catalog
from benchmark.event_fixture import route_crossing_fixture, bus_passenger_fixture
from benchmark.task_oracle import load_profile


def geometry():
    route=[dict(x=s,y=0,z=0,yaw=0,distance_m=s,road_id=1,section_id=0,lane_id=-1)
           for s in range(0,101,5)]
    waypoint=NS(road_id=1,section_id=0,lane_id=-1,is_junction=False,lane_width=3.5,
                transform=NS(rotation=NS(yaw=0)))
    return route,waypoint,NS(get_waypoint=lambda location:waypoint)


def test_unmarked_crossing_is_explicit_and_uses_road_width():
    route,_,world_map=geometry()
    result=route_crossing_fixture(world_map,route,50,NS)
    assert result['stop_line_route_s_m']==45
    assert result['geometry_source']=='configured_unmarked_crossing_on_route_lane'
    assert {p[1] for p in result['conflict_polygon_xy']}=={-1.75,1.75}


def test_map_lane_mismatch_cannot_silently_shift_crossing():
    route,waypoint,world_map=geometry()
    waypoint.lane_id=-2
    with pytest.raises(ConfigError,match='differs'):
        route_crossing_fixture(world_map,route,50,NS)


def test_junction_crossing_not_guessed():
    route,waypoint,world_map=geometry()
    waypoint.is_junction=True
    with pytest.raises(ConfigError,match='non-junction'):
        route_crossing_fixture(world_map,route,50,NS)


def test_scene3_worker_profile_bound():
    catalog=load_catalog('scene_3')
    profile=load_profile(catalog,catalog.select('scene3_worker_crossing')[0])
    assert profile['steps'][0]['require_stop'] is True


def test_bus_fixture_uses_explicit_roles_and_is_not_official_crosswalk():
    route,_,world_map=geometry()
    event=dict(kind='bus_stop',anchor_progress_m=50,passengers=[
        dict(role_name='p1',longitudinal_offset_m=-5),dict(role_name='p2',longitudinal_offset_m=8)])
    result=bus_passenger_fixture(world_map,route,event,{'p1','p2'},NS)
    assert result['geometry_source']=='configured_bus_stop_route_lane_area'
    assert result['half_length_m']==11
    assert result['stop_line_route_s_m']==37
    with pytest.raises(ConfigError,match='explicit unique'):
        bus_passenger_fixture(world_map,route,event,{'unknown'},NS)
