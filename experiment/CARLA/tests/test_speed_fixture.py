from types import SimpleNamespace as NS
import pytest
from benchmark.speed_fixture import slice_speed_fixture
from benchmark.catalog import ConfigError


def route():
    return [dict(distance_m=s,x=s,y=0,z=0,road_id=1,lane_id=-1) for s in range(0,501,20)]


def test_slice_preserves_source_and_trigger_offset():
    source=route()
    spec=dict(activate_m=300,end_route_s_m=400)
    world=NS(get_waypoint=lambda p:NS(is_junction=p.x==180,road_id=1,lane_id=-1))
    local,profile,offset=slice_speed_fixture(source,spec,world,NS)
    assert offset==200
    assert local[0]['distance_m']==0 and local[0]['x']==200
    assert profile['activate_m']==100 and profile['end_route_s_m']==200
    assert source[10]['distance_m']==200 and spec['activate_m']==300


def test_no_valid_approach_is_not_silently_relocated():
    world=NS(get_waypoint=lambda p:NS(is_junction=True,road_id=1,lane_id=-1))
    with pytest.raises(ConfigError,match='preparation entry'):
        slice_speed_fixture(route(),dict(activate_m=300),world,NS)


def test_prior_evidence_cannot_be_fabricated():
    with pytest.raises(ConfigError,match='prior task'):
        slice_speed_fixture(route(),dict(activate_m=300,requires_task_success=['earlier']),None,NS)
