from types import SimpleNamespace as NS
import pytest
from benchmark.truth_capture import RouteProjector
from benchmark.catalog import ConfigError


def fixture():
    route=[dict(x=x,y=y,z=0.,distance_m=s) for x,y,s in
        [(-20,0,0),(20,0,40),(20,20,60),(-20,20,100),(-20,-3.5,123.5),(20,-3.5,163.5)]]
    return RouteProjector(route)


def test_measured_motion_does_not_jump_to_a_closer_later_lap():
    projector=fixture()
    result=projector.project(NS(x=0.,y=-3.3,z=0.),20.,window=180.,heading_deg=0.,maximum_offset_m=2.)
    assert result['route_s_m']==20.
    assert result['route_error_m']==pytest.approx(3.3)


def test_reachability_cannot_erase_lateral_offset_or_invent_a_branch():
    projector=fixture()
    with pytest.raises(ConfigError,match='without measured'):
        projector.project(NS(x=0.,y=-3.3,z=0.),70.,window=180.,heading_deg=0.,maximum_offset_m=2.)
