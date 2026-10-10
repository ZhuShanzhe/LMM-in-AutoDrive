from types import SimpleNamespace as NS
import pytest
from benchmark.catalog import ConfigError
from benchmark.truth_capture import RouteProjector
from benchmark.role_journal import RoleJournal


def test_crossing_actor_projection_preserves_motion_bound_without_lane_heading():
    route = [dict(x=float(x), y=0., z=0., distance_m=float(x)) for x in range(51)]
    projector = RouteProjector(route)
    position = NS(x=10., y=1., z=0.)
    result = projector.project(position, 10., heading_deg=90., maximum_offset_m=2.,
        require_route_heading=False)
    assert result['route_s_m'] == 10. and result['route_error_m'] == 1.
    with pytest.raises(ConfigError):
        projector.project(position, 10., heading_deg=90., maximum_offset_m=2.)


def test_crossing_actor_cannot_jump_to_later_lap():
    route = [dict(x=float(x), y=0., z=0., distance_m=float(x)) for x in range(51)]
    projector = RouteProjector(route)
    result = projector.project(NS(x=40., y=1., z=0.), 10., maximum_offset_m=2.,
        require_route_heading=False)
    assert result['route_s_m'] <= 12.
    assert result['route_error_m'] >= 28.


@pytest.mark.parametrize('kind,expected', [('yield_pedestrian', False), ('pass_target', True)])
def test_role_journal_uses_actor_type_for_heading_constraint(kind, expected):
    journal = RoleJournal([dict(steps=[dict(kind=kind, target_role='target')])])
    calls = []

    class Projector:
        def project_actor(self, actor, hint, stamp, **kwargs):
            calls.append(kwargs)
            return dict(route_s_m=hint)

    journal._project_actor('target', object(), NS(timestamp=NS(elapsed_seconds=1.)), Projector(), 10.)
    assert calls[0]['require_route_heading'] is expected
    assert calls[0]['motion_state'] is journal.motion['target']
