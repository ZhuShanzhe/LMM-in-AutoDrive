import pytest

from benchmark.junction_alias import coincident_junction_alias_matches


def observation(**changes):
    return dict(in_junction=True, junction_id=2014, lane_key='2039:0:-1',
        route_s_m=4961.938, route_error_m=.056, yaw_deg=-.299,
        route_yaw_deg=-.241, **changes)


def segment(**changes):
    result = dict(start_m=4960., end_m=4965., lane_keys=['2034:0:4'],
        verified_junction_ids=[2014],
        junction_geometry_source='static_topology_verified_route_projection')
    result.update(changes)
    return result


def check(ego, corridor=None):
    return coincident_junction_alias_matches(ego, corridor or [segment()], .35, 5.)


def test_crossing_connector_id_does_not_override_measured_route_pose():
    assert check(observation())


@pytest.mark.parametrize('field,value', [
    ('route_error_m', .351), ('yaw_deg', 6.), ('in_junction', False),
    ('junction_id', 999), ('route_s_m', 4965.1),
    ('route_error_m', float('nan')), ('route_yaw_deg', float('inf')),
])
def test_actual_departure_or_unverified_geometry_is_not_accepted(field, value):
    ego = observation()
    ego[field] = value
    assert not check(ego)


def test_requires_static_topology_proof():
    assert not check(observation(), [segment(junction_geometry_source='model')])
    assert not check(observation(), [segment(verified_junction_ids=[])])


def test_parallel_lane_and_turning_branch_still_fail():
    ego = observation()
    ego['route_error_m'] = 3.5
    assert not check(ego)
    ego['route_error_m'] = .05
    ego['yaw_deg'] = 90.
    assert not check(ego)
