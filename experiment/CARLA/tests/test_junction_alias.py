import pytest

from benchmark.junction_alias import coincident_junction_alias_matches


def fixture():
    return dict(in_junction=True, lane_key='2057:0:1', route_error_m=.05,
        yaw_deg=-.6, route_yaw_deg=-.2), [dict(lane_keys=['2034:0:4'],
        coincident_junction_lane_keys=['2057:0:1'],
        junction_alias_source='static_topology_coincident_within_0.05m_1deg')]


def test_verified_geometric_overlap_is_not_a_lane_change():
    ego, segments = fixture()
    assert coincident_junction_alias_matches(ego, segments, .35, 5.)


@pytest.mark.parametrize('field,value', [
    ('in_junction', False), ('lane_key', 'wrong-road'), ('route_error_m', .351),
    ('route_error_m', -1.), ('route_error_m', float('nan')), ('route_error_m', True),
    ('yaw_deg', 5.), ('yaw_deg', None), ('route_yaw_deg', None)])
def test_wrong_lane_geometry_or_missing_observation_is_rejected(field, value):
    ego, segments = fixture()
    ego[field] = value
    assert not coincident_junction_alias_matches(ego, segments, .35, 5.)


@pytest.mark.parametrize('field', ['coincident_junction_lane_keys', 'junction_alias_source'])
def test_geometric_similarity_alone_is_not_a_static_alias(field):
    ego, segments = fixture()
    del segments[0][field]
    assert not coincident_junction_alias_matches(ego, segments, .35, 5.)


@pytest.mark.parametrize('position,expected', [(4947.95, True), (4948.3, False), (None, False)])
def test_turning_connector_alias_is_limited_to_its_actual_shared_prefix(position, expected):
    ego, segments = fixture()
    ego['route_s_m'] = position
    segments[0].pop('coincident_junction_lane_keys')
    segments[0]['coincident_junction_lane_ranges'] = [dict(lane_key='2057:0:1',
        start_m=4947.85, end_m=4948.05)]
    assert coincident_junction_alias_matches(ego, segments, .35, 5.) is expected
