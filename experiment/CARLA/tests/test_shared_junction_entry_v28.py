import pytest
from benchmark.junction_alias import coincident_junction_alias_matches


def segment():
    return [dict(start_m=4945., end_m=4950., lane_keys=['2034:0:4'],
        shared_entry_alias_source='static_topology_shared_entry_0.2m_10deg',
        shared_entry_junction_lane_ranges=[dict(lane_key='2057:0:1',
            start_m=4947.85, end_m=4949.35)])]


def observed(**changes):
    value=dict(in_junction=True, lane_key='2057:0:1', route_s_m=4949.17,
        route_error_m=.0677, yaw_deg=-.1147, route_yaw_deg=-.2406)
    value.update(changes)
    return value


def test_nearest_waypoint_tangent_does_not_replace_actual_route_heading():
    assert coincident_junction_alias_matches(observed(), segment(), .35, 5.)


@pytest.mark.parametrize('changes', [dict(yaw_deg=6.), dict(route_error_m=.351),
    dict(route_s_m=4949.4), dict(in_junction=False), dict(lane_key='other'),
    dict(route_yaw_deg=float('nan'))])
def test_actual_branch_departure_and_unproven_prefix_remain_rejected(changes):
    assert not coincident_junction_alias_matches(observed(**changes), segment(), .35, 5.)
