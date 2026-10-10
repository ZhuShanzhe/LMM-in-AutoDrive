import pytest
from benchmark.junction_alias import coincident_junction_alias_matches


def observation(**changes):
    row=dict(in_junction=True,lane_key='branch',route_s_m=10.4,
             route_error_m=.02,yaw_deg=0,route_yaw_deg=0)
    row.update(changes)
    return row


def segments():
    return [dict(start_m=10,end_m=15,lane_keys=['straight'],
        shared_entry_alias_source='static_topology_shared_entry_0.2m_5deg',
        shared_entry_junction_lane_ranges=[dict(lane_key='branch',start_m=10.1,end_m=10.9)])]


def test_nearest_connector_tie_is_allowed_only_on_proven_shared_entry():
    assert coincident_junction_alias_matches(observation(),segments(),.35,5)


@pytest.mark.parametrize('changes',[dict(in_junction=False),dict(lane_key='other'),
    dict(route_s_m=11),dict(route_error_m=.36),dict(yaw_deg=6),
    dict(route_error_m=float('nan')),dict(route_s_m=True)])
def test_unproven_or_departed_branch_is_rejected(changes):
    assert not coincident_junction_alias_matches(observation(**changes),segments(),.35,5)


def test_unrecognized_static_provenance_is_rejected():
    rows=segments()
    rows[0]['shared_entry_alias_source']='model_suggestion'
    assert not coincident_junction_alias_matches(observation(),rows,.35,5)
