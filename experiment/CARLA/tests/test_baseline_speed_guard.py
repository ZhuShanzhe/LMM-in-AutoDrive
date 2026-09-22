import pytest

from benchmark.baseline_speed_guard import guard_speed


def test_cap_preserves_original_command_and_route_target():
    intent=dict(action='keep_lane',target_speed_kmh=45,command_id='c1',target_location={'x':12})
    effective, evidence=guard_speed(intent,30)
    assert intent['target_speed_kmh']==45
    assert effective['target_speed_kmh']==29
    assert effective['target_location']==intent['target_location']
    assert evidence['requested_target_kmh']==45 and evidence['capped']


def test_cap_releases_when_reported_limit_increases():
    result, evidence=guard_speed(dict(action='keep_lane',target_speed_kmh=45),90)
    assert result['target_speed_kmh']==45 and not evidence['capped']


@pytest.mark.parametrize('limit',[None,0,-1,float('nan'),float('inf'),True])
def test_unknown_limit_stops_instead_of_assuming_unrestricted(limit):
    result, evidence=guard_speed(dict(action='keep_lane',target_speed_kmh=45),limit)
    assert result['action']=='stop' and result['target_speed_kmh']==0


def test_stop_is_not_overridden_by_guard():
    result,_=guard_speed(dict(action='stop',target_speed_kmh=0,emergency=True),90)
    assert result['action']=='stop' and result['emergency']
