from types import SimpleNamespace
import pytest
from control.generic_route_pid import GenericRoutePID


def controller(limit):
    value=object.__new__(GenericRoutePID)
    value._default_speed_kmh=45.
    value._requested_cruise_speed_kmh=45.
    value.ego=SimpleNamespace(get_speed_limit=lambda:limit)
    return value


def test_requested_speed_can_exceed_default_but_not_road_limit():
    value=controller(60.)
    assert value.set_cruise_target_kmh(50.)==50.
    value.set_high_level_decision(dict(action='accelerate',target_speed_kmh=50.))
    assert value._target_speed==50.
    value.ego.get_speed_limit=lambda:30.
    value.set_high_level_decision(dict(action='accelerate',target_speed_kmh=50.))
    assert value._target_speed==30.
    value.ego.get_speed_limit=lambda:60.
    assert value.set_cruise_target_kmh(None)==50.


def test_cruise_goal_does_not_replace_model_stop_or_lower_speed():
    value=controller(60.)
    value.set_cruise_target_kmh(50.)
    value.set_high_level_decision(dict(action='decelerate',target_speed_kmh=12.))
    assert value._target_speed==12.
    value.set_high_level_decision(dict(action='stop',target_speed_kmh=50.))
    assert value._target_speed==0.


@pytest.mark.parametrize('limit',[0.,float('nan'),float('inf')])
def test_unknown_road_limit_retains_conservative_default(limit):
    assert controller(limit).set_cruise_target_kmh(80.)==45.


@pytest.mark.parametrize('target',[-1,121,float('nan'),True])
def test_invalid_goal_rejected(target):
    with pytest.raises(ValueError):controller(60.).set_cruise_target_kmh(target)
