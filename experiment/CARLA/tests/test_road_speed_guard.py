from types import SimpleNamespace

import pytest

from control.road_speed_guard import apply_road_speed_guard


def apply(speed, acceleration=0., limit=30., brake=0.):
    control = SimpleNamespace(throttle=.5, brake=brake, steer=.3)
    evidence = apply_road_speed_guard(control, speed_kmh=speed, speed_limit_kmh=limit,
        acceleration_mps2=acceleration, dt_s=.05)
    return control, evidence


def test_accelerating_approach_releases_torque_before_overshoot():
    control, evidence = apply(29., 2.)
    assert control.throttle == 0.
    assert control.brake >= .1
    assert control.steer == .3
    assert evidence['predicted_speed_kmh'] > 30.


def test_distant_goal_does_not_clip_acceleration():
    control, evidence = apply(20., 2.)
    assert control.throttle == .5 and control.brake == 0.
    assert not evidence['changed']


def test_braking_evidence_does_not_predict_positive_acceleration():
    control, evidence = apply(29.9, -2.)
    assert control.throttle == 0. and control.brake == 0.
    assert evidence['predicted_speed_kmh'] == 29.9


def test_emergency_brake_is_never_reduced():
    control, _ = apply(31., 2., brake=1.)
    assert control.throttle == 0. and control.brake == 1.


@pytest.mark.parametrize('limit', [0., -1., float('nan'), float('inf'), True])
def test_missing_limit_does_not_invent_a_limit(limit):
    control, evidence = apply(20., limit=limit)
    assert control.throttle == .5 and control.brake == 0.
    assert evidence['status'] == 'UNAVAILABLE'


def test_any_finite_limit_uses_the_same_contract():
    control, _ = apply(59., 2., limit=60.)
    assert control.throttle == 0. and control.brake >= .1
