import pytest

from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy, cautious_permission, apply_behavior_decision, stopping_clearance
from lightweight_vla_adapter.src.longitudinal_contract import enforce_longitudinal_contract
from lightweight_vla_adapter.tests.test_short_horizon_crawl import sample


@pytest.mark.parametrize('distance', [20., 25., 40., 55.])
def test_clear_road_is_bounded_by_actual_sensor_horizon(distance):
    r = sample(RiskBehaviorPolicy(), 0, level='low', speed=0., distance=distance)
    p = cautious_permission(r)
    assert p and stopping_clearance(p['max_speed_kmh']) <= distance
    d = dict(action='keep_lane', target_speed_kmh=60., frame_id=r['frame_id'],
             decision_status='READY', source_step_action='SET_SPEED')
    out, _ = apply_behavior_decision(d, r)
    out, _ = enforce_longitudinal_contract(out, r, 0.)
    assert out['target_speed_kmh'] == p['max_speed_kmh'] < 60.


@pytest.mark.parametrize('level', ['low', 'medium'])
def test_confirmed_horizon_recovers_above_ten_without_assuming_thirty(level):
    policy = RiskBehaviorPolicy()
    for i in range(16):
        r = sample(policy, i, level=level, speed=26. if i == 0 else 8., distance=12.)
    p = cautious_permission(r)
    assert p and 10. < p['max_speed_kmh'] < 30.
    assert stopping_clearance(p['max_speed_kmh']) <= 12.
    assert r['behavior_policy']['response_tier'] == 'LIMITED_HORIZON_BOUNDED'
    r = sample(policy, 16, level=level, speed=8., distance=12., fault='obstacle')
    assert cautious_permission(r) is None


def test_high_visual_advisory_still_capped_at_ten():
    policy = RiskBehaviorPolicy()
    for i in range(16):
        r = sample(policy, i, level='high', speed=8., distance=40.)
    assert cautious_permission(r)['max_speed_kmh'] == 10.


@pytest.mark.parametrize('fault', ['stale', 'invalid', 'no_support', 'ttc', 'near_radar', 'crossing'])
def test_clear_risk_label_cannot_override_physical_or_freshness_failure(fault):
    r = sample(RiskBehaviorPolicy(), 0, level='low', speed=0., distance=40., fault=fault)
    assert cautious_permission(r) is None


def test_expired_clear_bound_fails_closed():
    r = sample(RiskBehaviorPolicy(), 0, level='low', speed=0., distance=40.)
    r['risk_evidence']['decision_timestamp_s'] += .5
    d = dict(action='accelerate', target_speed_kmh=60., frame_id=r['frame_id'])
    out, _ = apply_behavior_decision(d, r)
    assert out['action'] == 'stop' and out['target_speed_kmh'] == 0.


def test_cap_recovery_uses_conservative_history_but_new_tightening_is_immediate():
    policy = RiskBehaviorPolicy()
    first = sample(policy, 0, level='low', speed=0., distance=20.)
    low = cautious_permission(first)['max_speed_kmh']
    for i in range(1, 5):
        r = sample(policy, i, level='low', speed=0., distance=40.)
        assert cautious_permission(r)['max_speed_kmh'] == low
    r = sample(policy, 6, level='low', speed=0., distance=40.)
    assert cautious_permission(r)['max_speed_kmh'] > low
    r = sample(policy, 7, level='low', speed=0., distance=18.)
    assert cautious_permission(r)['max_speed_kmh'] < low
    r = sample(policy, 8, level='low', speed=0., distance=40., fault='near_radar')
    assert cautious_permission(r) is None


def test_repeated_capture_does_not_refresh_cap_history():
    policy = RiskBehaviorPolicy()
    sample(policy, 0, level='low', speed=0., distance=20.)
    for _ in range(6):
        sample(policy, 0, level='low', speed=0., distance=40.)
    assert len(policy.cap_history) == 1
    assert policy.cap_history[0][0] == 1.
