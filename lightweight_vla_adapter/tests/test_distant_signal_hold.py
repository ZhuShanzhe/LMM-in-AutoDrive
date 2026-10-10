import pytest

from lightweight_vla_adapter.src.traffic_control_contract import TrafficControlContract


@pytest.mark.parametrize('state', ['RED', 'UNKNOWN', 'YELLOW'])
@pytest.mark.parametrize('distance', [35., 54., 70.])
def test_distant_unconstraining_signal_does_not_create_zero_distance_hold(state, distance):
    contract = TrafficControlContract()
    decision = dict(action='keep_lane', target_speed_kmh=10.)
    observed = dict(applicable=True, signal_id='signal', state=state, confidence=1.,
                    timestamp_s=0., stop_distance_m=distance)
    result, info = contract.apply(decision, observed, timestamp_s=0., ego_speed_mps=1.)
    assert result == decision and not info['stop_hold_latched']
    result, info = contract.apply(decision, dict(applicable=False), timestamp_s=.1, ego_speed_mps=1.)
    assert result == decision and not info['active']


def test_distant_signal_that_requires_braking_still_latches():
    contract = TrafficControlContract()
    decision = dict(action='keep_lane', target_speed_kmh=60.)
    observed = dict(applicable=True, signal_id='signal', state='UNKNOWN', confidence=0.,
                    timestamp_s=0., stop_distance_m=20.)
    _, info = contract.apply(decision, observed, timestamp_s=0., ego_speed_mps=10.)
    assert info['stop_hold_latched']
    result, _ = contract.apply(decision, dict(applicable=False), timestamp_s=.1, ego_speed_mps=5.)
    assert result['action'] == 'stop'


def test_explicit_stop_near_red_does_not_forget_signal():
    contract = TrafficControlContract()
    decision = dict(action='stop', target_speed_kmh=0.)
    observed = dict(applicable=True, signal_id='signal', state='RED', confidence=1.,
                    timestamp_s=0., stop_distance_m=4.)
    contract.apply(decision, observed, timestamp_s=0., ego_speed_mps=0.)
    result, _ = contract.apply(dict(action='accelerate', target_speed_kmh=20.),
                               dict(applicable=False), timestamp_s=.1, ego_speed_mps=0.)
    assert result['action'] == 'stop'
