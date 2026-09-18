from lightweight_vla_adapter.src.traffic_control_contract import TrafficControlContract


def test_disappearing_unreleased_signal_does_not_allow_acceleration():
    contract=TrafficControlContract()
    decision=dict(action='accelerate',target_speed_kmh=20.)
    contract.apply(decision,dict(applicable=True,signal_id='a',state='RED',confidence=1.,stop_distance_m=5.,timestamp_s=0.),timestamp_s=0.,ego_speed_mps=0.)
    result,info=contract.apply(decision,dict(applicable=False),timestamp_s=.1,ego_speed_mps=0.)
    assert result['action']=='stop' and info['active']


def observation(state='RED',distance=4.,t=0.):
    return dict(applicable=True,signal_id='signal-a',state=state,confidence=.95,
        stop_distance_m=distance,timestamp_s=t)


def test_red_overrides_liveness_and_positive_sequence():
    contract=TrafficControlContract()
    result,info=contract.apply(dict(action='accelerate',target_speed_kmh=30.,target_acceleration_mps2=2.),
        observation(),timestamp_s=0.,ego_speed_mps=3.)
    assert result['action']=='stop' and result['target_speed_kmh']==0. and info['active']
    assert 'target_acceleration_mps2' not in result


def test_only_fresh_sustained_green_releases():
    contract=TrafficControlContract();decision=dict(action='keep_lane',target_speed_kmh=20.)
    for t in [0.,.1,.2]:
        result,info=contract.apply(decision,observation('GREEN',t=t),timestamp_s=t,ego_speed_mps=0.)
        assert info['active']
    result,info=contract.apply(decision,observation('GREEN',t=.4),timestamp_s=.4,ego_speed_mps=0.)
    assert not info['active'] and result==decision
    result,info=contract.apply(decision,observation('GREEN',t=.4),timestamp_s=1.,ego_speed_mps=0.)
    assert info['active']


def test_unavailable_observation_not_misreported_as_green():
    _,info=TrafficControlContract().apply(dict(action='keep_lane',target_speed_kmh=20.),None,timestamp_s=0.,ego_speed_mps=0.)
    assert info['status']=='unavailable'
