from lightweight_vla_adapter.src.traffic_control_contract import TrafficControlContract
import pytest


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


def retirement(signal='signal-a',t=.1):
    return dict(signal_id=signal,source='static_hdmap_stop_line',observed_at_s=t,
        sensor_frame=10,remaining_m=-2.,lateral_offset_m=.1,heading_alignment=1.)


def test_verified_passed_line_retires_only_the_matching_blocked_signal():
    contract=TrafficControlContract();decision=dict(action='keep_lane',target_speed_kmh=20.)
    contract.apply(decision,observation(),timestamp_s=0.,ego_speed_mps=0.)
    result,info=contract.apply(decision,dict(applicable=False,retired_signals=[retirement()]),timestamp_s=.1,ego_speed_mps=0.)
    assert result==decision and not info['active'] and info['status']=='passed_stop_line_release'
    assert contract.last_blocked is None


def test_retirement_does_not_release_the_next_red_light():
    contract=TrafficControlContract();decision=dict(action='keep_lane',target_speed_kmh=20.)
    contract.apply(decision,observation(),timestamp_s=0.,ego_speed_mps=0.)
    current=observation(t=.1);current['signal_id']='signal-b';current['retired_signals']=[retirement()]
    result,info=contract.apply(decision,current,timestamp_s=.1,ego_speed_mps=0.)
    assert result['action']=='stop' and info['active'] and contract.last_blocked=='signal-b'


def test_stale_wrong_lane_or_unpassed_proof_cannot_clear_a_red_hold():
    for changes in [dict(signal_id='b'),dict(remaining_m=1.),dict(observed_at_s=-1.),
                    dict(lateral_offset_m=4.),dict(heading_alignment=-1.),
                    dict(source='timeout'),dict(remaining_m=float('nan'))]:
        contract=TrafficControlContract();decision=dict(action='keep_lane',target_speed_kmh=20.)
        contract.apply(decision,observation(),timestamp_s=0.,ego_speed_mps=0.)
        proof=retirement();proof.update(changes)
        result,info=contract.apply(decision,dict(applicable=False,retired_signals=[proof]),timestamp_s=.1,ego_speed_mps=0.)
        assert result['action']=='stop' and info['active']


@pytest.mark.parametrize('state',['RED','YELLOW','GREEN'])
def test_stopped_ego_can_approach_fresh_light_within_braking_envelope(state):
    decision=dict(action='keep_lane',target_speed_kmh=30.,allow_positive_acceleration=True,
        target_acceleration_mps2=2.,longitudinal_sequence_schema='test',sequence_valid_until_s=1.)
    result,info=TrafficControlContract().apply(decision,observation(state,distance=15.8),
        timestamp_s=0.,ego_speed_mps=0.)
    assert 0.<result['target_speed_kmh']<30.
    assert result['allow_positive_acceleration'] and info['approach_acceleration_authorized']
    assert not any(k in result for k in ('target_acceleration_mps2','longitudinal_sequence_schema','sequence_valid_until_s'))
    assert info['active'] and info['stop_hold_latched']


@pytest.mark.parametrize('reason',['upstream','emergency','stale','unknown','stop_line'])
def test_approach_does_not_override_other_acceleration_prohibitions(reason):
    decision=dict(action='keep_lane',target_speed_kmh=30.,allow_positive_acceleration=True)
    obs=observation(distance=15.8)
    now=0.
    if reason=='upstream':decision['allow_positive_acceleration']=False
    elif reason=='emergency':decision.update(action='emergency_brake',emergency=True)
    elif reason=='stale':now=1.
    elif reason=='unknown':obs['state']='UNKNOWN'
    elif reason=='stop_line':obs['stop_distance_m']=5.
    result,info=TrafficControlContract().apply(decision,obs,timestamp_s=now,ego_speed_mps=0.)
    assert not result['allow_positive_acceleration']
    assert not info['approach_acceleration_authorized']
    if reason=='stop_line':assert result['action']=='stop' and result['target_speed_kmh']==0.


def test_allowed_approach_still_latches_stop_on_observation_loss():
    contract=TrafficControlContract();decision=dict(action='keep_lane',target_speed_kmh=30.)
    first,info=contract.apply(decision,observation(distance=15.8),timestamp_s=0.,ego_speed_mps=0.)
    assert first['allow_positive_acceleration']
    lost,info=contract.apply(decision,None,timestamp_s=.1,ego_speed_mps=0.)
    assert lost['action']=='stop' and lost['target_speed_kmh']==0.
    assert not lost['allow_positive_acceleration']
