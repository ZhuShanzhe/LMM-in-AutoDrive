import pytest
from lightweight_vla_adapter.src.sequence_execution import SequenceExecutionPolicy


def call(policy,t,a=1.,risk='low',episode='a'):
    return policy.update(dict(action='keep_lane',target_speed_kmh=30.),
        dict(schema_version='longitudinal_sequence/1.0',dt_s=.1,speed_mps=[5.]*30,acceleration_mps2=[a]*30),
        timestamp_s=t,episode_id=episode,speed_kmh=18.,desired_speed_kmh=30.,risk=dict(risk_level=risk))


def test_regular_sign_changes_wait_one_second():
    p=SequenceExecutionPolicy()
    call(p,0,1.)
    for i in range(1,10):
        assert call(p,i/10,-1.)['action']=='accelerate'
        assert not p.diagnostics['operation_updated']
    assert call(p,1.,-1.)['action']=='decelerate'


def test_emergency_does_not_wait_for_interval():
    p=SequenceExecutionPolicy()
    call(p,0,2.)
    assert call(p,.1,-7.,'high')['action']=='emergency_brake'
    assert p.diagnostics['safety_bypass']


def test_gap_or_episode_resets_held_command():
    p=SequenceExecutionPolicy()
    call(p,0,1.)
    assert call(p,.5,-1.)['action']=='keep_lane'
    assert p.command['target_acceleration_mps2']==0.
    assert p.diagnostics['observation_gap_reset']
    assert call(p,.6,1.,episode='b')['action']=='accelerate'


def test_recovery_does_not_bypass_dwell_but_emergency_can():
    p=SequenceExecutionPolicy()
    call(p,0,1.)
    for i in (5,6,7,8,9):
        call(p,i/10,-1.)
        assert not p.diagnostics['operation_updated']
    assert call(p,1.,-1.)['action']=='decelerate'
    assert call(p,1.5,-7.,risk='high')['action']=='emergency_brake'


def test_invalid_sequence_is_rejected():
    with pytest.raises(ValueError): call(SequenceExecutionPolicy(),0,float('nan'))


def test_reference_keeps_plan_origin_when_vehicle_lags():
    p=SequenceExecutionPolicy()
    first=call(p,0,1.)
    for i in range(1,6): result=call(p,i/10,1.)
    assert first['target_speed_kmh']==pytest.approx(5.1*3.6)
    assert result['target_speed_kmh']==pytest.approx(5.6*3.6)
    assert p.command['target_acceleration_mps2']==pytest.approx(1.)
