import pytest
from lightweight_vla_adapter.src.sequence_tracker import SequenceLongitudinalTracker
from lightweight_vla_adapter.src.sequence_execution import SequenceExecutionPolicy


def test_compensation_can_exceed_old_insufficient_limit():
    tracker=SequenceLongitudinalTracker()
    for _ in range(300):tracker.step(10.,0.,9.,.05)
    assert 2.<tracker.integral<=8.


def test_saturated_actuator_does_not_wind_up():
    tracker=SequenceLongitudinalTracker()
    for _ in range(100):tracker.step(30.,3.,0.,.05)
    assert tracker.integral==0.


def test_hard_braking_overrides_accumulated_positive_compensation():
    tracker=SequenceLongitudinalTracker();tracker.integral=8.
    throttle,brake=tracker.step(5.,-6.,10.,.05)
    assert throttle==0 and brake>.5


def sequence_call(policy,t,accel=0.,risk='low'):
    return policy.update({},dict(schema_version='longitudinal_sequence/1.0',dt_s=.1,
        speed_mps=[5.]*30,acceleration_mps2=[accel]*30),timestamp_s=t,episode_id='a',
        speed_kmh=14.4,desired_speed_kmh=18.,risk=dict(risk_level=risk))


def test_endpoint_preserves_absolute_goal_after_execution_lag():
    legacy=SequenceExecutionPolicy();fixed=SequenceExecutionPolicy(reference_mode='speed_endpoint')
    sequence_call(legacy,0);sequence_call(fixed,0)
    assert legacy.command['target_acceleration_mps2']==0.
    assert fixed.command['target_acceleration_mps2']==pytest.approx(1.)
    for i in range(1,10):
        sequence_call(fixed,i*.1)
        assert not fixed.diagnostics['operation_updated']
    assert fixed.diagnostics['selected_acceleration_mps2']==pytest.approx(1.)


def test_endpoint_never_uses_positive_correction_during_urgent_braking():
    fixed=SequenceExecutionPolicy(reference_mode='speed_endpoint')
    sequence_call(fixed,0)
    assert sequence_call(fixed,.1,-7.,'high')['action']=='emergency_brake'
    assert fixed.diagnostics['safety_bypass']


def test_invalid_reference_mode_is_rejected():
    with pytest.raises(ValueError):SequenceExecutionPolicy(reference_mode='unknown')


def test_half_second_updates_and_urgent_bypass():
    p=SequenceExecutionPolicy(interval_s=.5,forecast_steps=5)
    sequence_call(p,0,1.)
    for t in (.1,.2,.3,.4):
        sequence_call(p,t,-1.)
        assert not p.diagnostics['operation_updated']
    assert sequence_call(p,.5,-1.)['action']=='decelerate'
    assert p.diagnostics['operation_updated']
    assert sequence_call(p,.6,-7.,'high')['action']=='emergency_brake'


@pytest.mark.parametrize('interval',[.49,float('nan'),float('inf')])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):SequenceExecutionPolicy(interval_s=interval)
