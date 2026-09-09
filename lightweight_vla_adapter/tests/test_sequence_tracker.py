import pytest

from lightweight_vla_adapter.src.sequence_tracker import SequenceLongitudinalTracker


def test_start_uses_acceleration_feedforward():
    tracker = SequenceLongitudinalTracker()
    throttle, brake = tracker.step(.2, 2., 0., .05)
    assert throttle > .5 and brake == 0.


def test_braking_and_stop_hold():
    tracker = SequenceLongitudinalTracker()
    throttle, brake = tracker.step(9.4, -6., 10., .05)
    assert throttle == 0. and brake > .5
    tracker.reset()
    assert tracker.step(0., 0., 0., .05) == (0., .3)


def test_reset_removes_history():
    tracker = SequenceLongitudinalTracker()
    tracker.step(.2, 2., 0., .05)
    tracker.step(.4, 2., .2, .05)
    tracker.reset()
    assert tracker.previous_speed is None
    assert tracker.acceleration == tracker.integral == 0.


@pytest.mark.parametrize('dt', [0., float('nan'), 1.])
def test_rejects_invalid_timing(dt):
    with pytest.raises(ValueError):
        SequenceLongitudinalTracker().step(1., 1., 0., dt)
