import pytest
from benchmark.traffic_guard import follow_target


def test_closing_on_slower_vehicle_reduces_target_before_collision():
    target, emergency=follow_target(60,60/3.6,42/3.6,35)
    assert 0<target<60
    assert not emergency


def test_short_stopping_gap_uses_emergency_brake():
    target, emergency=follow_target(60,60/3.6,0,20)
    assert emergency
    assert target==0


def test_large_clearance_does_not_increase_requested_speed():
    target, emergency=follow_target(30,5,10,100)
    assert target==30 and not emergency


def test_stop_command_remains_stopped():
    assert follow_target(0,0,10,30)[0]==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_invalid_speed_is_rejected(bad):
    with pytest.raises(ValueError): follow_target(60,bad,10,30)
