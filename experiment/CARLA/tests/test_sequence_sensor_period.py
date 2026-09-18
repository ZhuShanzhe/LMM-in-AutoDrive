import pytest
from universal_vla_controller import resolve_sensor_tick


def test_legacy_sampling_period_is_unchanged():
    assert resolve_sensor_tick({},.05,2)==.1


def test_explicit_period_matches_twenty_hz_collection():
    assert resolve_sensor_tick({'sensor_tick_s':.05},.05,2)==.05


@pytest.mark.parametrize('period',[.025,.07,.2,float('nan'),float('inf')])
def test_invalid_or_unsynchronized_period_is_rejected(period):
    with pytest.raises(ValueError):resolve_sensor_tick({'sensor_tick_s':period},.05,2)
