from types import SimpleNamespace

import pytest

from control.generic_route_pid import GenericRoutePID
from lightweight_vla_adapter.src.environment_speed_policy import (
    environment_speed_bound, apply_environment_speed_bound)


@pytest.mark.parametrize('goal', [0., 1.6, 10., 40.])
def test_weather_uses_route_ceiling_not_current_instruction_goal(goal):
    controller = GenericRoutePID.__new__(GenericRoutePID)
    controller._default_speed_kmh = 60.
    controller._requested_cruise_speed_kmh = 60.
    controller.ego = SimpleNamespace(get_speed_limit=lambda: 60.)
    controller.set_cruise_target_kmh(goal)
    bound = environment_speed_bound(60., controller.configured_speed_cap_kmh(),
        precipitation=90., wetness=100., sun_altitude=-20.)
    assert bound['max_speed_kmh'] == 50.
    result, _ = apply_environment_speed_bound(dict(target_speed_kmh=goal), bound, 0.)
    assert result['target_speed_kmh'] == goal
    assert controller.explicit_cruise_target_kmh() == goal


@pytest.mark.parametrize('invalid', [True, -1., float('inf'), float('nan')])
def test_invalid_route_ceiling_is_not_authorized(invalid):
    controller = GenericRoutePID.__new__(GenericRoutePID)
    controller._default_speed_kmh = invalid
    with pytest.raises(ValueError):
        controller.configured_speed_cap_kmh()
