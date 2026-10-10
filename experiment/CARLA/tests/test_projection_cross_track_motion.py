import math
from types import SimpleNamespace as NS

import pytest

from benchmark.truth_capture import RouteProjector


def actor(x, y, speed=6., yaw=180., z=0.):
    return NS(get_transform=lambda: NS(location=NS(x=x, y=y, z=z), rotation=NS(yaw=yaw)),
              get_velocity=lambda: NS(x=-speed, y=0., z=0.))


def route(points):
    output = []
    distance = 0.
    for index, (x, y) in enumerate(points):
        if index:
            px, py = points[index-1]
            distance += math.hypot(x-px, y-py)
        output.append(dict(x=x, y=y, z=0., distance_m=distance))
    return output


def test_adjacent_lane_crosses_diagonal_connector_without_frozen_progress():
    projector = RouteProjector(route([(20., -3.5), (11.28, -3.5), (7.085, 0.), (-20., 0.)]))
    hint = 8.72
    first = dict(route_s_m=hint, route_error_m=3.5, route_yaw_deg=180.)
    state = dict(x=11.28, y=0., z=0., yaw=180., speed_mps=6., timestamp_s=0., projection=first)
    second = projector.project_actor(actor(10.68, 0.), first['route_s_m'], .1, motion_state=state)
    assert second['route_s_m'] > first['route_s_m']+2.
    assert second['route_error_m'] < first['route_error_m']
    hint = second['route_s_m']
    for index in range(1, 41):
        result = projector.project_actor(actor(10.68-index*.6, 0.), hint, .1+index*.1,
                                         motion_state=state)
        assert result['route_s_m'] >= hint
        hint = result['route_s_m']
    assert hint == pytest.approx(8.72+math.hypot(4.195, 3.5)+7.085-(10.68-40*.6))
    assert result['route_error_m'] < .01


@pytest.mark.parametrize('previous_error', [3.5, 1000., float('nan'), -1.])
def test_cross_track_slack_never_selects_an_overlapping_later_lap(previous_error):
    projector = RouteProjector(route([(0., 0.), (10., 0.), (10., 100.),
                                      (0., 100.), (0., 0.), (10., 0.)]))
    state = dict(x=4.9, y=0., z=0., yaw=0., speed_mps=1., timestamp_s=0.,
                 projection=dict(route_s_m=4.9, route_error_m=previous_error, route_yaw_deg=0.))
    result = projector.project_actor(actor(5., 0., speed=1., yaw=0.), 4.9, .1, motion_state=state)
    assert result['route_s_m'] == pytest.approx(5.)


def test_zero_cross_track_retains_the_original_motion_bound():
    projector = RouteProjector(route([(0., 0.), (100., 0.)]))
    state = dict(x=5., y=0., z=0., yaw=0., speed_mps=1., timestamp_s=0.,
                 projection=dict(route_s_m=5., route_error_m=0., route_yaw_deg=0.))
    result = projector.project_actor(actor(5.1, 0., speed=1., yaw=0.), 5., .1, motion_state=state)
    assert result['route_s_m'] == pytest.approx(5.1)
