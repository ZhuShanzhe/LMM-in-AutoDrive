import pytest
from control.maneuver_sensor_corridor import maneuver_sensor_corridor


def test_corridor_starts_at_ego_and_converges_to_latched_target_lane():
    result = maneuver_sensor_corridor([(-10., 3.5, 0.), (0., 3.5, 0.),
                                       (10., 3.5, 0.), (20., 3.5, 0.)], (0., 0., 0.))
    assert result[0] == (0., 0., 0.)
    assert result[10] == (10., 3.5, 0.)
    assert result[-1] == (20., 3.5, 0.)
    assert all(0. <= p[1] <= 3.5 for p in result)


def test_already_entered_target_does_not_start_a_second_lane_change():
    result = maneuver_sensor_corridor([(-10., 3.5, 0.), (0., 3.5, 0.),
                                       (10., 3.5, 0.), (20., 3.5, 0.)], (1., 3.5, 0.))
    assert all(p[1] == 3.5 for p in result)


def test_static_road_grade_is_retained():
    result = maneuver_sensor_corridor([(-10., 3.5, -1.), (0., 3.5, 0.),
                                       (10., 3.5, 1.), (20., 3.5, 2.)], (0., 0., 0.))
    assert result[-1][2] == 2.


@pytest.mark.parametrize('points,ego', [
    ([], (0., 0., 0.)),
    ([(0., 0., 0.)]*4, (0., 0., 0.)),
    ([(-10., 3.5, 0.), (0., 3.5, 0.), (10., 3.5, 4.)], (0., 0., 0.)),
    ([(-10., 3.5, 0.), (0., 3.5, 0.), (10., 3.5, 0.)], (0., 10., 0.)),
    ([(-10., 3.5, 0.), (0., float('nan'), 0.), (10., 3.5, 0.)], (0., 0., 0.))])
def test_missing_ambiguous_or_invalid_geometry_remains_unavailable(points, ego):
    assert maneuver_sensor_corridor(points, ego) is None
