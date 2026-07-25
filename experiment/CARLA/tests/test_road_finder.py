import unittest

from scenarios.utils.road import RoadFinder


class _Rotation:
    def __init__(self, yaw):
        self.yaw = yaw


class _Transform:
    def __init__(self, yaw):
        self.rotation = _Rotation(yaw)


class _Waypoint:
    def __init__(self, yaw=0.0):
        self.transform = _Transform(yaw)
        self.next_waypoint = None
        self.previous_waypoint = None

    def next(self, _distance):
        return [self.next_waypoint] if self.next_waypoint is not None else []

    def previous(self, _distance):
        return [self.previous_waypoint] if self.previous_waypoint is not None else []


def _lane(length, *, turn_at=None):
    waypoints = [_Waypoint(20.0 if index == turn_at else 0.0) for index in range(length)]
    for index, waypoint in enumerate(waypoints):
        if index + 1 < len(waypoints):
            waypoint.next_waypoint = waypoints[index + 1]
        if index > 0:
            waypoint.previous_waypoint = waypoints[index - 1]
    return waypoints


def _lane_with_yaws(yaws):
    waypoints = [_Waypoint(yaw) for yaw in yaws]
    for index, waypoint in enumerate(waypoints):
        if index + 1 < len(waypoints):
            waypoint.next_waypoint = waypoints[index + 1]
        if index > 0:
            waypoint.previous_waypoint = waypoints[index - 1]
    return waypoints


class RoadFinderTests(unittest.TestCase):
    def setUp(self):
        self.finder = RoadFinder.__new__(RoadFinder)

    def test_checks_forward_and_backward_directions(self):
        waypoints = _lane(5)
        self.assertTrue(
            self.finder.check_straight_lane(
                waypoints[2], distance=10, step=5, direction="next"
            )
        )
        self.assertTrue(
            self.finder.check_straight_lane(
                waypoints[2], distance=10, step=5, direction="previous"
            )
        )

    def test_rejects_turn_in_backward_segment(self):
        waypoints = _lane(5, turn_at=1)
        self.assertFalse(
            self.finder.check_straight_lane(
                waypoints[3], distance=10, step=5, direction="previous"
            )
        )

    def test_rejects_gradual_cumulative_turn(self):
        waypoints = _lane_with_yaws([0.0, 6.0, 12.0, 18.0])
        self.assertFalse(
            self.finder.check_straight_lane(
                waypoints[0], distance=15, step=5, direction="next"
            )
        )

    def test_rejects_unknown_direction(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            self.finder.check_straight_lane(
                _Waypoint(), distance=5, step=5, direction="sideways"
            )


if __name__ == "__main__":
    unittest.main()
