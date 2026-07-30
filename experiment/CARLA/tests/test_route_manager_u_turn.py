import unittest
from types import SimpleNamespace

from continuous.route_manager import RouteManager


def waypoint(yaw):
    value = SimpleNamespace(
        transform=SimpleNamespace(rotation=SimpleNamespace(yaw=float(yaw))),
    )
    value.next = lambda _step: []
    return value


class RouteManagerUTurnTests(unittest.TestCase):
    def test_u_turn_selects_the_branch_with_reversed_heading(self):
        manager = RouteManager.__new__(RouteManager)
        current = waypoint(0.0)
        straight = waypoint(0.0)
        left = waypoint(-90.0)
        reverse = waypoint(180.0)

        selected = manager._choose_turn(current, [straight, left, reverse], "u_turn")

        self.assertIs(selected, reverse)


if __name__ == "__main__":
    unittest.main()
