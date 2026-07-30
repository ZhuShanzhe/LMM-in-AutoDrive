from __future__ import annotations

import unittest
from types import SimpleNamespace

from control.route_adapter import attach_route_target, route_target_location


def route(options):
    return [
        (
            SimpleNamespace(
                transform=SimpleNamespace(
                    location=SimpleNamespace(x=float(index) * 5.0, y=0.0, z=0.0)
                )
            ),
            option,
        )
        for index, option in enumerate(options)
    ]


class RouteAdapterTest(unittest.TestCase):
    def test_turn_scans_to_matching_maneuver(self):
        target, diagnostics = route_target_location(
            route(["LANEFOLLOW", "LANEFOLLOW", "LEFT", "LEFT", "LANEFOLLOW"]),
            0,
            action="turn_left",
            lookahead_m=5.0,
        )
        self.assertEqual(diagnostics["status"], "RESOLVED")
        self.assertEqual(diagnostics["maneuver_index"], 2)
        self.assertEqual(target["x"], 15.0)

    def test_wrong_turn_direction_is_rejected(self):
        target, diagnostics = route_target_location(
            route(["LANEFOLLOW", "RIGHT", "LANEFOLLOW"]),
            0,
            action="turn_left",
        )
        self.assertIsNone(target)
        self.assertEqual(diagnostics["status"], "UNAVAILABLE")

    def test_lane_change_does_not_receive_route_target(self):
        decision = {"action": "lane_change_left", "target_location": None}
        result, diagnostics = attach_route_target(
            decision, route(["LANEFOLLOW"]), 0
        )
        self.assertIsNone(result["target_location"])
        self.assertEqual(diagnostics["status"], "NOT_REQUIRED")


if __name__ == "__main__":
    unittest.main()
