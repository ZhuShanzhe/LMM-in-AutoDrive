"""Contract and route-progress tests for the Town05 Scene 2 runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_complex_avoidance_town05 import load_config
from scenarios.complex.town05_scene2 import (
    RouteProgressTracker,
    cumulative_route_distances,
)


@dataclass
class Location:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Transform:
    location: Location


@dataclass
class Waypoint:
    transform: Transform


def route_at(*coordinates: tuple[float, float]):
    return [
        (Waypoint(Transform(Location(x, y))), None)
        for x, y in coordinates
    ]


class Scene2Town05Tests(unittest.TestCase):
    def test_runtime_contract_has_required_scope(self):
        config = load_config(
            ROOT / "configs" / "scene_2_town05_runtime.json"
        )
        self.assertEqual(config["map"], "Town05_Opt")
        self.assertEqual(len(config["commands"]), 15)
        self.assertGreaterEqual(
            config["route"]["target_length_m"],
            8000.0,
        )
        self.assertGreaterEqual(config["traffic"]["vehicles"], 55)
        self.assertFalse(config["traffic"]["hybrid_physics"])
        self.assertFalse(
            config["traffic"]["respawn_dormant_vehicles"]
        )

    def test_cumulative_route_distances(self):
        route = route_at((0.0, 0.0), (3.0, 4.0), (9.0, 4.0))
        self.assertEqual(
            cumulative_route_distances(route),
            [0.0, 5.0, 11.0],
        )

    def test_progress_does_not_jump_to_repeated_geometry(self):
        route = route_at(
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
        )
        distances = cumulative_route_distances(route)
        tracker = RouteProgressTracker(
            route,
            distances,
            search_ahead=2,
            search_behind=1,
        )
        first = tracker.update(Location(1.0, 0.0))
        second = tracker.update(Location(11.0, 0.0))
        self.assertEqual(first, 0.0)
        self.assertEqual(second, 10.0)
        self.assertLess(tracker.index, 3)


if __name__ == "__main__":
    unittest.main()
