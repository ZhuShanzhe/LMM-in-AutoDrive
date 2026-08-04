"""Town05_Opt route and logical-lane adapter for Scene 3.

The existing emergency actor runtime uses route-relative metres and logical
lanes (-1 left, -2 centre, -3 right, -4 shoulder, -5 sidewalk).  Official
CARLA maps use changing road/lane ids, so this adapter translates the stable
Scene 3 contract to real Town05 waypoints without exposing those ids upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
from typing import Any, Mapping, Sequence

from scenarios.complex.town05_scene2 import (
    RouteProgressTracker,
    build_repeated_route,
)


LOGICAL_ROUTE_ID = 1
LOGICAL_LEFT_LANE = -1
LOGICAL_CENTRE_LANE = -2
LOGICAL_RIGHT_LANE = -3
LOGICAL_SHOULDER_LANE = -4
LOGICAL_SIDEWALK_LANE = -5


def _lane_type_name(waypoint: Any | None) -> str:
    if waypoint is None:
        return ""
    return str(getattr(waypoint, "lane_type", "")).lower()


def _same_direction(reference: Any, candidate: Any) -> bool:
    reference_id = int(getattr(reference, "lane_id", 0))
    candidate_id = int(getattr(candidate, "lane_id", 0))
    if reference_id == 0 or candidate_id == 0:
        return True
    return (reference_id < 0) == (candidate_id < 0)


def _adjacent_driving(waypoint: Any, side: str) -> Any | None:
    method = "get_left_lane" if side == "left" else "get_right_lane"
    candidate = getattr(waypoint, method)()
    if (
        candidate is not None
        and "driving" in _lane_type_name(candidate)
        and _same_direction(waypoint, candidate)
    ):
        return candidate
    return None


def _roadside_lane(waypoint: Any, lane_type: str) -> Any | None:
    wanted = lane_type.lower()
    for side in ("right", "left"):
        current = waypoint
        method = "get_right_lane" if side == "right" else "get_left_lane"
        for _ in range(8):
            current = getattr(current, method)()
            if current is None:
                break
            if wanted in _lane_type_name(current):
                return current
    return None


@dataclass
class Town05RouteContext:
    route: Sequence[tuple[Any, Any]]
    distances_m: Sequence[float]
    tracker: RouteProgressTracker
    adapter: "Town05RouteMapAdapter"

    @property
    def length_m(self) -> float:
        return float(self.distances_m[-1])

    def progress(self, location: Any) -> float:
        return float(self.tracker.update(location))


class Town05RouteMapAdapter:
    """Map Scene 3 route metres/logical lanes onto official map waypoints."""

    def __init__(
        self,
        official_map: Any,
        route: Sequence[tuple[Any, Any]],
        distances_m: Sequence[float],
    ) -> None:
        if not route or len(route) != len(distances_m):
            raise ValueError("route and distances_m must be non-empty and aligned")
        self.official_map = official_map
        self.route = route
        self.distances_m = [float(value) for value in distances_m]

    def route_waypoint(self, progress_m: float) -> Any:
        target = min(max(float(progress_m), 0.0), self.distances_m[-1])
        index = bisect.bisect_left(self.distances_m, target)
        if index >= len(self.route):
            index = len(self.route) - 1
        if index > 0 and (
            abs(self.distances_m[index - 1] - target)
            < abs(self.distances_m[index] - target)
        ):
            index -= 1
        return self.route[index][0]

    def logical_waypoint(self, logical_lane_id: int, progress_m: float) -> Any | None:
        centre = self.route_waypoint(progress_m)
        logical_lane_id = int(logical_lane_id)
        if logical_lane_id == LOGICAL_CENTRE_LANE:
            return centre
        if logical_lane_id == LOGICAL_LEFT_LANE:
            return _adjacent_driving(centre, "left")
        if logical_lane_id == LOGICAL_RIGHT_LANE:
            return _adjacent_driving(centre, "right")
        if logical_lane_id == LOGICAL_SHOULDER_LANE:
            return _roadside_lane(centre, "shoulder") or _roadside_lane(
                centre, "sidewalk"
            )
        if logical_lane_id == LOGICAL_SIDEWALK_LANE:
            return _roadside_lane(centre, "sidewalk")
        raise ValueError(f"unsupported Scene 3 logical lane {logical_lane_id}")

    def get_waypoint_xodr(
        self,
        road_id: int,
        lane_id: int,
        s_m: float,
    ) -> Any | None:
        if int(road_id) != LOGICAL_ROUTE_ID:
            return None
        return self.logical_waypoint(int(lane_id), float(s_m))

    def get_waypoint(self, *args: Any, **kwargs: Any) -> Any:
        return self.official_map.get_waypoint(*args, **kwargs)

    def legal_driving_lane_ids(self, progress_m: float) -> set[int]:
        result: set[int] = set()
        for logical_lane in (
            LOGICAL_LEFT_LANE,
            LOGICAL_CENTRE_LANE,
            LOGICAL_RIGHT_LANE,
        ):
            waypoint = self.logical_waypoint(logical_lane, progress_m)
            if waypoint is not None:
                result.add(int(waypoint.lane_id))
        return result

    def waypoint_matches_logical_lane(
        self,
        waypoint: Any,
        logical_lane_id: int,
        progress_m: float,
    ) -> bool:
        expected = self.logical_waypoint(logical_lane_id, progress_m)
        if expected is None:
            return False
        return (
            int(getattr(waypoint, "road_id", -9999))
            == int(getattr(expected, "road_id", -9998))
            and int(getattr(waypoint, "lane_id", 0))
            == int(getattr(expected, "lane_id", 1))
        )

    def validate_anchor(self, progress_m: float, lanes: Sequence[int]) -> None:
        missing = [
            lane
            for lane in lanes
            if self.logical_waypoint(int(lane), float(progress_m)) is None
        ]
        if missing:
            raise RuntimeError(
                "Town05 route anchor lacks required logical lanes at "
                f"{float(progress_m):.1f} m: {missing}"
            )


def build_town05_route_context(
    official_map: Any,
    route_config: Mapping[str, Any],
) -> Town05RouteContext:
    target_length_m = float(route_config["target_length_m"])
    route, distances = build_repeated_route(
        official_map,
        int(route_config["start_spawn_index"]),
        int(route_config["turnaround_spawn_index"]),
        target_length_m,
        float(route_config.get("route_sampling_m", 2.0)),
    )
    if float(distances[-1]) < target_length_m:
        raise RuntimeError("Town05 route is shorter than the configured 6 km")
    adapter = Town05RouteMapAdapter(official_map, route, distances)
    return Town05RouteContext(
        route=route,
        distances_m=distances,
        tracker=RouteProgressTracker(
            route,
            distances,
            search_ahead=int(route_config.get("tracker_search_ahead", 160)),
            search_behind=int(route_config.get("tracker_search_behind", 12)),
        ),
        adapter=adapter,
    )


def validate_scene3_event_anchors(
    context: Town05RouteContext,
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Fail before spawning when a selected corridor cannot host an event."""

    for event in events:
        scenario = str(event["scenario"])
        distance_m = float(event["distance_m"])
        if scenario in {"cut_in_vehicle", "maintenance_vehicle_blockage"}:
            context.adapter.validate_anchor(distance_m, (-1, -2))
        elif scenario in {"progressive_lane_closure", "right_lane_work_zone"}:
            if scenario == "progressive_lane_closure":
                section = event["closure"]
                start_m = float(section["taper_start_s_m"])
                end_m = float(section["taper_end_s_m"])
            else:
                section = event["zone"]
                start_m = float(section["start_s_m"])
                end_m = float(section["end_s_m"])
            sample_m = start_m
            while sample_m <= end_m:
                context.adapter.validate_anchor(sample_m, (-1, -2, -3))
                sample_m += 50.0
            context.adapter.validate_anchor(end_m, (-1, -2, -3))
        elif scenario in {"temporary_worker_crossing", "work_zone_advance_warning"}:
            context.adapter.validate_anchor(distance_m, (-2, -3, -4))
