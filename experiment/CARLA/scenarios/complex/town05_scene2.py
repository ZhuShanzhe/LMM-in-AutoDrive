"""Town05 route, traffic flow, and deterministic actors for Scene 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from scenarios.following_lane_source import LaneTrafficSource


WALKER_LATERAL_OFFSETS_M = (
    0.0,
    0.75,
    -0.75,
    1.5,
    -1.5,
    2.5,
    -2.5,
    4.0,
    -4.0,
    6.0,
    -6.0,
    8.0,
    -8.0,
)


def stable_variant_index(
    event_id: str,
    variant_count: int,
    episode_index: int,
) -> int:
    """Select a reproducible event variant without Python's salted hash."""

    if int(variant_count) < 1:
        raise ValueError("variant_count must be positive")
    checksum = sum(
        (index + 1) * byte
        for index, byte in enumerate(str(event_id).encode("utf-8"))
    )
    return (checksum + int(episode_index)) % int(variant_count)


def materialize_event_variants(
    events: Sequence[Mapping[str, Any]],
    episode_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Merge one deterministic variant into every event definition."""

    materialized: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    for source in events:
        event = dict(source)
        variants = list(event.pop("variants", []))
        event_id = str(event["id"])
        if variants:
            variant = dict(
                variants[
                    stable_variant_index(
                        event_id,
                        len(variants),
                        episode_index,
                    )
                ]
            )
            selected[event_id] = str(
                variant.pop("variant_id", "variant")
            )
            event.update(variant)
        else:
            selected[event_id] = "default"
        materialized.append(event)
    return materialized, selected


def walker_spawn_offsets() -> tuple[tuple[float, float], ...]:
    """Return deterministic (side-shift, z-lift) spawn retries."""

    return tuple(
        (offset, z_lift)
        for z_lift in (0.45, 0.75, 1.05)
        for offset in WALKER_LATERAL_OFFSETS_M
    )


def vehicle_spawn_offsets(
    hidden_staging: bool,
) -> tuple[tuple[float, float], ...]:
    """Return longitudinal and vertical offsets for special vehicles."""

    if hidden_staging:
        # Spawn above the map where road actors cannot occupy the volume,
        # disable physics immediately, then move the actor underground.
        return ((0.0, 30.0), (0.0, 50.0), (0.0, 70.0))
    return (
        (0.0, 0.45),
        (2.0, 0.45),
        (-2.0, 0.45),
        (4.0, 0.45),
        (-4.0, 0.45),
    )


def distance_2d(left: Any, right: Any) -> float:
    return math.hypot(
        float(left.x) - float(right.x),
        float(left.y) - float(right.y),
    )


def speed_kmh(actor: Any) -> float:
    velocity = actor.get_velocity()
    return 3.6 * math.sqrt(
        velocity.x * velocity.x
        + velocity.y * velocity.y
        + velocity.z * velocity.z
    )


def cumulative_route_distances(
    route: Sequence[tuple[Any, Any]],
) -> list[float]:
    distances = [0.0]
    for index in range(1, len(route)):
        previous = route[index - 1][0].transform.location
        current = route[index][0].transform.location
        distances.append(
            distances[-1] + distance_2d(previous, current)
        )
    return distances


def route_spatial_audit(
    route: Sequence[tuple[Any, Any]],
    distances: Sequence[float],
    sample_interval_m: float = 20.0,
    cell_size_m: float = 20.0,
) -> dict[str, Any]:
    """Measure route retracing and junction density independently of turn labels."""

    if not route or len(route) != len(distances):
        raise ValueError("route and distances must be nonempty and aligned")
    if sample_interval_m <= 0.0 or cell_size_m <= 0.0:
        raise ValueError("route audit scales must be positive")

    cells: set[tuple[int, int]] = set()
    repeated = 0
    samples = 0
    sample_s = 0.0
    while sample_s <= float(distances[-1]):
        index = min(bisect.bisect_left(distances, sample_s), len(route) - 1)
        location = route[index][0].transform.location
        cell = (
            math.floor(float(location.x) / cell_size_m),
            math.floor(float(location.y) / cell_size_m),
        )
        repeated += int(cell in cells)
        cells.add(cell)
        samples += 1
        sample_s += sample_interval_m

    junction_entries = 0
    previously_inside = False
    for waypoint, _ in route:
        inside = bool(getattr(waypoint, "is_junction", False))
        junction_entries += int(inside and not previously_inside)
        previously_inside = inside

    return {
        "schema_version": "scene_route_spatial_audit/v1",
        "sample_interval_m": sample_interval_m,
        "cell_size_m": cell_size_m,
        "samples": samples,
        "unique_cells": len(cells),
        "repeated_cell_fraction": round(repeated / samples, 4),
        "junction_entries": junction_entries,
        "mean_m_per_junction": round(float(distances[-1]) / junction_entries, 1)
        if junction_entries else None,
    }


def route_curvature_degrees(
    route: Sequence[tuple[Any, Any]],
) -> float:
    """Return accumulated heading change, ignoring planner wrap-around."""

    total = 0.0
    for index in range(1, len(route)):
        previous = float(
            route[index - 1][0].transform.rotation.yaw
        )
        current = float(route[index][0].transform.rotation.yaw)
        delta = (current - previous + 180.0) % 360.0 - 180.0
        if abs(delta) <= 45.0:
            total += abs(delta)
    return total


def choose_curved_route_destination(
    carla_map: Any,
    start_spawn_index: int,
    sampling_m: float,
    candidate_limit: int = 32,
) -> tuple[int, float, float]:
    """Choose a deterministic long, curved leg on an installed Town map."""

    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner,
    )

    spawn_points = list(carla_map.get_spawn_points())
    if not 0 <= int(start_spawn_index) < len(spawn_points):
        raise ValueError("route start spawn index is unavailable")
    planner = GlobalRoutePlanner(carla_map, float(sampling_m))
    step = max(1, len(spawn_points) // max(1, int(candidate_limit)))
    candidate_indices = list(range(0, len(spawn_points), step))
    if len(spawn_points) - 1 not in candidate_indices:
        candidate_indices.append(len(spawn_points) - 1)
    scored = []
    for destination_index in candidate_indices:
        if destination_index == int(start_spawn_index):
            continue
        leg = planner.trace_route(
            spawn_points[int(start_spawn_index)].location,
            spawn_points[destination_index].location,
        )
        if len(leg) < 3:
            continue
        length_m = cumulative_route_distances(leg)[-1]
        curvature = route_curvature_degrees(leg)
        if length_m < 350.0 or curvature < 45.0:
            continue
        # Curvature is capped so a compact city block cannot outrank a
        # genuinely useful long leg merely by containing many junctions.
        score = length_m + min(curvature, 1080.0) * 0.75
        scored.append(
            (score, length_m, curvature, destination_index)
        )
    if not scored:
        raise RuntimeError("no suitable long curved route leg was found")
    _, length_m, curvature, destination_index = max(scored)
    return int(destination_index), float(length_m), float(curvature)


def build_repeated_route(
    carla_map: Any,
    start_spawn_index: int,
    turnaround_spawn_index: int,
    target_length_m: float,
    sampling_m: float,
) -> tuple[list[tuple[Any, Any]], list[float]]:
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner,
    )

    spawn_points = carla_map.get_spawn_points()
    maximum_index = max(
        int(start_spawn_index),
        int(turnaround_spawn_index),
    )
    if maximum_index >= len(spawn_points):
        raise ValueError(
            "route spawn index exceeds available Town05 spawn points"
        )
    planner = GlobalRoutePlanner(carla_map, float(sampling_m))
    endpoints = (
        int(start_spawn_index),
        int(turnaround_spawn_index),
    )
    route: list[tuple[Any, Any]] = []
    distances: list[float] = []
    leg_index = 0
    while not distances or distances[-1] < float(target_length_m):
        source = endpoints[leg_index % 2]
        destination = endpoints[(leg_index + 1) % 2]
        leg = planner.trace_route(
            spawn_points[source].location,
            spawn_points[destination].location,
        )
        if not leg:
            raise RuntimeError(
                "GlobalRoutePlanner returned an empty Town05 leg"
            )
        if route:
            leg = leg[1:]
        route.extend(leg)
        distances = cumulative_route_distances(route)
        leg_index += 1
        if leg_index > 20:
            raise RuntimeError("Town05 route failed to reach target length")

    end_index = next(
        index
        for index, distance in enumerate(distances)
        if distance >= float(target_length_m)
    )
    return route[: end_index + 1], distances[: end_index + 1]


def extend_route_for_coverage(
    planner: Any,
    spawn_points: Sequence[Any],
    prefix: Sequence[tuple[Any, Any]],
    target_length_m: float,
    candidate_limit: int = 32,
    candidate_offset: int = 1,
) -> tuple[list[tuple[Any, Any]], list[float], list[dict[str, Any]]]:
    """Continue an event-bearing route through new streets deterministically."""

    if not prefix or candidate_limit < 1:
        raise ValueError("route prefix and candidates are required")

    def occupied_cells(points: Sequence[tuple[Any, Any]]) -> set[tuple[int, int]]:
        return {
            (
                math.floor(point.transform.location.x / 20.0),
                math.floor(point.transform.location.y / 20.0),
            )
            for point, _ in points
        }

    route = list(prefix)
    stride = max(1, len(spawn_points) // candidate_limit)
    candidates = range(candidate_offset % stride, len(spawn_points), stride)
    selected_legs: list[dict[str, Any]] = []
    while cumulative_route_distances(route)[-1] < target_length_m:
        occupied = occupied_cells(route)
        source = route[-1][0].transform.location
        scored = []
        for destination_index in candidates:
            leg = planner.trace_route(
                source, spawn_points[destination_index].location
            )
            if len(leg) < 3:
                continue
            length_m = cumulative_route_distances(leg)[-1]
            if not 350.0 <= length_m <= 1600.0:
                continue
            segment_cells = occupied_cells(leg)
            novelty = len(segment_cells - occupied) / max(1, len(segment_cells))
            score = length_m * (novelty - 0.25 * (1.0 - novelty))
            scored.append((score, novelty, length_m, destination_index, leg))
        if not scored:
            raise RuntimeError("no coverage-route continuation from endpoint")
        _, novelty, length_m, destination_index, leg = max(
            scored, key=lambda item: (item[0], item[1], item[2], -item[3])
        )
        route.extend(leg[1:])
        selected_legs.append({
            "spawn_index": destination_index,
            "length_m": round(length_m, 1),
            "new_cell_fraction": round(novelty, 3),
        })
        if len(selected_legs) > 16:
            raise RuntimeError("coverage route exceeded 16 continuation legs")
    distances = cumulative_route_distances(route)
    end_index = bisect.bisect_left(distances, target_length_m)
    return route[:end_index + 1], distances[:end_index + 1], selected_legs


def build_configured_route(
    carla_map: Any, route_config: Mapping[str, Any],
) -> tuple[list[tuple[Any, Any]], list[float], list[dict[str, Any]], int]:
    """Build the identical route for runtime and actor-free preflight."""
    destination = route_config["turnaround_spawn_index"]
    if destination == "auto":
        destination, _, _ = choose_curved_route_destination(
            carla_map,
            int(route_config["start_spawn_index"]),
            float(route_config["route_sampling_m"]),
        )
    route, distances = build_repeated_route(
        carla_map,
        int(route_config["start_spawn_index"]),
        int(destination),
        float(route_config["target_length_m"]),
        float(route_config["route_sampling_m"]),
    )
    coverage_legs: list[dict[str, Any]] = []
    if route_config.get("strategy") == "coverage_greedy_v1":
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        prefix_end = bisect.bisect_left(
            distances, float(route_config["preserve_event_prefix_m"])
        )
        planner = GlobalRoutePlanner(
            carla_map, float(route_config["route_sampling_m"])
        )
        route, distances, coverage_legs = extend_route_for_coverage(
            planner,
            carla_map.get_spawn_points(),
            route[:prefix_end + 1],
            float(route_config["target_length_m"]),
            candidate_limit=int(route_config["coverage_candidate_limit"]),
            candidate_offset=int(route_config["coverage_candidate_offset"]),
        )
    return route, distances, coverage_legs, int(destination)


def route_index_at(
    distances: Sequence[float],
    progress_m: float,
) -> int:
    target = float(progress_m)
    return min(
        range(len(distances)),
        key=lambda index: abs(distances[index] - target),
    )


@dataclass
class RouteProgressTracker:
    route: Sequence[tuple[Any, Any]]
    distances: Sequence[float]
    index: int = 0
    search_ahead: int = 100
    search_behind: int = 10

    def update(self, location: Any) -> float:
        lower = max(0, self.index - self.search_behind)
        # Bound the forward search window so a repeated/looping route cannot
        # jump the tracker to a far-ahead but physically nearby segment.
        forward_window = min(int(self.search_ahead), 30)
        upper = min(len(self.route), self.index + forward_window)
        candidates = range(lower, upper)
        closest = min(
            candidates,
            key=lambda index: distance_2d(
                self.route[index][0].transform.location,
                location,
            ),
        )
        self.index = max(self.index, closest)
        return float(self.distances[self.index])


@dataclass
class ActorRegistry:
    actors: list[Any] = field(default_factory=list)

    def add(self, actor: Any | None) -> Any | None:
        if actor is not None:
            self.actors.append(actor)
        return actor

    def destroy(self, client: Any) -> None:
        for actor in reversed(self.actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                pass
        if self.actors:
            import carla

            client.apply_batch_sync(
                [
                    carla.command.DestroyActor(actor.id)
                    for actor in self.actors
                    if actor is not None
                ],
                False,
            )
        self.actors.clear()


def _set_random_blueprint_attributes(
    blueprint: Any,
    rng: random.Random,
    role_name: str,
) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    if blueprint.has_attribute("color"):
        values = list(
            blueprint.get_attribute("color").recommended_values
        )
        if values:
            blueprint.set_attribute("color", rng.choice(values))
    if blueprint.has_attribute("driver_id"):
        values = list(
            blueprint.get_attribute("driver_id").recommended_values
        )
        if values:
            blueprint.set_attribute("driver_id", rng.choice(values))


def _safe_car_blueprints(library: Any) -> list[Any]:
    excluded = (
        "ambulance",
        "firetruck",
        "carlacola",
        "fusorosa",
        "crossbike",
        "omafiets",
        "century",
    )
    result = []
    for blueprint in library.filter("vehicle.*"):
        if any(token in blueprint.id.lower() for token in excluded):
            continue
        if blueprint.has_attribute("number_of_wheels"):
            if int(blueprint.get_attribute("number_of_wheels")) != 4:
                continue
        result.append(blueprint)
    return result


class TownTrafficFlow:
    """Stable Town05 traffic based on CARLA's generate_traffic pattern."""

    BUS_BLUEPRINTS = (
        "vehicle.mitsubishi.fusorosa",
        "vehicle.volkswagen.t2_2021",
        "vehicle.volkswagen.t2",
    )

    def __init__(
        self,
        client: Any,
        world: Any,
        traffic_manager: Any,
        registry: ActorRegistry,
        route: Sequence[tuple[Any, Any]],
        config: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.world = world
        self.traffic_manager = traffic_manager
        self.registry = registry
        self.route = route
        self.config = config
        self.rng = random.Random(int(config["seed"]))
        self.vehicles: list[Any] = []
        self.walkers: list[Any] = []
        self.walker_controllers: list[Any] = []
        self.route_distances = cumulative_route_distances(route)
        self.reserved_locations: list[Any] = []
        self._maintenance_ticks = 0
        self._replenished = 0
        self._following_sources: dict[tuple[int, int], LaneTrafficSource] = {}
        self._initial_route_spawned = 0
        self.initial_route_spawns: list[dict[str, Any]] = []
        self.replenishment_events: list[dict[str, Any]] = []
        self.replenishment_settings = {
            "check_ticks": int(self.config.get("replenish_check_ticks", 20)),
            "minimum_front_vehicles": int(self.config.get("minimum_front_vehicles", 4)),
            "minimum_same_direction_vehicles": int(
                self.config.get("minimum_same_direction_vehicles", 2)
            ),
            "lookahead_m": max(
                360.0, float(self.config.get("replenish_lookahead_m", 360.0))
            ),
            "visibility_clearance_m": 350.0,
            "maximum_extra_actors": int(self.config.get("maximum_extra_actors", 12)),
            "following_source_actor_reserve": max(0, int(self.config.get("following_source_actor_reserve", 12))),
            "lane_spacing_m": max(35.0, float(self.config.get("replenish_lane_spacing_m", 75.0))),
            "staging_window_m": max(100.0, float(self.config.get("replenish_staging_window_m", 300.0))),
            "following_source_back_m": max(75.0, float(self.config.get("following_source_back_m", 100.0))),
            "following_source_spacing_m": max(40.0, float(self.config.get("following_source_spacing_m", 65.0))),
            "following_source_lane_vehicles": max(1, int(self.config.get("following_source_lane_vehicles", 2))),
            "following_source_front_vehicles": max(1, int(self.config.get("following_source_front_vehicles", 2))),
        }

    def spawn(
        self,
        reserved_locations: Iterable[Any],
        ego_location: Any,
        ego_progress_m: float = 0.0,
    ) -> None:
        self.reserved_locations = list(reserved_locations)
        self._spawn_vehicles(
            self.reserved_locations,
            ego_location,
        )
        for offset_m, lane_choice in ((50.0, 0), (75.0, 1),
                                      (105.0, 2), (135.0, 0),
                                      (170.0, 1), (205.0, 2),
                                      (240.0, 0), (275.0, 1),
                                      (315.0, 2), (355.0, 0),
                                      (400.0, 1)):
            spawned = self._spawn_route_vehicle(
                ego_location, ego_progress_m + offset_m,
                minimum_ego_distance_m=45.0, lane_choice=lane_choice,
            )
            if spawned is not None:
                self._initial_route_spawned += 1
                actor, location = spawned
                self.initial_route_spawns.append({
                    "actor_id": actor.id,
                    "requested_offset_m": offset_m,
                    "actual_distance_from_ego_m": round(distance_2d(location, ego_location), 1),
                })
        self._spawn_walkers()

    @staticmethod
    def _same_direction_lanes(base: Any) -> list[Any]:
        """Return all drivable lanes of this road, ordered right to left."""
        def matching(candidate: Any) -> bool:
            return bool(candidate is not None
                        and candidate.road_id == base.road_id
                        and candidate.lane_id * base.lane_id > 0
                        and "Driving" in str(candidate.lane_type))

        lanes = [base]
        seen = {(base.road_id, base.lane_id)}
        for side, prepend in (("get_right_lane", True), ("get_left_lane", False)):
            current = base
            for _ in range(8):
                neighbor = getattr(current, side)()
                if not matching(neighbor) or (neighbor.road_id, neighbor.lane_id) in seen:
                    break
                seen.add((neighbor.road_id, neighbor.lane_id))
                if prepend:
                    lanes.insert(0, neighbor)
                else:
                    lanes.append(neighbor)
                current = neighbor
        return lanes

    def _spawn_route_vehicle(self, ego_location: Any, route_s_m: float,
                             *, minimum_ego_distance_m: float,
                             lane_choice: int,
                             strict_lane_choice: bool = False,
                             speed_range_kmh: tuple[float, float] = (28.0, 42.0)) -> tuple[Any, Any] | None:
        import bisect
        import carla

        index = bisect.bisect_left(self.route_distances, route_s_m)
        offsets = (0, 1, 2, 3, 4) if strict_lane_choice else (0, 10, 20, 30, 40)
        for offset in offsets:
            candidate_index = min(index + offset, len(self.route) - 1)
            base = self.route[candidate_index][0]
            if base.is_junction:
                continue
            lanes = self._same_direction_lanes(base)
            if strict_lane_choice and lane_choice >= len(lanes):
                continue
            selected = lane_choice % len(lanes)
            lane_order = ([lanes[selected]] if strict_lane_choice
                          else lanes[selected:] + lanes[:selected])
            for lane in lane_order:
                location = lane.transform.location
                if (distance_2d(location, ego_location) < minimum_ego_distance_m
                        or any(distance_2d(location, reserved) < 18
                               for reserved in self.reserved_locations)
                        or any(actor.is_alive and distance_2d(location, actor.get_location()) < 22
                               for actor in self.vehicles)):
                    continue
                blueprint = self.rng.choice(
                    _safe_car_blueprints(self.world.get_blueprint_library())
                )
                _set_random_blueprint_attributes(
                    blueprint, self.rng,
                    "scene2_route_flow_{0:04d}".format(
                        self._initial_route_spawned + self._replenished),
                )
                transform = carla.Transform(
                    carla.Location(x=location.x, y=location.y, z=location.z + 0.25),
                    lane.transform.rotation,
                )
                actor = self.world.try_spawn_actor(blueprint, transform)
                if actor is None:
                    continue
                actor.set_autopilot(True, self.traffic_manager.get_port())
                if lane is base:
                    path = self._route_path(candidate_index)
                    if path:
                        self.traffic_manager.set_path(actor, path)
                self.traffic_manager.distance_to_leading_vehicle(actor, 5.0)
                self.traffic_manager.vehicle_percentage_speed_difference(
                    actor, self.rng.uniform(-3.0, 14.0)
                )
                if hasattr(self.traffic_manager, "set_desired_speed"):
                    self.traffic_manager.set_desired_speed(
                        actor, self.rng.uniform(*speed_range_kmh)
                    )
                self.traffic_manager.auto_lane_change(actor, False)
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.registry.add(actor)
                self.vehicles.append(actor)
                return actor, location
        return None

    def _route_path(self, start_index: int) -> list[Any]:
        import bisect

        start_m = self.route_distances[start_index]
        end_m = min(start_m + 600.0, self.route_distances[-1])
        indices = {
            bisect.bisect_left(self.route_distances, distance_m)
            for distance_m in range(int(start_m) + 25, int(end_m) + 1, 20)
        }
        return [self.route[index][0].transform.location for index in sorted(indices)
                if start_index < index < len(self.route)]

    @staticmethod
    def _nearby_traffic_counts(ego: Any, vehicles: Sequence[Any]) -> tuple[int, int, int]:
        origin = ego.get_location()
        forward = ego.get_transform().get_forward_vector()
        front = same_visible = same_nearby = 0
        for actor in vehicles:
            if not actor.is_alive:
                continue
            location = actor.get_location()
            dx, dy = location.x - origin.x, location.y - origin.y
            along = dx * forward.x + dy * forward.y
            across = abs(dx * forward.y - dy * forward.x)
            if not 0 < along < 350 or across >= along * 1.43:
                continue
            other_forward = actor.get_transform().get_forward_vector()
            aligned = (other_forward.x * forward.x + other_forward.y * forward.y) > 0.5
            if aligned:
                same_nearby += 1
            if along < 120:
                front += 1
                same_visible += int(aligned)
        return front, same_visible, same_nearby

    def _route_ahead_counts(self, progress_m: float) -> tuple[int, int]:
        """Count both visible and staged traffic on the upcoming route."""
        import bisect

        start = bisect.bisect_left(self.route_distances, progress_m + 8.0)
        staging_horizon_m = max(
            500.0, self.replenishment_settings["lookahead_m"] + 140.0
        )
        end = bisect.bisect_right(
            self.route_distances, progress_m + staging_horizon_m
        )
        if start >= end:
            return 0, 0
        sample_indices = list(range(start, end, 5))
        if sample_indices[-1] != end - 1:
            sample_indices.append(end - 1)
        close = ahead = 0
        get_map = getattr(self.world, "get_map", None)
        road_map = get_map() if callable(get_map) else None
        for actor in self.vehicles:
            if not actor.is_alive:
                continue
            position = actor.get_location()
            nearest = min(sample_indices, key=lambda index: distance_2d(
                position, self.route[index][0].transform.location
            ))
            waypoint = self.route[nearest][0]
            if distance_2d(position, waypoint.transform.location) > 12.0:
                continue
            if road_map is not None:
                import carla

                actor_waypoint = road_map.get_waypoint(
                    position, project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if (actor_waypoint is None
                        or actor_waypoint.road_id != waypoint.road_id
                        or actor_waypoint.section_id != waypoint.section_id
                        or actor_waypoint.lane_id * waypoint.lane_id <= 0):
                    continue
            route_forward = waypoint.transform.get_forward_vector()
            actor_forward = actor.get_transform().get_forward_vector()
            if actor_forward.x * route_forward.x + actor_forward.y * route_forward.y <= 0.5:
                continue
            ahead += 1
            if self.route_distances[nearest] - progress_m <= 120.0:
                close += 1
        return close, ahead

    def _staging_lane_gaps(self, ego_location: Any, progress_m: float) -> list[tuple[float, int]]:
        """Find empty lane cells beyond the forward camera and event actors."""
        start = progress_m + self.replenishment_settings["lookahead_m"]
        start += 35.0 * (self._replenished % 4)
        end = min(start + self.replenishment_settings["staging_window_m"],
                  self.route_distances[-1] - 30.0)
        if start >= end:
            return []

        spacing = self.replenishment_settings["lane_spacing_m"]
        actors = [(actor, actor.get_location()) for actor in self.vehicles if actor.is_alive]
        get_map = getattr(self.world, "get_map", None)
        road_map = get_map() if callable(get_map) else None
        actor_lanes = {}
        if road_map is not None:
            import carla
            for actor, location in actors:
                actor_lanes[id(actor)] = road_map.get_waypoint(
                    location, project_to_road=True, lane_type=carla.LaneType.Driving)

        cells = []
        lane_coverage = {}
        route_s_m = start
        while route_s_m <= end:
            index = min(bisect.bisect_left(self.route_distances, route_s_m), len(self.route) - 1)
            base = self.route[index][0]
            if not base.is_junction:
                for rank, lane in enumerate(self._same_direction_lanes(base)):
                    location = lane.transform.location
                    if distance_2d(location, ego_location) < self.replenishment_settings["visibility_clearance_m"]:
                        continue
                    if any(distance_2d(location, reserved) < 25.0
                           for reserved in self.reserved_locations):
                        continue
                    forward = lane.transform.get_forward_vector()
                    occupied = False
                    for actor, actor_location in actors:
                        dx = actor_location.x - location.x
                        dy = actor_location.y - location.y
                        along = abs(dx * forward.x + dy * forward.y)
                        across = abs(dx * forward.y - dy * forward.x)
                        if along >= spacing * 0.85 or across >= 2.5:
                            continue
                        actor_lane = actor_lanes.get(id(actor))
                        if (actor_lane is not None and hasattr(lane, "road_id")
                                and (actor_lane.road_id != lane.road_id
                                     or actor_lane.lane_id != lane.lane_id
                                     or getattr(actor_lane, "section_id", None)
                                     != getattr(lane, "section_id", None))):
                            continue
                        occupied = True
                        break
                    lane_coverage[rank] = lane_coverage.get(rank, 0) + int(occupied)
                    if not occupied:
                        cells.append((route_s_m, rank))
            route_s_m += spacing

        rotation = self._replenished % max(1, len(lane_coverage))
        return sorted(cells, key=lambda cell: (
            lane_coverage.get(cell[1], 0),
            (cell[1] - rotation) % max(1, len(lane_coverage)),
            cell[0],
        ))

    def _following_lane_sources(
        self, ego_location: Any, progress_m: float
    ) -> list[tuple[float, int, tuple[int, int]]]:
        """Move the per-lane spawn sources along the route behind the ego."""
        back_m = self.replenishment_settings["following_source_back_m"]
        if progress_m < back_m + 30.0:
            return []
        ego_index = bisect.bisect_left(self.route_distances, progress_m)
        if self.route[min(ego_index, len(self.route) - 1)][0].is_junction:
            return []
        source_s_m = progress_m - back_m
        index = bisect.bisect_left(self.route_distances, source_s_m)
        for offset in range(11):
            candidate_index = max(0, index - offset)
            candidate = self.route[candidate_index][0]
            if (not candidate.is_junction
                    and not any(self.route[nearby][0].is_junction
                                for nearby in range(max(0, candidate_index - 5),
                                                    min(len(self.route), candidate_index + 6)))):
                break
        else:
            return []
        source_s_m = self.route_distances[candidate_index]
        spacing = self.replenishment_settings["following_source_spacing_m"]
        actors = [actor.get_location() for actor in self.vehicles if actor.is_alive]
        candidates = []
        for rank, lane in enumerate(self._same_direction_lanes(candidate)):
            location = lane.transform.location
            if distance_2d(location, ego_location) < back_m * 0.7:
                continue
            if any(distance_2d(location, reserved) < 35.0
                   for reserved in self.reserved_locations):
                continue
            key = (lane.road_id, lane.lane_id)
            source = self._following_sources.setdefault(
                key, LaneTrafficSource(lane, source_s_m)
            )
            source.waypoint = lane
            source.route_s_m = source_s_m
            source.actors = [actor for actor in source.actors
                             if actor.is_alive
                             and distance_2d(actor.get_location(), ego_location) < 300.0]
            if progress_m - source.last_spawn_progress_m < spacing:
                continue
            forward = lane.transform.get_forward_vector()
            near_source = near_ego = 0
            for actor_location in actors:
                dx, dy = actor_location.x - location.x, actor_location.y - location.y
                along = dx * forward.x + dy * forward.y
                across = abs(dx * forward.y - dy * forward.x)
                if across >= 2.5:
                    continue
                near_source += int(-spacing * 0.5 < along < spacing)
                near_ego += int(-50.0 < along < back_m + 150.0)
            front_source_actors = sum(
                (actor.get_location().x - location.x) * forward.x
                + (actor.get_location().y - location.y) * forward.y > back_m
                for actor in source.actors
            )
            if (len(source.actors) >=
                    self.replenishment_settings["following_source_lane_vehicles"]
                    + self.replenishment_settings["following_source_front_vehicles"]):
                continue
            if front_source_actors >= self.replenishment_settings["following_source_front_vehicles"]:
                continue
            if near_source or near_ego >= self.replenishment_settings["following_source_lane_vehicles"]:
                continue
            candidates.append((near_ego, rank, source_s_m, key))
        candidates.sort(key=lambda item: (item[0], (item[1] - self._replenished) % 8))
        return [(source_s, rank, key) for _, rank, source_s, key in candidates]

    def maintain(self, ego: Any, progress_m: float) -> None:
        """Replenish only beyond the camera range, without moving task actors."""
        self._maintenance_ticks += 1
        if self._maintenance_ticks % self.replenishment_settings["check_ticks"]:
            return
        dead = [actor for actor in self.vehicles if not actor.is_alive]
        if dead:
            self.vehicles = [actor for actor in self.vehicles if actor.is_alive]
            for actor in dead:
                if actor in self.registry.actors:
                    self.registry.actors.remove(actor)
        origin = ego.get_location()
        visible, same_visible, same_nearby = self._nearby_traffic_counts(
            ego, self.vehicles
        )
        route_close, route_ahead = self._route_ahead_counts(progress_m)
        gaps = self._staging_lane_gaps(origin, progress_m)
        following_sources = self._following_lane_sources(origin, progress_m)
        if not gaps and not following_sources:
            return
        source = None
        for actor in self.vehicles:
            if not actor.is_alive or distance_2d(actor.get_location(), origin) < 350:
                continue
            location = actor.get_location()
            nearest = min(
                range(0, len(self.route), 20),
                key=lambda index: distance_2d(
                    self.route[index][0].transform.location, location
                ),
            )
            if self.route_distances[nearest] < progress_m - 300:
                source = actor
                break
        if source is None:
            distant = [actor for actor in self.vehicles
                       if actor.is_alive
                       and distance_2d(actor.get_location(), origin) >= 500.0]
            if distant:
                source = max(distant, key=lambda actor:
                             distance_2d(actor.get_location(), origin))
        if (source is None and len(self.vehicles) >=
                int(self.config["vehicles"]) +
                self.replenishment_settings["maximum_extra_actors"]):
            return
        result = None
        start = lane_choice = source_kind = source_key = None
        front_candidates = [(distance_m, lane, "forward_staging", None)
                            for distance_m, lane in gaps[:6]]
        rear_candidates = [(distance_m, lane, "following_lane_source", key)
                           for distance_m, lane, key in following_sources[:3]]
        candidates = (rear_candidates + front_candidates if self._replenished % 3 == 2
                      else front_candidates + rear_candidates)
        for candidate_s_m, candidate_lane, candidate_kind, candidate_key in candidates:
            result = self._spawn_route_vehicle(
                origin, candidate_s_m,
                minimum_ego_distance_m=(
                    self.replenishment_settings["visibility_clearance_m"]
                    if candidate_kind == "forward_staging" else 70.0
                ),
                lane_choice=candidate_lane,
                strict_lane_choice=True,
                speed_range_kmh=((28.0, 42.0) if candidate_kind == "forward_staging"
                                 else (50.0, 60.0)),
            )
            if result is not None:
                start, lane_choice = candidate_s_m, candidate_lane
                source_kind, source_key = candidate_kind, candidate_key
                break
        if result is None:
            return
        replacement, location = result
        self._replenished += 1
        if source_key is not None:
            lane_source = self._following_sources[source_key]
            lane_source.actors.append(replacement)
            lane_source.last_spawn_progress_m = progress_m
            if hasattr(ego, "get_velocity") and hasattr(replacement, "set_target_velocity"):
                import carla
                speed = min(ego.get_velocity().length(), 15.0)
                previous = lane_source.actors[-2] if len(lane_source.actors) > 1 else None
                if previous is not None:
                    speed = min(speed, previous.get_velocity().length())
                direction = lane_source.waypoint.transform.get_forward_vector()
                replacement.set_target_velocity(carla.Vector3D(
                    x=speed * direction.x, y=speed * direction.y, z=0.0
                ))
        self.replenishment_events.append({
            "ego_progress_m": round(float(progress_m), 1),
            "spawn_distance_from_ego_m": round(distance_2d(location, origin), 1),
            "front_vehicles_before": visible,
            "same_direction_visible_before": same_visible,
            "same_direction_nearby_before": same_nearby,
            "route_vehicles_120m_before": route_close,
            "route_vehicles_staging_before": route_ahead,
            "requested_spawn_route_m": round(start, 1),
            "lane_from_right": lane_choice + 1,
            "source_kind": source_kind,
            "new_actor_id": replacement.id,
            "retired_actor_id": source.id if source is not None else None,
        })
        if source is not None:
            source.set_autopilot(False, self.traffic_manager.get_port())
            source.destroy()
            self.vehicles.remove(source)
            self.registry.actors.remove(source)

    def _ordered_spawn_points(
        self,
        reserved_locations: Sequence[Any],
        ego_location: Any,
    ) -> list[Any]:
        spawn_points = list(self.world.get_map().get_spawn_points())
        self.rng.shuffle(spawn_points)
        route_locations = [
            waypoint.transform.location
            for waypoint, _ in self.route[::20]
        ]
        radius = float(self.config["route_spawn_radius_m"])

        # Keep random background traffic out of the launch corridor.  Special
        # Scene 2 actors are staged separately, so the required slow vehicle,
        # pedestrian, bus-stop and cyclist events remain unchanged.
        startup_length_m = float(
            self.config.get("startup_corridor_length_m", 450.0)
        )
        startup_radius_m = float(
            self.config.get("startup_corridor_radius_m", 12.0)
        )
        startup_locations: list[Any] = []
        travelled_m = 0.0
        previous_location = None
        for waypoint, _ in self.route:
            location = waypoint.transform.location
            if previous_location is not None:
                travelled_m += distance_2d(previous_location, location)
            if travelled_m > startup_length_m:
                break
            startup_locations.append(location)
            previous_location = location

        print(
            "Traffic startup corridor: length={0:.0f} m, radius={1:.0f} "
            "m, ego exclusion={2:.0f} m".format(
                startup_length_m,
                startup_radius_m,
                float(self.config.get("ego_spawn_exclusion_m", 35.0)),
            )
        )

        def allowed(transform: Any) -> bool:
            location = transform.location
            if distance_2d(location, ego_location) < float(
                self.config.get("ego_spawn_exclusion_m", 35.0)
            ):
                return False
            if any(
                distance_2d(location, route_location)
                < startup_radius_m
                for route_location in startup_locations
            ):
                return False
            return all(
                distance_2d(location, reserved) >= 14.0
                for reserved in reserved_locations
            )

        available = [point for point in spawn_points if allowed(point)]
        near = [
            point
            for point in available
            if min(
                distance_2d(point.location, route_location)
                for route_location in route_locations
            )
            <= radius
        ]
        far = [point for point in available if point not in near]
        near_start = sorted(
            [
                point
                for point in near
                if distance_2d(point.location, ego_location) <= 150.0
            ],
            key=lambda point: distance_2d(
                point.location,
                ego_location,
            ),
        )
        near_start_ids = {id(point) for point in near_start}
        route_remainder = [
            point for point in near if id(point) not in near_start_ids
        ]
        return near_start[:16] + route_remainder + far

    def _spawn_vehicles(
        self,
        reserved_locations: Sequence[Any],
        ego_location: Any,
    ) -> None:
        import carla

        library = self.world.get_blueprint_library()
        cars = _safe_car_blueprints(library)
        buses = []
        for identifier in self.BUS_BLUEPRINTS:
            try:
                buses.append(library.find(identifier))
            except (IndexError, RuntimeError):
                continue
        requested = max(
            0, int(self.config["vehicles"])
            - self.replenishment_settings["following_source_actor_reserve"],
        )
        bus_count = min(int(self.config["buses"]), requested)
        blueprints = [
            buses[index % len(buses)]
            for index in range(bus_count)
        ] if buses else []
        blueprints.extend(
            self.rng.choice(cars)
            for _ in range(requested - len(blueprints))
        )
        spawn_points = self._ordered_spawn_points(
            reserved_locations,
            ego_location,
        )
        batch = []
        for index, (blueprint, transform) in enumerate(
            zip(blueprints, spawn_points)
        ):
            _set_random_blueprint_attributes(
                blueprint,
                self.rng,
                "scene2_traffic_{0:03d}".format(index),
            )
            batch.append(
                carla.command.SpawnActor(
                    blueprint,
                    transform,
                ).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor,
                        True,
                        self.traffic_manager.get_port(),
                    )
                )
            )
        responses = self.client.apply_batch_sync(batch, False)
        for response in responses:
            if response.error:
                continue
            actor = self.world.get_actor(response.actor_id)
            if actor is None:
                continue
            self.registry.add(actor)
            self.vehicles.append(actor)
            self.traffic_manager.distance_to_leading_vehicle(
                actor,
                self.rng.uniform(4.5, 6.0),
            )
            self.traffic_manager.vehicle_percentage_speed_difference(
                actor,
                self.rng.uniform(-3.0, 14.0),
            )
            auto_lane_change = bool(
                self.config.get("ambient_auto_lane_change", False)
            )
            self.traffic_manager.auto_lane_change(
                actor,
                auto_lane_change,
            )
            lane_change = (
                float(self.config["random_lane_change_percentage"])
                if auto_lane_change
                else 0.0
            )
            self.traffic_manager.random_left_lanechange_percentage(
                actor,
                lane_change,
            )
            self.traffic_manager.random_right_lanechange_percentage(
                actor,
                lane_change,
            )
            self.traffic_manager.update_vehicle_lights(actor, True)

    def _spawn_walkers(self) -> None:
        import carla

        library = self.world.get_blueprint_library()
        walker_blueprints = list(
            library.filter("walker.pedestrian.*")
        )
        controller_blueprint = library.find("controller.ai.walker")
        requested = int(self.config["ambient_walkers"])
        for index in range(requested):
            location = None
            if self.route:
                # Put most ambient pedestrians on sidewalks alongside the
                # recorded route.  Pure navigation-mesh random sampling can
                # place every walker in another district of a large town.
                route_index = min(
                    len(self.route) - 1,
                    35 + index * max(12, len(self.route) // max(requested, 1)),
                )
                road_waypoint = self.route[route_index][0]
                sidewalk = _lane_sidewalk(
                    road_waypoint,
                    "right" if index % 2 == 0 else "left",
                )
                if sidewalk is not None:
                    location = sidewalk.transform.location
            if location is None:
                location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            transform = carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.25,
                )
            )
            blueprint = self.rng.choice(walker_blueprints)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            self.registry.add(actor)
            self.walkers.append(actor)
            controller = self.world.try_spawn_actor(
                controller_blueprint,
                carla.Transform(),
                attach_to=actor,
            )
            if controller is None:
                continue
            self.registry.add(controller)
            self.walker_controllers.append(controller)
            controller.start()
            destination = self.world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(self.rng.uniform(1.0, 1.55))

    def nearby_counts(
        self,
        location: Any,
        radius_m: float = 85.0,
    ) -> dict[str, int]:
        return {
            "vehicles": sum(
                actor.is_alive
                and distance_2d(actor.get_location(), location) <= radius_m
                for actor in self.vehicles
            ),
            "walkers": sum(
                actor.is_alive
                and distance_2d(actor.get_location(), location) <= radius_m
                for actor in self.walkers
            ),
        }


def _lane_sidewalk(
    waypoint: Any,
    side: str,
    maximum_hops: int = 12,
) -> Any | None:
    current = waypoint
    method_name = "get_left_lane" if side == "left" else "get_right_lane"
    for _ in range(maximum_hops):
        current = getattr(current, method_name)()
        if current is None:
            return None
        if "Sidewalk" in str(current.lane_type):
            return current
    return None


def roadside_bus_waypoint(driving_waypoint: Any, carla_module: Any) -> Any:
    shoulder = driving_waypoint.get_right_lane()
    if (
        shoulder is None
        or shoulder.lane_type != carla_module.LaneType.Shoulder
        or float(shoulder.lane_width) < 3.0
    ):
        raise ValueError("bus stop requires a right-side shoulder at the route anchor")
    return shoulder


def crossing_endpoints(waypoint: Any) -> tuple[Any, Any]:
    import carla

    left = _lane_sidewalk(waypoint, "left")
    right = _lane_sidewalk(waypoint, "right")
    if left is not None and right is not None:
        return left.transform.location, right.transform.location
    transform = waypoint.transform
    right_vector = transform.get_right_vector()
    center = transform.location
    return (
        center
        - carla.Location(
            x=right_vector.x * 9.0,
            y=right_vector.y * 9.0,
        ),
        center
        + carla.Location(
            x=right_vector.x * 9.0,
            y=right_vector.y * 9.0,
        ),
    )


def crosswalk_polygon_endpoints(
    carla_map: Any,
    polygon_index: int,
    inset_m: float = 0.35,
    clearance_m: float = 0.0,
) -> tuple[Any, Any]:
    """Return pedestrian endpoints along an official crosswalk long axis."""

    import carla
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('crosswalk clearance must be finite and nonnegative')

    def point_distance(left: Any, right: Any) -> float:
        return math.hypot(
            float(left.x) - float(right.x),
            float(left.y) - float(right.y),
        )

    polygons: list[list[Any]] = []
    current: list[Any] = []
    for point in carla_map.get_crosswalks():
        if not current:
            current = [point]
            continue
        current.append(point)
        if (
            len(current) >= 4
            and point_distance(current[0], current[-1]) < 0.05
        ):
            polygons.append(current[:-1])
            current = []

    index = int(polygon_index)
    if not 0 <= index < len(polygons):
        raise ValueError(
            "crosswalk polygon index {0} is unavailable; map has {1}".format(
                index,
                len(polygons),
            )
        )
    points = polygons[index]
    if len(points) < 3:
        raise RuntimeError("selected crosswalk polygon is degenerate")

    center_x = sum(float(point.x) for point in points) / len(points)
    center_y = sum(float(point.y) for point in points) / len(points)
    center_z = sum(float(point.z) for point in points) / len(points)
    covariance_xx = sum(
        (float(point.x) - center_x) ** 2 for point in points
    ) / len(points)
    covariance_yy = sum(
        (float(point.y) - center_y) ** 2 for point in points
    ) / len(points)
    covariance_xy = sum(
        (float(point.x) - center_x)
        * (float(point.y) - center_y)
        for point in points
    ) / len(points)
    angle = 0.5 * math.atan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    )
    axis_x = math.cos(angle)
    axis_y = math.sin(angle)
    projections = [
        (float(point.x) - center_x) * axis_x
        + (float(point.y) - center_y) * axis_y
        for point in points
    ]
    minimum = min(projections)
    maximum = max(projections)
    inset = min(
        max(0.0, float(inset_m)),
        max(0.0, (maximum - minimum) * 0.2),
    )
    start_projection = minimum + inset
    target_projection = maximum - inset
    if clearance_m > 0:
        start_projection = minimum - clearance_m
        target_projection = maximum + clearance_m
    if target_projection - start_projection < 4.0:
        raise RuntimeError("selected crosswalk is too short for Scene 2")
    return (
        carla.Location(
            x=center_x + axis_x * start_projection,
            y=center_y + axis_y * start_projection,
            z=center_z,
        ),
        carla.Location(
            x=center_x + axis_x * target_projection,
            y=center_y + axis_y * target_projection,
            z=center_z,
        ),
    )


class ScriptedWalker:
    def __init__(
        self,
        actor: Any,
        target: Any,
        speed_mps: float,
        *,
        activation_transform: Any | None = None,
        pause_fraction: float | None = None,
        pause_ticks: int = 0,
    ) -> None:
        self.actor = actor
        self.target = target
        self.speed_mps = float(speed_mps)
        self.activation_transform = activation_transform
        self.active = False
        self.completed = False
        self.pause_fraction = (
            None
            if pause_fraction is None
            else max(0.05, min(0.95, float(pause_fraction)))
        )
        self.pause_ticks = max(0, int(pause_ticks))
        # Per-event completion tolerance.  The default remains
        # strict for bus passengers; the official crosswalk can
        # opt into a curb-safe value from runtime configuration.
        self.completion_distance_m = 0.8
        self._pause_remaining = 0
        self._pause_started = False
        self._initial_distance: float | None = None
        self._activation_pending = False
        self._activation_settle_ticks = 0
        self._activation_retries = 0
        self._activation_validation_pending = False

    def start(self) -> None:
        if self.active:
            return
        if not self.actor.is_alive:
            raise RuntimeError(
                "scripted walker actor became unavailable before activation"
            )
        if self.activation_transform is not None:
            self.actor.set_transform(self.activation_transform)
            self._activation_pending = True
            self._activation_settle_ticks = 1
        self.active = True

    def update(self) -> None:
        if not self.active or self.completed:
            return
        if not self.actor.is_alive:
            raise RuntimeError(
                "scripted walker actor became unavailable after "
                "activation"
            )
        if self._activation_pending:
            if self._activation_settle_ticks > 0:
                self._activation_settle_ticks -= 1
                return
            # CARLA walkers may reset to the world origin if physics is
            # re-enabled in the same server tick as an underground teleport.
            # Commit the visible transform for one tick first, then enable
            # physics and write the transform again after the physics reset.
            self.actor.set_simulate_physics(True)
            self.actor.set_transform(self.activation_transform)
            self._activation_pending = False
            self._activation_validation_pending = True
            return
        import carla

        location = self.actor.get_location()
        if (
            self._activation_validation_pending
            and self.activation_transform is not None
        ):
            expected = self.activation_transform.location
            restore_error_m = distance_2d(location, expected)
            restore_height_error_m = abs(
                float(location.z) - float(expected.z)
            )
            if restore_error_m > 2.0 or restore_height_error_m > 2.0:
                if self._activation_retries >= 3:
                    raise RuntimeError(
                        "scripted walker restored at an unexpected "
                        "location: horizontal_error_m={0:.3f}, "
                        "height_error_m={1:.3f}".format(
                            restore_error_m,
                            restore_height_error_m,
                        )
                    )
                self._activation_retries += 1
                self.actor.set_transform(self.activation_transform)
                return
            self._activation_validation_pending = False
        dx = self.target.x - location.x
        dy = self.target.y - location.y
        distance = math.hypot(dx, dy)
        if self._initial_distance is None:
            self._initial_distance = max(distance, 0.001)
        progress = 1.0 - distance / self._initial_distance
        if (
            self.pause_fraction is not None
            and not self._pause_started
            and progress >= self.pause_fraction
        ):
            self._pause_started = True
            self._pause_remaining = self.pause_ticks
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            self.actor.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(),
                    speed=0.0,
                )
            )
            return
        if distance <= self.completion_distance_m:
            self.actor.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(),
                    speed=0.0,
                )
            )
            self.completed = True
            return
        self.actor.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(
                    x=dx / distance,
                    y=dy / distance,
                    z=0.0,
                ),
                speed=self.speed_mps,
            )
        )


class DeterministicSceneEvents:
    """Special actors remain staged and activate from route progress."""

    def __init__(
        self,
        world: Any,
        traffic_manager: Any,
        registry: ActorRegistry,
        route: Sequence[tuple[Any, Any]],
        distances: Sequence[float],
        events: Sequence[Mapping[str, Any]],
        seed: int,
        episode_index: int = 0,
        preserve_roles: Sequence[str] = (),
        ego: Any | None = None,
    ) -> None:
        self.world = world
        self.ego = ego
        self.traffic_manager = traffic_manager
        self.registry = registry
        self.route = route
        self.distances = distances
        self.events, self.selected_variants = materialize_event_variants(
            events,
            episode_index,
        )
        self.rng = random.Random(int(seed) + 17)
        self.states = {
            str(event["id"]): "STAGED"
            for event in self.events
        }
        self.reserved_locations: list[Any] = []
        self.scripted_walkers: dict[str, ScriptedWalker] = {}
        self.bindings: dict[str, Any] = {}
        self.spawn_diagnostics: dict[str, dict[str, Any]] = {}
        self.slow_vehicle: Any | None = None
        self._slow_vehicle_prestaged = False
        self.route_slow_vehicles: dict[str, Any] = {}
        for event in self.events:
            if event['kind'] == 'route_slow_vehicle':
                if event.get('resolve_progress_m') is not None:
                    raise ValueError('route slow vehicles remain physical until episode cleanup')
                if not 0 <= float(event['activate_progress_m']) < float(event['anchor_progress_m']):
                    raise ValueError('route slow vehicle release must precede its parked position')
                if float(event['complete_progress_m']) <= float(event['anchor_progress_m']):
                    raise ValueError('route slow vehicle event end must follow its parked position')
        self.cyclist: Any | None = None
        self.cyclist_transform: Any | None = None
        self._cyclist_placed = False
        self.bus: Any | None = None
        self.bus_transform: Any | None = None
        self.bus_active_ticks = 0
        self._retired_events: set[str] = set()
        self.preserve_roles = frozenset(str(role) for role in preserve_roles)
        self.preserve_roles |= frozenset(
            role for event in self.events if event.get('retain_after_completion',False)
            for role in event.get('ground_truth',{}).get('actor_roles',[])
        )
        self._spawned = False

    def _set_desired_speed(
        self,
        actor: Any,
        target_speed_kmh: float,
        fallback_difference_pct: float,
    ) -> None:
        setter = getattr(
            self.traffic_manager,
            "set_desired_speed",
            None,
        )
        if callable(setter):
            setter(actor, float(target_speed_kmh))
            return
        self.traffic_manager.vehicle_percentage_speed_difference(
            actor,
            float(fallback_difference_pct),
        )

    def _retire_actor(self, role_name: str, actor: Any) -> bool:
        """Stop and hide an actor whose event has completed.

        Completed deterministic actors must remain registered for normal
        teardown, but they must no longer occupy the ego route.  Leaving a
        stopped bus, walker, or slow vehicle in place can make BehaviorAgent
        wait forever after the event has already been marked RESOLVED.
        """

        if actor is None:
            return False
        try:
            if not bool(actor.is_alive):
                return False
        except (AttributeError, RuntimeError):
            return False

        import carla

        type_id = str(getattr(actor, "type_id", ""))
        if type_id.startswith("vehicle."):
            try:
                actor.set_autopilot(
                    False,
                    self.traffic_manager.get_port(),
                )
            except RuntimeError:
                pass
            actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )
        elif type_id.startswith("walker.pedestrian."):
            actor.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(),
                    speed=0.0,
                )
            )

        actor.set_simulate_physics(False)
        hidden = actor.get_transform()
        hidden.location.z = -50.0
        actor.set_transform(hidden)
        self.spawn_diagnostics.setdefault(role_name, {}).update(
            {
                "retirement": "hidden_physics_disabled",
                "retirement_z_m": -50.0,
            }
        )
        return True

    def _retire_event_actors(self, event_id: str, *, force: bool = False) -> int:
        """Remove resolved event actors from the drivable corridor once."""

        event_id = str(event_id)
        if event_id in self._retired_events:
            return 0

        roles: list[str]
        if event_id == "slow_vehicle":
            roles = ["scene2_slow_vehicle"]
        elif event_id == "crosswalk_pedestrian":
            roles = ["scene2_crosswalk_pedestrian"]
        elif event_id == "bus_stop_passengers":
            roles = ["scene2_bus_stop_bus"] + [
                role
                for role in sorted(self.bindings)
                if role.startswith("scene2_bus_passenger_")
            ]
        elif event_id == "slow_cyclist":
            roles = ["scene2_slow_cyclist"]
        elif event_id in self.route_slow_vehicles:
            roles = [
                str(event["actor_role"])
                for event in self.events if str(event["id"]) == event_id
            ]
        else:
            roles = []

        retained = [] if force else [
            role for role in roles if role in self.preserve_roles
        ]
        if retained:
            for role in retained:
                self.spawn_diagnostics.setdefault(role, {}).update(
                    retirement="deferred_for_independent_evaluation")
            return 0

        retired = sum(
            1
            for role_name in roles
            if self._retire_actor(
                role_name,
                self.bindings.get(role_name),
            )
        )
        self._retired_events.add(event_id)
        return retired

    def _waypoint(self, progress_m: float) -> Any:
        return self.route[
            route_index_at(self.distances, progress_m)
        ][0]

    def _prestage_slow_vehicle(
        self, event: Mapping[str, Any], progress_m: float
    ) -> None:
        if self._slow_vehicle_prestaged or self.slow_vehicle is None:
            return
        activate_at = float(event["activate_progress_m"])
        if progress_m < activate_at - float(
            event.get("prestage_before_activation_m", 80.0)
        ):
            return
        if self.ego is None:
            raise RuntimeError("slow vehicle prestaging requires the ego actor")

        import carla

        ego_transform = self.ego.get_transform()
        origin = ego_transform.location
        forward = ego_transform.get_forward_vector()
        vehicles = self.world.get_actors().filter("vehicle.*")
        for offset_m in (0.0, 12.0, -12.0, 24.0, -24.0):
            waypoint = self._waypoint(float(event["anchor_progress_m"]) + offset_m)
            location = waypoint.transform.location
            dx, dy = location.x - origin.x, location.y - origin.y
            along = dx * forward.x + dy * forward.y
            lateral = abs(dx * forward.y - dy * forward.x)
            if along > 0.0 and lateral < along:
                continue
            if any(
                actor.id != self.slow_vehicle.id
                and distance_2d(actor.get_location(), location) < 9.0
                for actor in vehicles if actor.is_alive
            ):
                continue
            transform = carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 0.45),
                waypoint.transform.rotation,
            )
            self.slow_vehicle.set_transform(transform)
            self.slow_vehicle.set_simulate_physics(True)
            self.slow_vehicle.apply_control(
                carla.VehicleControl(brake=1.0, hand_brake=True)
            )
            self.reserved_locations.append(transform.location)
            self._slow_vehicle_prestaged = True
            self.spawn_diagnostics["scene2_slow_vehicle"].update({
                "prestage_progress_m": round(float(progress_m), 3),
                "prestage_route_offset_m": offset_m,
                "prestage_forward_m": round(along, 3),
                "prestage_lateral_m": round(lateral, 3),
                "activation_source": "physical_parked_release",
            })
            return
        if progress_m >= activate_at - 20.0:
            raise RuntimeError("slow vehicle cannot be prestaged outside ego view")

    def _activate_slow_vehicle(
        self,
        event: Mapping[str, Any],
        progress_m: float,
    ) -> None:
        """Release the already visible slow vehicle without relocating it."""

        if self.slow_vehicle is None or not self._slow_vehicle_prestaged:
            raise RuntimeError("slow vehicle was not prestaged before activation")
        import carla

        self.slow_vehicle.apply_control(carla.VehicleControl())
        self.slow_vehicle.set_autopilot(
            True,
            self.traffic_manager.get_port(),
        )
        self._set_desired_speed(
            self.slow_vehicle,
            float(event["target_speed_kmh"]),
            60.0,
        )
        self.traffic_manager.auto_lane_change(
            self.slow_vehicle,
            False,
        )
        self.traffic_manager.update_vehicle_lights(
            self.slow_vehicle,
            True,
        )
        self.spawn_diagnostics["scene2_slow_vehicle"].update(
            {
                "activation_progress_m": round(float(progress_m), 3),
            }
        )

    def _spawn_route_slow_vehicle(self, event: Mapping[str, Any]) -> None:
        """Park before capture; release in place without teleporting or hiding."""
        import carla
        role = str(event['actor_role'])
        actor = self._spawn_vehicle(
            ('vehicle.audi.tt', 'vehicle.nissan.micra'),
            self._waypoint(float(event['anchor_progress_m'])), role,
            hidden_staging=False,
        )
        actor.set_autopilot(False, self.traffic_manager.get_port())
        actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        self.route_slow_vehicles[str(event['id'])] = actor
        self.spawn_diagnostics[role]['activation_source'] = 'physical_parked_release'

    def _activate_route_slow_vehicle(self, event: Mapping[str, Any]) -> None:
        import carla
        actor = self.route_slow_vehicles[str(event['id'])]
        if not actor.is_alive:
            raise RuntimeError('route slow vehicle disappeared before release')
        actor.apply_control(carla.VehicleControl())
        actor.set_autopilot(True, self.traffic_manager.get_port())
        self._set_desired_speed(actor, float(event['target_speed_kmh']), 60.0)
        self.traffic_manager.auto_lane_change(actor, False)
        self.traffic_manager.update_vehicle_lights(actor, True)

    def _activate_bus(self) -> None:
        """Keep the physically staged bus stationary without relocating it."""

        if self.bus is None or self.bus_transform is None:
            raise RuntimeError('bus-stop actor was not prepared')
        if not self.bus.is_alive:
            raise RuntimeError('bus-stop actor disappeared before activation')
        import carla

        self.bus.set_autopilot(
            False,
            self.traffic_manager.get_port(),
        )
        self.bus.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
        self.spawn_diagnostics["scene2_bus_stop_bus"].update(
            {
                "activation_source": "physical_parked_bus_no_relocation",
                "activation_location": {
                    "x": round(
                        float(self.bus_transform.location.x), 3
                    ),
                    "y": round(
                        float(self.bus_transform.location.y), 3
                    ),
                    "z": round(
                        float(self.bus_transform.location.z), 3
                    ),
                },
            }
        )

    def _spawn_vehicle(
        self,
        blueprint_ids: Sequence[str],
        waypoint: Any,
        role_name: str,
        *,
        hidden_staging: bool = False,
    ) -> Any:
        library = self.world.get_blueprint_library()
        blueprint = None
        for identifier in blueprint_ids:
            try:
                blueprint = library.find(identifier)
                break
            except (IndexError, RuntimeError):
                continue
        if blueprint is None:
            raise RuntimeError(
                "none of the special actor blueprints are available"
            )
        _set_random_blueprint_attributes(
            blueprint,
            self.rng,
            role_name,
        )
        import carla

        original = waypoint.transform
        forward = original.get_forward_vector()
        actor = None
        attempts = 0
        for longitudinal_m, vertical_m in vehicle_spawn_offsets(
            hidden_staging
        ):
            attempts += 1
            transform = carla.Transform(
                carla.Location(
                    x=original.location.x + forward.x * longitudinal_m,
                    y=original.location.y + forward.y * longitudinal_m,
                    z=original.location.z + vertical_m,
                ),
                original.rotation,
            )
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError("failed to stage {0}".format(role_name))

        # ``get_transform()`` can transiently report the CARLA world origin
        # before the first tick.  Preserve the exact transform accepted by
        # try_spawn_actor so hidden actors can later return to the real route.
        if not hasattr(self, "_vehicle_spawn_transforms"):
            self._vehicle_spawn_transforms = {}
        self._vehicle_spawn_transforms[role_name] = carla.Transform(
            carla.Location(
                x=float(transform.location.x),
                y=float(transform.location.y),
                z=float(transform.location.z),
            ),
            carla.Rotation(
                pitch=float(transform.rotation.pitch),
                yaw=float(transform.rotation.yaw),
                roll=float(transform.rotation.roll),
            ),
        )
        if hidden_staging:
            actor.set_simulate_physics(False)
            hidden_transform = actor.get_transform()
            hidden_transform.location.z = -20.0
            actor.set_transform(hidden_transform)
        self.registry.add(actor)
        self.bindings[role_name] = actor
        self.spawn_diagnostics[role_name] = {
            "attempts": attempts,
            "source": (
                "hidden_air_staging"
                if hidden_staging
                else "route_waypoint_retry"
            ),
        }
        if not hidden_staging:
            self.reserved_locations.append(actor.get_location())
        return actor

    def _spawn_walker(
        self,
        start: Any,
        target: Any,
        role_name: str,
        speed_mps: float,
        *,
        pause_fraction: float | None = None,
        pause_ticks: int = 0,
        physical_staging: bool = False,
    ) -> ScriptedWalker:
        import carla

        library = self.world.get_blueprint_library()
        blueprints = list(library.filter("walker.pedestrian.*"))
        if not blueprints:
            raise RuntimeError("no pedestrian blueprints are available")
        self.rng.shuffle(blueprints)
        crossing_dx = float(target.x) - float(start.x)
        crossing_dy = float(target.y) - float(start.y)
        length = max(0.001, math.hypot(crossing_dx, crossing_dy))
        shift_x = -crossing_dy / length
        shift_y = crossing_dx / length
        actor = None
        activation_transform = None
        selected_target = target
        attempts = 0
        source = "sidewalk_retry"
        for attempt_index, (offset, z_lift) in enumerate(
            walker_spawn_offsets()
        ):
            attempts += 1
            blueprint = blueprints[attempt_index % len(blueprints)]
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "true")
            if blueprint.has_attribute("speed"):
                blueprint.set_attribute(
                    "speed", "{0:.3f}".format(float(speed_mps))
                )
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)
            candidate = carla.Location(
                x=float(start.x) + shift_x * offset,
                y=float(start.y) + shift_y * offset,
                z=float(start.z) + z_lift,
            )
            actor = self.world.try_spawn_actor(
                blueprint,
                carla.Transform(candidate),
            )
            if actor is not None:
                activation_transform = carla.Transform(
                    carla.Location(
                        x=float(candidate.x),
                        y=float(candidate.y),
                        z=float(candidate.z),
                    )
                )
                selected_target = carla.Location(
                    x=float(target.x) + shift_x * offset,
                    y=float(target.y) + shift_y * offset,
                    z=float(target.z),
                )
                break
        if actor is None:
            # Do not rely on a global random nav-mesh sample for a local
            # event.  On Town05 the chance of landing near this one bus stop
            # is very small.  Instead, walk along the already planned route
            # and retry exact sidewalk waypoints before using randomness.
            source = "route_sidewalk_fallback"
            closest_index = min(
                range(len(self.route)),
                key=lambda index: distance_2d(
                    self.route[index][0].transform.location,
                    start,
                ),
            )
            route_offsets = (0, 2, -2, 4, -4, 7, -7, 11, -11, 16, -16, 24, -24)
            seen_locations: set[tuple[int, int]] = set()
            for route_offset in route_offsets:
                route_index = closest_index + route_offset
                if not 0 <= route_index < len(self.route):
                    continue
                road_waypoint = self.route[route_index][0]
                for side in ("right", "left"):
                    sidewalk = _lane_sidewalk(road_waypoint, side)
                    if sidewalk is None:
                        continue
                    navigation = sidewalk.transform.location
                    if distance_2d(navigation, start) > 65.0:
                        continue
                    key = (round(navigation.x * 10), round(navigation.y * 10))
                    if key in seen_locations:
                        continue
                    seen_locations.add(key)
                    attempts += 1
                    blueprint = blueprints[attempts % len(blueprints)]
                    if blueprint.has_attribute("is_invincible"):
                        blueprint.set_attribute("is_invincible", "true")
                    if blueprint.has_attribute("speed"):
                        blueprint.set_attribute(
                            "speed", "{0:.3f}".format(float(speed_mps))
                        )
                    if blueprint.has_attribute("role_name"):
                        blueprint.set_attribute("role_name", role_name)
                    actor = self.world.try_spawn_actor(
                        blueprint,
                        carla.Transform(
                            carla.Location(
                                x=navigation.x,
                                y=navigation.y,
                                z=navigation.z + 0.55,
                            )
                        ),
                    )
                    if actor is not None:
                        activation_transform = carla.Transform(
                            carla.Location(
                                x=float(navigation.x),
                                y=float(navigation.y),
                                z=float(navigation.z) + 0.55,
                            )
                        )
                        selected_target = carla.Location(
                            x=navigation.x + crossing_dx,
                            y=navigation.y + crossing_dy,
                            z=navigation.z,
                        )
                        break
                if actor is not None:
                    break
        if actor is None:
            # Packaged towns occasionally contain an invisible collision
            # volume on a sidewalk waypoint.  A nearby navigation-mesh point
            # is safer than dropping a required competition actor.
            source = "navigation_mesh_fallback"
            for _ in range(256):
                navigation = self.world.get_random_location_from_navigation()
                if navigation is None or distance_2d(navigation, start) > 65.0:
                    continue
                attempts += 1
                blueprint = blueprints[attempts % len(blueprints)]
                if blueprint.has_attribute("is_invincible"):
                    blueprint.set_attribute("is_invincible", "true")
                if blueprint.has_attribute("speed"):
                    blueprint.set_attribute(
                        "speed", "{0:.3f}".format(float(speed_mps))
                    )
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", role_name)
                actor = self.world.try_spawn_actor(
                    blueprint,
                    carla.Transform(
                        carla.Location(
                            x=navigation.x,
                            y=navigation.y,
                            z=navigation.z + 0.45,
                        )
                    ),
                )
                if actor is not None:
                    activation_transform = carla.Transform(
                        carla.Location(
                            x=float(navigation.x),
                            y=float(navigation.y),
                            z=float(navigation.z) + 0.45,
                        )
                    )
                    selected_target = carla.Location(
                        x=navigation.x + crossing_dx,
                        y=navigation.y + crossing_dy,
                        z=navigation.z,
                    )
                    break
        if actor is None:
            raise RuntimeError(
                "failed to stage {0} after {1} collision-safe attempts".format(
                    role_name,
                    attempts,
                )
            )
        if activation_transform is None:
            raise RuntimeError(
                "scripted walker activation transform was not captured"
            )
        activation_location = carla.Location(
            x=float(activation_transform.location.x),
            y=float(activation_transform.location.y),
            z=float(activation_transform.location.z),
        )
        hidden_transform = carla.Transform(
            carla.Location(
                x=float(activation_transform.location.x),
                y=float(activation_transform.location.y),
                z=-20.0,
            ),
            carla.Rotation(
                pitch=float(activation_transform.rotation.pitch),
                yaw=float(activation_transform.rotation.yaw),
                roll=float(activation_transform.rotation.roll),
            ),
        )
        if physical_staging:
            actor.apply_control(carla.WalkerControl(speed=0.0))
        else:
            actor.set_simulate_physics(False)
            actor.set_transform(hidden_transform)
        self.registry.add(actor)
        self.bindings[role_name] = actor
        self.spawn_diagnostics[role_name] = {
            "attempts": attempts,
            "source": source,
            "staging": "physical_waiting" if physical_staging else "hidden_physics_disabled",
            "collision_survival": "invincible_actor",
            "reactivation": "start_walking_in_place" if physical_staging else "two_phase_physics_then_transform",
            "activation_transform_source": "spawn_candidate",
            "configured_walker_speed_mps": float(speed_mps),
        }
        self.reserved_locations.append(activation_location)
        return ScriptedWalker(
            actor,
            selected_target,
            speed_mps,
            activation_transform=None if physical_staging else activation_transform,
            pause_fraction=pause_fraction,
            pause_ticks=pause_ticks,
        )

    def spawn(self) -> None:
        if self._spawned:
            return
        import carla

        by_kind = {
            str(event["kind"]): event
            for event in self.events
        }
        slow_config = by_kind["slow_vehicle"]
        slow_waypoint = self._waypoint(
            float(slow_config["anchor_progress_m"])
        )
        self.slow_vehicle = self._spawn_vehicle(
            (
                "vehicle.audi.a2",
                "vehicle.nissan.micra",
                "vehicle.mercedes.coupe_2020",
            ),
            slow_waypoint,
            "scene2_slow_vehicle",
            hidden_staging=True,
        )
        self.slow_vehicle.set_autopilot(
            False,
            self.traffic_manager.get_port(),
        )

        for event in self.events:
            if event['kind'] == 'route_slow_vehicle':
                self._spawn_route_slow_vehicle(event)

        crossing_config = by_kind["crossing_pedestrian"]
        crossing_waypoint = self._waypoint(
            float(crossing_config["anchor_progress_m"])
        )
        crosswalk_polygon_index = crossing_config.get(
            "crosswalk_polygon_index"
        )
        if crosswalk_polygon_index is None:
            start, target = crossing_endpoints(crossing_waypoint)
        else:
            start, target = crosswalk_polygon_endpoints(
                self.world.get_map(),
                int(crosswalk_polygon_index),
                clearance_m=float(crossing_config.get('endpoint_clearance_m',0.0)),
            )
        if bool(crossing_config.get("reverse_direction", False)):
            start, target = target, start
        self.scripted_walkers["crosswalk_pedestrian"] = (
            self._spawn_walker(
                start,
                target,
                "scene2_crosswalk_pedestrian",
                float(crossing_config["walker_speed_mps"]),
                pause_fraction=crossing_config.get("pause_fraction"),
                pause_ticks=int(crossing_config.get("pause_ticks", 0)),
                physical_staging=bool(crossing_config.get('physical_staging',False)),
            )
        )

        self.spawn_diagnostics[
            "scene2_crosswalk_pedestrian"
        ].update(
            {
                "geometry_source": "official_crosswalk_polygon",
                "crosswalk_polygon_index": int(
                    crossing_config["crosswalk_polygon_index"]
                ),
                "crosswalk_route_progress_m": float(
                    crossing_config["anchor_progress_m"]
                ),
                "crosswalk_length_m": distance_2d(start, target),
            }
        )
        # Event completion is separate from the oracle's footprint-clear test.
        self.scripted_walkers[
            "crosswalk_pedestrian"
        ].completion_distance_m = float(
            crossing_config.get("completion_distance_m", 0.8)
        )
        self.spawn_diagnostics[
            "scene2_crosswalk_pedestrian"
        ]["completion_distance_m"] = float(
            crossing_config.get("completion_distance_m", 0.8)
        )

        bus_config = by_kind["bus_stop"]
        bus_route_waypoint = self._waypoint(
            float(bus_config["anchor_progress_m"])
        )
        bus_waypoint = roadside_bus_waypoint(bus_route_waypoint, carla)
        self.bus = self._spawn_vehicle(
            (
                "vehicle.mitsubishi.fusorosa",
                "vehicle.volkswagen.t2_2021",
            ),
            bus_waypoint,
            "scene2_bus_stop_bus",
            hidden_staging=False,
        )
        self.bus_transform = self._vehicle_spawn_transforms['scene2_bus_stop_bus']
        self.bus.set_autopilot(False,self.traffic_manager.get_port())
        self.bus.apply_control(
            carla.VehicleControl(hand_brake=True)
        )
        try:
            self.bus.set_light_state(
                carla.VehicleLightState(
                    carla.VehicleLightState.Position
                    | carla.VehicleLightState.LowBeam
                    | carla.VehicleLightState.Brake
                    | carla.VehicleLightState.RightBlinker
                )
            )
        except RuntimeError:
            pass
        sidewalk = (
            _lane_sidewalk(bus_waypoint, "right")
            or _lane_sidewalk(bus_waypoint, "left")
        )
        sidewalk_side = (
            "right"
            if _lane_sidewalk(bus_waypoint, "right") is not None
            else "left"
        )
        base = (
            sidewalk.transform.location
            if sidewalk is not None
            else crossing_endpoints(bus_waypoint)[1]
        )
        forward = bus_waypoint.transform.get_forward_vector()
        passenger_speed = float(
            bus_config["passenger_speed_mps"]
        )
        passenger_layout = bus_config.get('passengers', [
            dict(role_name=f'scene2_bus_passenger_{i+1}',longitudinal_offset_m=offset,lateral_walk_m=1.5)
            for i,offset in enumerate((-5.0,5.0,8.0))])
        roles = [p['role_name'] for p in passenger_layout]
        if not roles or len(set(roles)) != len(roles) or any(not r.startswith('scene2_bus_passenger_') for r in roles):
            raise ValueError('bus passengers require unique scene2_bus_passenger_ roles')
        for passenger in passenger_layout:
            longitudinal = float(passenger['longitudinal_offset_m'])
            lateral = float(passenger['lateral_walk_m'])
            if not math.isfinite(longitudinal) or not math.isfinite(lateral) or lateral == 0:
                raise ValueError('invalid passenger movement offsets')
            passenger_road = self._waypoint(
                float(bus_config["anchor_progress_m"]) + longitudinal
            )
            passenger_sidewalk = (
                _lane_sidewalk(passenger_road, sidewalk_side)
                or _lane_sidewalk(
                    passenger_road,
                    "left" if sidewalk_side == "right" else "right",
                )
            )
            if passenger_sidewalk is not None:
                start = passenger_sidewalk.transform.location
            else:
                start = carla.Location(
                    x=base.x + forward.x * longitudinal,
                    y=base.y + forward.y * longitudinal,
                    z=base.z,
                )
            passenger_right = passenger_road.transform.get_right_vector()
            # This Town05 sidewalk terminates at collision geometry roughly
            # 1.2 m from the staged walkers.  Keep every passenger on the
            # locally traversable side and use a target that can be reached
            # before that curb.  The previous third-passenger direction
            # pointed into the collision volume and could never complete.
            target = carla.Location(
                x=start.x + passenger_right.x * lateral,
                y=start.y + passenger_right.y * lateral,
                z=start.z,
            )
            key = passenger['role_name'].removeprefix('scene2_')
            self.scripted_walkers[key] = self._spawn_walker(
                start,
                target,
                passenger['role_name'],
                passenger_speed,
                physical_staging=True,
            )

        prop_blueprint = self.world.get_blueprint_library().find(
            "static.prop.busstop"
        )
        prop_transform = carla.Transform(
            carla.Location(
                x=base.x + forward.x * 10.0,
                y=base.y + forward.y * 10.0,
                z=base.z,
            ),
            bus_waypoint.transform.rotation,
        )
        self.registry.add(
            self.world.try_spawn_actor(
                prop_blueprint,
                prop_transform,
            )
        )

        cyclist_config = by_kind["cyclist"]
        self.spawn_diagnostics["scene2_slow_cyclist"] = {
            "source": "deferred_event_staging",
            "staging_progress_m": self._staging_progress(cyclist_config),
            "spawned": False,
        }
        self._spawned = True

    @staticmethod
    def _staging_progress(event):
        """Event setup only: future-lap actors must not occupy earlier laps."""
        activate = float(event.get("activate_progress_m", float(event["anchor_progress_m"]) - 80.))
        lead = float(event.get("staging_lead_m", 200.))
        if not math.isfinite(activate) or not math.isfinite(lead) or lead < 0:
            raise ValueError("finite event activation and nonnegative staging lead required")
        return max(0., activate - lead)

    def _stage_cyclist(self, cyclist_config, progress_m):
        import carla
        if self.cyclist is not None:
            return
        cyclist_waypoint = self._waypoint(
            float(cyclist_config["anchor_progress_m"])
        )
        self.cyclist = self._spawn_vehicle(
            (
                "vehicle.bh.crossbike",
                "vehicle.gazelle.omafiets",
                "vehicle.diamondback.century",
            ),
            cyclist_waypoint,
            "scene2_slow_cyclist",
            hidden_staging=True,
        )
        self.cyclist.set_autopilot(False,self.traffic_manager.get_port())
        self.spawn_diagnostics["scene2_slow_cyclist"].update(
            {
                "spawned": True,
                "staged_at_progress_m": float(progress_m),
                "staging_progress_m": self._staging_progress(cyclist_config),
                "activation_source": "offscreen_prestage_then_parked_release",
            }
        )

    def skip_events_before(self, start_progress_m: float) -> list[str]:
        """Cold segments must not activate actors belonging to earlier tasks."""

        skipped = []
        for event in self.events:
            event_id = str(event["id"])
            if float(event["anchor_progress_m"]) >= start_progress_m - 300.0:
                continue
            if self.states[event_id] == "STAGED":
                self.states[event_id] = "SKIPPED"
                self._retire_event_actors(event_id, force=True)
                skipped.append(event_id)
        return skipped

    def _prestage_cyclist(self, event: Mapping[str, Any], progress_m: float) -> None:
        """Place the parked cyclist only after the earlier route overlap is passed."""
        if self._cyclist_placed or self.cyclist is None:
            return
        if progress_m < float(event["activate_progress_m"]) - 450.0:
            return
        import carla

        if self.ego is None:
            raise RuntimeError("cyclist prestaging requires the ego actor")
        origin = self.ego.get_location()
        for offset_m in (0.0, 12.0, -12.0, 24.0, -24.0):
            waypoint = self._waypoint(float(event["anchor_progress_m"]) + offset_m)
            location = waypoint.transform.location
            if distance_2d(origin, location) < 200.0:
                continue
            if any(
                actor.id != self.cyclist.id and actor.type_id.startswith("vehicle.")
                and distance_2d(actor.get_location(), location) < 9.0
                for actor in self.world.get_actors().filter("vehicle.*")
            ):
                continue
            transform = carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 0.45),
                waypoint.transform.rotation,
            )
            self.cyclist.set_transform(transform)
            self.cyclist.set_simulate_physics(True)
            self.cyclist.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            self.cyclist_transform = transform
            self._cyclist_placed = True
            self.reserved_locations.append(transform.location)
            self.spawn_diagnostics["scene2_slow_cyclist"].update({
                "prestage_progress_m": round(progress_m, 3),
                "prestage_distance_from_ego_m": round(distance_2d(origin, location), 3),
                "activation_location": {
                    "x": round(location.x, 3), "y": round(location.y, 3),
                    "z": round(transform.location.z, 3),
                },
            })
            return
        if progress_m >= float(event["activate_progress_m"]) - 180.0:
            raise RuntimeError("cyclist could not be staged before entering ego view")

    def update(self, progress_m: float) -> list[dict[str, Any]]:
        changes = []
        slow_event = next((event for event in self.events if event["kind"] == "slow_vehicle"), None)
        if slow_event is not None and self.states[str(slow_event["id"])] == "STAGED":
            self._prestage_slow_vehicle(slow_event, progress_m)
        cyclist_event = next((event for event in self.events if event["kind"] == "cyclist"), None)
        if cyclist_event is not None and self.states[str(cyclist_event["id"])] == "STAGED":
            self._prestage_cyclist(cyclist_event, progress_m)
        for event in self.events:
            event_id = str(event["id"])
            state = self.states[event_id]
            activate_at = float(
                event.get(
                    "activate_progress_m",
                    event.get("anchor_progress_m", 0.0) - 80.0,
                )
            )
            if (event["kind"] == "cyclist" and state == "STAGED"
                    and progress_m >= self._staging_progress(event)):
                self._stage_cyclist(event, progress_m)
            if state == "STAGED" and progress_m >= activate_at:
                self.states[event_id] = "ACTIVE"
                changes.append(
                    {
                        "event_id": event_id,
                        "state": "ACTIVE",
                        "progress_m": progress_m,
                        "variant_id": self.selected_variants[event_id],
                    }
                )
                if event["kind"] == "slow_vehicle":
                    self._activate_slow_vehicle(event, progress_m)
                elif event['kind'] == 'route_slow_vehicle':
                    self._activate_route_slow_vehicle(event)
                elif event["kind"] == "crossing_pedestrian":
                    self.scripted_walkers[
                        "crosswalk_pedestrian"
                    ].start()
                elif event["kind"] == "bus_stop":
                    self._activate_bus()
                    for key, walker in self.scripted_walkers.items():
                        if key.startswith("bus_passenger_"):
                            walker.start()
                elif event["kind"] == "cyclist":
                    import carla
                    if self.cyclist is None or not self.cyclist.is_alive or not self._cyclist_placed:
                        raise RuntimeError('staged cyclist disappeared before activation')
                    if (
                        self.cyclist is not None
                        and self.cyclist_transform is not None
                    ):
                        self.cyclist.apply_control(carla.VehicleControl())
                        self.cyclist.set_autopilot(
                            True,
                            self.traffic_manager.get_port(),
                        )
                        self._set_desired_speed(
                            self.cyclist,
                            float(event["target_speed_kmh"]),
                            75.0,
                        )
                        self.traffic_manager.auto_lane_change(
                            self.cyclist,
                            False,
                        )

        for walker in self.scripted_walkers.values():
            walker.update()

        for event in self.events:
            event_id = str(event["id"])
            resolve_at = event.get("resolve_progress_m")
            if event['kind'] == 'route_slow_vehicle':
                resolve_at = event['complete_progress_m']
            if (
                resolve_at is not None
                and self.states[event_id] == "ACTIVE"
                and progress_m >= float(resolve_at)
            ):
                self.states[event_id] = "RESOLVED"
                changes.append(
                    {
                        "event_id": event_id,
                        "state": "RESOLVED",
                        "progress_m": progress_m,
                        "variant_id": self.selected_variants[event_id],
                    }
                )
                if event['kind'] != 'route_slow_vehicle':
                    self._retire_event_actors(event_id)

        crossing = self.scripted_walkers.get(
            "crosswalk_pedestrian"
        )
        if (
            crossing is not None
            and crossing.completed
            and self.states["crosswalk_pedestrian"] == "ACTIVE"
        ):
            self.states["crosswalk_pedestrian"] = "RESOLVED"
            changes.append(
                {
                    "event_id": "crosswalk_pedestrian",
                    "state": "RESOLVED",
                    "progress_m": progress_m,
                    "variant_id": self.selected_variants[
                        "crosswalk_pedestrian"
                    ],
                }
            )
            self._retire_event_actors("crosswalk_pedestrian")
        bus_walkers = [
            walker
            for key, walker in self.scripted_walkers.items()
            if key.startswith("bus_passenger_")
        ]
        if self.states["bus_stop_passengers"] == "ACTIVE":
            self.bus_active_ticks += 1
        if (
            bus_walkers
            and all(walker.completed for walker in bus_walkers)
            and self.bus_active_ticks >= 80
            and self.states["bus_stop_passengers"] == "ACTIVE"
        ):
            self.states["bus_stop_passengers"] = "RESOLVED"
            changes.append(
                {
                    "event_id": "bus_stop_passengers",
                    "state": "RESOLVED",
                    "progress_m": progress_m,
                    "variant_id": self.selected_variants[
                        "bus_stop_passengers"
                    ],
                }
            )
            self._retire_event_actors("bus_stop_passengers")
        return changes

    def summary(self) -> dict[str, int]:
        result = {"STAGED": 0, "ACTIVE": 0, "RESOLVED": 0, "SKIPPED": 0}
        for state in self.states.values():
            result[state] += 1
        return result

    def ground_truth_actor_bindings(self) -> dict[str, Any]:
        return dict(self.bindings)

    def ground_truth_runtime_state(self) -> dict[str, Any]:
        return {
            "selected_variants": dict(self.selected_variants),
            "spawn_diagnostics": dict(self.spawn_diagnostics),
        }
