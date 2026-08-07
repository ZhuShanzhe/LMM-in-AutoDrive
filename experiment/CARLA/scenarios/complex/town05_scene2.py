"""Town05 route, traffic flow, and deterministic actors for Scene 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Mapping, Sequence


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

    def spawn(
        self,
        reserved_locations: Iterable[Any],
        ego_location: Any,
    ) -> None:
        self._spawn_vehicles(
            list(reserved_locations),
            ego_location,
        )
        self._spawn_walkers()

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
        requested = int(self.config["vehicles"])
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
) -> tuple[Any, Any]:
    """Return pedestrian endpoints along an official crosswalk long axis."""

    import carla

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
    ) -> None:
        self.world = world
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
        self.cyclist: Any | None = None
        self.cyclist_transform: Any | None = None
        self.bus: Any | None = None
        self.bus_transform: Any | None = None
        self.bus_active_ticks = 0
        self._retired_events: set[str] = set()
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

    def _retire_event_actors(self, event_id: str) -> int:
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
        else:
            roles = []

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

    def _free_vehicle_waypoint(
        self,
        progress_m: float,
        lead_distance_m: float,
        staged_actor: Any,
    ) -> tuple[Any, float]:
        """Choose a route point ahead that is not occupied by traffic."""

        vehicles = list(
            self.world.get_actors().filter("vehicle.*")
        )
        candidate_leads = (
            float(lead_distance_m),
            float(lead_distance_m) + 12.0,
            max(24.0, float(lead_distance_m) - 12.0),
            float(lead_distance_m) + 24.0,
        )
        for lead in candidate_leads:
            waypoint = self._waypoint(float(progress_m) + lead)
            location = waypoint.transform.location
            occupied = False
            for vehicle in vehicles:
                if int(vehicle.id) == int(staged_actor.id):
                    continue
                try:
                    distance = vehicle.get_location().distance(location)
                except RuntimeError:
                    continue
                if float(distance) < 9.0:
                    occupied = True
                    break
            if not occupied:
                return waypoint, lead
        return self._waypoint(
            float(progress_m) + float(lead_distance_m)
        ), float(lead_distance_m)

    def _activate_slow_vehicle(
        self,
        event: Mapping[str, Any],
        progress_m: float,
    ) -> None:
        """Place the slow vehicle near ego only when its event activates."""

        if self.slow_vehicle is None:
            return
        lead_distance_m = float(
            event.get("spawn_lead_distance_m", 45.0)
        )
        waypoint, selected_lead_m = self._free_vehicle_waypoint(
            progress_m,
            lead_distance_m,
            self.slow_vehicle,
        )
        transform = waypoint.transform
        transform.location.z += 0.45
        self.slow_vehicle.set_transform(transform)
        self.slow_vehicle.set_simulate_physics(True)
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
                "activation_source": "route_relative_restage",
                "activation_progress_m": round(float(progress_m), 3),
                "activation_lead_distance_m": round(
                    selected_lead_m,
                    3,
                ),
            }
        )

    def _activate_bus(self) -> None:
        """Restore the staged bus only when its event becomes active."""

        if self.bus is None or self.bus_transform is None:
            return
        import carla

        self.bus.set_transform(self.bus_transform)
        self.bus.set_simulate_physics(True)
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
                "activation_source": "captured_bus_stop_transform",
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
        actor.set_simulate_physics(False)
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
        actor.set_transform(hidden_transform)
        self.registry.add(actor)
        self.bindings[role_name] = actor
        self.spawn_diagnostics[role_name] = {
            "attempts": attempts,
            "source": source,
            "staging": "hidden_physics_disabled",
            "collision_survival": "invincible_actor",
            "reactivation": "two_phase_physics_then_transform",
            "activation_transform_source": "spawn_candidate",
            "configured_walker_speed_mps": float(speed_mps),
        }
        self.reserved_locations.append(activation_location)
        return ScriptedWalker(
            actor,
            selected_target,
            speed_mps,
            activation_transform=activation_transform,
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
        # Town05's official crosswalk polygon ends at a curb.
        # Accept completion only after the walker has cleared the
        # carriageway, while allowing the final curb collision.
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
        bus_waypoint = self._waypoint(
            float(bus_config["anchor_progress_m"])
        )
        bus_waypoint_transform = bus_waypoint.transform
        self.bus_transform = carla.Transform(
            carla.Location(
                x=float(bus_waypoint_transform.location.x),
                y=float(bus_waypoint_transform.location.y),
                z=float(bus_waypoint_transform.location.z) + 0.45,
            ),
            carla.Rotation(
                pitch=float(bus_waypoint_transform.rotation.pitch),
                yaw=float(bus_waypoint_transform.rotation.yaw),
                roll=float(bus_waypoint_transform.rotation.roll),
            ),
        )
        self.bus = self._spawn_vehicle(
            (
                "vehicle.mitsubishi.fusorosa",
                "vehicle.volkswagen.t2_2021",
            ),
            bus_waypoint,
            "scene2_bus_stop_bus",
            hidden_staging=True,
        )
        self.reserved_locations.append(self.bus_transform.location)
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
        for index, longitudinal in enumerate((-5.0, 5.0, 8.0)):
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
            lateral = 1.5
            target = carla.Location(
                x=start.x + passenger_right.x * lateral,
                y=start.y + passenger_right.y * lateral,
                z=start.z,
            )
            key = "bus_passenger_{0}".format(index + 1)
            self.scripted_walkers[key] = self._spawn_walker(
                start,
                target,
                "scene2_{0}".format(key),
                passenger_speed,
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
        )
        self.cyclist_transform = self._vehicle_spawn_transforms[
            "scene2_slow_cyclist"
        ]
        hidden = carla.Transform(
            carla.Location(
                x=float(self.cyclist_transform.location.x),
                y=float(self.cyclist_transform.location.y),
                z=-20.0,
            ),
            carla.Rotation(
                pitch=float(self.cyclist_transform.rotation.pitch),
                yaw=float(self.cyclist_transform.rotation.yaw),
                roll=float(self.cyclist_transform.rotation.roll),
            ),
        )
        self.cyclist.set_simulate_physics(False)
        self.cyclist.set_transform(hidden)
        self.spawn_diagnostics["scene2_slow_cyclist"].update(
            {
                "activation_source": "captured_spawn_transform",
                "activation_location": {
                    "x": round(
                        float(self.cyclist_transform.location.x), 3
                    ),
                    "y": round(
                        float(self.cyclist_transform.location.y), 3
                    ),
                    "z": round(
                        float(self.cyclist_transform.location.z), 3
                    ),
                },
            }
        )
        self._spawned = True

    def update(self, progress_m: float) -> list[dict[str, Any]]:
        changes = []
        for event in self.events:
            event_id = str(event["id"])
            state = self.states[event_id]
            activate_at = float(
                event.get(
                    "activate_progress_m",
                    event.get("anchor_progress_m", 0.0) - 80.0,
                )
            )
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
                    if (
                        self.cyclist is not None
                        and self.cyclist_transform is not None
                    ):
                        self.cyclist.set_transform(
                            self.cyclist_transform
                        )
                        self.cyclist.set_simulate_physics(True)
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
        result = {"STAGED": 0, "ACTIVE": 0, "RESOLVED": 0}
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
