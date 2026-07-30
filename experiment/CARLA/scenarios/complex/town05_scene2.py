"""Town05 route, traffic flow, and deterministic actors for Scene 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Mapping, Sequence


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
        upper = min(len(self.route), self.index + self.search_ahead)
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

        def allowed(transform: Any) -> bool:
            location = transform.location
            if distance_2d(location, ego_location) < 18.0:
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
                self.rng.uniform(2.8, 4.2),
            )
            self.traffic_manager.vehicle_percentage_speed_difference(
                actor,
                self.rng.uniform(-3.0, 14.0),
            )
            self.traffic_manager.auto_lane_change(actor, True)
            lane_change = float(
                self.config["random_lane_change_percentage"]
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
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            transform = carla.Transform(location)
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


class ScriptedWalker:
    def __init__(
        self,
        actor: Any,
        target: Any,
        speed_mps: float,
    ) -> None:
        self.actor = actor
        self.target = target
        self.speed_mps = float(speed_mps)
        self.active = False
        self.completed = False

    def start(self) -> None:
        self.active = True

    def update(self) -> None:
        if not self.active or self.completed or not self.actor.is_alive:
            return
        import carla

        location = self.actor.get_location()
        dx = self.target.x - location.x
        dy = self.target.y - location.y
        distance = math.hypot(dx, dy)
        if distance <= 0.8:
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
    ) -> None:
        self.world = world
        self.traffic_manager = traffic_manager
        self.registry = registry
        self.route = route
        self.distances = distances
        self.events = list(events)
        self.rng = random.Random(int(seed) + 17)
        self.states = {
            str(event["id"]): "STAGED"
            for event in self.events
        }
        self.reserved_locations: list[Any] = []
        self.scripted_walkers: dict[str, ScriptedWalker] = {}
        self.slow_vehicle: Any | None = None
        self.cyclist: Any | None = None
        self.cyclist_transform: Any | None = None
        self.bus: Any | None = None
        self.bus_active_ticks = 0
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

    def _waypoint(self, progress_m: float) -> Any:
        return self.route[
            route_index_at(self.distances, progress_m)
        ][0]

    def _spawn_vehicle(
        self,
        blueprint_ids: Sequence[str],
        waypoint: Any,
        role_name: str,
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
        transform = waypoint.transform
        transform.location.z += 0.35
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError("failed to stage {0}".format(role_name))
        self.registry.add(actor)
        self.reserved_locations.append(actor.get_location())
        return actor

    def _spawn_walker(
        self,
        start: Any,
        target: Any,
        role_name: str,
        speed_mps: float,
    ) -> ScriptedWalker:
        import carla

        library = self.world.get_blueprint_library()
        blueprint = self.rng.choice(
            list(library.filter("walker.pedestrian.*"))
        )
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        transform = carla.Transform(
            carla.Location(
                x=start.x,
                y=start.y,
                z=start.z + 0.35,
            )
        )
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError("failed to stage {0}".format(role_name))
        self.registry.add(actor)
        self.reserved_locations.append(actor.get_location())
        return ScriptedWalker(actor, target, speed_mps)

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
        )
        self.slow_vehicle.set_autopilot(
            True,
            self.traffic_manager.get_port(),
        )
        self._set_desired_speed(
            self.slow_vehicle,
            float(slow_config["target_speed_kmh"]),
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

        crossing_config = by_kind["crossing_pedestrian"]
        crossing_waypoint = self._waypoint(
            float(crossing_config["anchor_progress_m"])
        )
        start, target = crossing_endpoints(crossing_waypoint)
        self.scripted_walkers["crosswalk_pedestrian"] = (
            self._spawn_walker(
                start,
                target,
                "scene2_crosswalk_pedestrian",
                float(crossing_config["walker_speed_mps"]),
            )
        )

        bus_config = by_kind["bus_stop"]
        bus_waypoint = self._waypoint(
            float(bus_config["anchor_progress_m"])
        )
        self.bus = self._spawn_vehicle(
            (
                "vehicle.mitsubishi.fusorosa",
                "vehicle.volkswagen.t2_2021",
            ),
            bus_waypoint,
            "scene2_bus_stop_bus",
        )
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
        base = (
            sidewalk.transform.location
            if sidewalk is not None
            else crossing_endpoints(bus_waypoint)[1]
        )
        forward = bus_waypoint.transform.get_forward_vector()
        right_vector = bus_waypoint.transform.get_right_vector()
        passenger_speed = float(
            bus_config["passenger_speed_mps"]
        )
        for index, longitudinal in enumerate((-5.0, 5.0, 8.0)):
            start = carla.Location(
                x=base.x + forward.x * longitudinal,
                y=base.y + forward.y * longitudinal,
                z=base.z,
            )
            toward_bus = index == 2
            lateral = -2.0 if toward_bus else 2.0
            target = carla.Location(
                x=start.x + right_vector.x * lateral,
                y=start.y + right_vector.y * lateral,
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
        self.cyclist_transform = self.cyclist.get_transform()
        hidden = self.cyclist.get_transform()
        hidden.location.z = -20.0
        self.cyclist.set_simulate_physics(False)
        self.cyclist.set_transform(hidden)
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
                    }
                )
                if event["kind"] == "crossing_pedestrian":
                    self.scripted_walkers[
                        "crosswalk_pedestrian"
                    ].start()
                elif event["kind"] == "bus_stop":
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
                }
            )
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
                }
            )
        return changes

    def summary(self) -> dict[str, int]:
        result = {"STAGED": 0, "ACTIVE": 0, "RESOLVED": 0}
        for state in self.states.values():
            result[state] += 1
        return result
