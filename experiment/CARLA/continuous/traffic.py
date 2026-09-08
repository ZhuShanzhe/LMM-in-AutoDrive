"""Traffic Manager-backed route following and reproducible background traffic."""

import random

import carla

from control.protocol import normalize_intent


def route_point_at(route_manager, distance_m):
    """Return the first route point at or after a global route distance."""
    for point in route_manager.route[route_manager.current_index:]:
        if point["distance_m"] >= float(distance_m):
            return point
    return route_manager.route[-1]


def find_sidewalk_waypoint(road_waypoint, side):
    """Find a pedestrian-capable roadside lane from a driving lane."""
    getter_name = "get_right_lane" if str(side).lower() == "right" else "get_left_lane"
    candidate = road_waypoint
    for _ in range(12):
        candidate = getattr(candidate, getter_name)()
        if candidate is None:
            return None
        if candidate.lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder):
            return candidate
    return None


class TrafficManagerRouteController:
    """Leave low-level control to CARLA Traffic Manager along a fixed route."""

    def __init__(self, vehicle, traffic_manager, route_manager, target_speed_kmh,
                 horizon_m=400.0, refresh_margin_m=100.0):
        self.vehicle = vehicle
        self.traffic_manager = traffic_manager
        self.target_speed_kmh = float(target_speed_kmh)
        self.route_manager = route_manager
        self.route = self.route_manager.route
        self.horizon_steps = max(2, int(float(horizon_m) / self._route_step_m()))
        self.refresh_margin_steps = max(1, int(float(refresh_margin_m) / self._route_step_m()))
        self._last_route_index = -1
        self._configure()

    def run_step(self, intent, dt):
        del dt
        normalized = normalize_intent(intent, self.target_speed_kmh)
        if normalized["emergency"] or normalized["action"] == "emergency_brake":
            self.vehicle.set_autopilot(False)
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0), normalized
        if self.route_manager.current_index >= self._last_route_index - self.refresh_margin_steps:
            self._install_path()
        return None, normalized

    def _configure(self):
        if not self.route:
            raise ValueError("Traffic Manager route cannot be empty")
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self._install_path()
        if hasattr(self.traffic_manager, "set_desired_speed"):
            self.traffic_manager.set_desired_speed(self.vehicle, self.target_speed_kmh)
        else:
            speed_limit_kmh = max(float(self.vehicle.get_speed_limit()), 1.0)
            speed_difference_pct = 100.0 * (
                speed_limit_kmh - self.target_speed_kmh
            ) / speed_limit_kmh
            self.traffic_manager.vehicle_percentage_speed_difference(
                self.vehicle,
                max(-80.0, min(80.0, speed_difference_pct)),
            )
        self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 8.0)

    def _install_path(self):
        start = max(0, self.route_manager.current_index)
        end = min(len(self.route), start + self.horizon_steps)
        path = [
            carla.Location(x=item["x"], y=item["y"], z=item.get("z", 0.0))
            for item in self.route[start:end:4]
        ]
        if len(path) < 2:
            path = [
                carla.Location(x=item["x"], y=item["y"], z=item.get("z", 0.0))
                for item in self.route[start:]
            ]
        if len(path) >= 2:
            self.traffic_manager.set_path(self.vehicle, path)
        self._last_route_index = end - 1

    def _route_step_m(self):
        if len(self.route) < 2:
            return 5.0
        return max(0.1, self.route[1]["distance_m"] - self.route[0]["distance_m"])


class TrafficFlowManager:
    """Spawn traffic actors with a repeatable seed and expose run metadata."""

    def __init__(self, world, config):
        self.world = world
        self.config = dict(config or {})
        self.port = int(self.config.get("traffic_manager_port", 8000))
        self.client = None
        self.traffic_manager = None
        self.actors = []
        self.seed = int(self.config.get("seed", 20260724))
        self.deactivated_at_m = None
        self._last_replenish_progress_m = 0.0
        self._route_spawn_sequence = 0

    def start(self, client):
        self.client = client
        self.traffic_manager = client.get_trafficmanager(self.port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed)
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            float(self.config.get("following_distance_m", 6.0))
        )
        self.traffic_manager.global_percentage_speed_difference(
            float(self.config.get("speed_difference_pct", 8.0))
        )
        return self.traffic_manager

    def spawn_background(self, ego_vehicle, route_manager=None):
        if not self.config.get("enabled", False):
            return []
        if self.traffic_manager is None:
            raise RuntimeError("Traffic Manager must be started before spawning traffic")
        count = int(self.config.get("background_count", 0))
        if route_manager is not None:
            self._spawn_route_traffic(route_manager)
            self._last_replenish_progress_m = route_manager.progress_m
        random_count = int(self.config.get(
            "random_background_count", max(0, count - len(self.actors))
        ))
        if random_count <= 0:
            return list(self.actors)
        min_distance = float(self.config.get("min_distance_from_ego_m", 55.0))
        ego_location = ego_vehicle.get_location()
        spawn_points = list(self.world.get_map().get_spawn_points())
        random.Random(self.seed).shuffle(spawn_points)
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawned_random = 0
        for index, transform in enumerate(spawn_points):
            if len(self.actors) >= count or spawned_random >= random_count:
                break
            if transform.location.distance(ego_location) < min_distance:
                continue
            blueprint = blueprints[index % len(blueprints)]
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "background_traffic")
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", colors[index % len(colors)])
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            self._start_actor(actor)
            self.actors.append(actor)
            spawned_random += 1
        return list(self.actors)

    def tick(self, route_manager, ego_vehicle=None):
        self._despawn_behind_ego(ego_vehicle)
        interval_m = float(self.config.get("route_replenish_every_m", 0.0))
        if interval_m <= 0.0 or route_manager.progress_m - self._last_replenish_progress_m < interval_m:
            return
        offsets = list(self.config.get("route_replenish_offsets_m", []))
        if not offsets:
            return
        active_count = sum(1 for actor in self.actors if actor is not None and actor.is_alive)
        available = max(
            0,
            int(self.config.get("max_active_count", active_count + len(offsets)))
            - active_count,
        )
        if available <= 0:
            self._last_replenish_progress_m = route_manager.progress_m
            return
        self._spawn_route_traffic(
            route_manager,
            offsets=offsets,
            maximum=min(
                available,
                int(self.config.get("route_replenish_batch", len(offsets))),
            ),
        )
        self._last_replenish_progress_m = route_manager.progress_m

    def _spawn_route_traffic(self, route_manager, offsets=None, maximum=None):
        offsets = list(offsets or self.config.get("route_spawn_offsets_m", []))
        lane_offsets = list(self.config.get("route_lane_offsets", []))
        if not offsets:
            return 0
        world_map = self.world.get_map()
        blueprints = self.world.get_blueprint_library()
        vehicle_types = list(self.config.get("route_vehicle_types", [
            "vehicle.audi.tt",
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.volkswagen.t2",
        ]))
        target_speed = float(self.config.get("route_vehicle_speed_kmh", 38.0))
        spawned = 0
        for offset_m in offsets:
            if maximum is not None and spawned >= maximum:
                break
            index = self._route_spawn_sequence
            self._route_spawn_sequence += 1
            point = route_point_at(
                route_manager, route_manager.progress_m + float(offset_m)
            )
            waypoint = world_map.get_waypoint(
                carla.Location(x=point["x"], y=point["y"], z=point["z"]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                continue
            lane_offset = int(lane_offsets[index % len(lane_offsets)]) if lane_offsets else 0
            waypoint = self._offset_lane(waypoint, lane_offset, index)
            if waypoint is None:
                continue
            vehicle_type = vehicle_types[index % len(vehicle_types)]
            blueprint = blueprints.find(vehicle_type)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "route_traffic")
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", colors[index % len(colors)])
            transform = carla.Transform(
                carla.Location(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    z=waypoint.transform.location.z + 0.25,
                ),
                waypoint.transform.rotation,
            )
            if not self._has_spawn_clearance(transform.location):
                continue
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            self._start_actor(actor, target_speed + (index % 3 - 1) * 4.0)
            self.actors.append(actor)
            spawned += 1
        return spawned

    def _has_spawn_clearance(self, location):
        """Avoid spawning a route vehicle into an occupied traffic gap."""
        clearance_m = float(self.config.get("spawn_clearance_m", 16.0))
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor is None or not actor.is_alive:
                continue
            if actor.get_location().distance(location) < clearance_m:
                return False
        return True

    def _despawn_behind_ego(self, ego_vehicle):
        """Recycle passed background vehicles so replenishment stays visually stable."""
        if ego_vehicle is None or not ego_vehicle.is_alive:
            return
        behind_m = float(self.config.get("despawn_behind_m", 120.0))
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        kept = []
        for actor in self.actors:
            if actor is None or not actor.is_alive:
                continue
            delta_x = actor.get_location().x - ego_location.x
            delta_y = actor.get_location().y - ego_location.y
            longitudinal_m = delta_x * forward.x + delta_y * forward.y
            if longitudinal_m < -behind_m:
                actor.destroy()
                continue
            kept.append(actor)
        self.actors = kept

    def _start_actor(self, actor, desired_speed_kmh=None):
        actor.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.auto_lane_change(
            actor, bool(self.config.get("auto_lane_change", False))
        )
        if bool(self.config.get("ignore_traffic_lights", False)):
            self.traffic_manager.ignore_lights_percentage(actor, 100.0)
        if bool(self.config.get("ignore_traffic_signs", False)):
            self.traffic_manager.ignore_signs_percentage(actor, 100.0)
        if desired_speed_kmh is not None and hasattr(self.traffic_manager, "set_desired_speed"):
            self.traffic_manager.set_desired_speed(actor, max(1.0, float(desired_speed_kmh)))

    @staticmethod
    def _offset_lane(waypoint, lane_offset, fallback_index=0):
        if lane_offset == 0:
            return waypoint
        base_waypoint = waypoint
        getter_name = "get_left_lane" if lane_offset < 0 else "get_right_lane"
        for _ in range(abs(lane_offset)):
            candidate = getattr(waypoint, getter_name)()
            if (
                candidate is None
                or candidate.lane_type != carla.LaneType.Driving
                or candidate.road_id != waypoint.road_id
                or candidate.lane_id * waypoint.lane_id <= 0
            ):
                break
            waypoint = candidate
        if waypoint is not base_waypoint:
            return waypoint
        alternatives = []
        for alternate_getter in ("get_left_lane", "get_right_lane"):
            candidate = getattr(base_waypoint, alternate_getter)()
            while (
                candidate is not None
                and candidate.lane_type == carla.LaneType.Driving
                and candidate.road_id == base_waypoint.road_id
                and candidate.lane_id * base_waypoint.lane_id > 0
            ):
                alternatives.append(candidate)
                candidate = getattr(candidate, alternate_getter)()
        if not alternatives:
            return None
        return alternatives[fallback_index % len(alternatives)]

    def snapshot(self):
        return {
            "enabled": bool(self.config.get("enabled", False)),
            "seed": self.seed,
            "traffic_manager_port": self.port,
            "background_actor_ids": [actor.id for actor in self.actors if actor.is_alive],
            "background_actor_count": sum(1 for actor in self.actors if actor.is_alive),
            "deactivated_at_m": self.deactivated_at_m,
        }

    def deactivate_background(self, route_progress_m):
        active_count = sum(1 for actor in self.actors if actor is not None and actor.is_alive)
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors = []
        self.deactivated_at_m = round(float(route_progress_m), 3)
        return active_count

    def destroy(self):
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors = []

    def restore_runtime(self):
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)


class PedestrianFlowManager:
    """Deterministic sidewalk walkers visible along the ego route."""

    def __init__(self, world, config):
        self.world = world
        self.config = dict(config or {})
        self.walkers = []
        self.seed = int(self.config.get("seed", 20260724))
        self._last_replenish_progress_m = 0.0
        self._spawn_sequence = 0
        self.motion_mode = str(self.config.get("motion_mode", "kinematic")).lower()

    def spawn(self, route_manager):
        if not self.config.get("enabled", False):
            return []
        self._spawn_offsets(
            route_manager, list(self.config.get("route_spawn_offsets_m", []))
        )
        self._last_replenish_progress_m = route_manager.progress_m
        return [item["actor"] for item in self.walkers]

    def _spawn_offsets(self, route_manager, offsets, maximum=None):
        sides = list(self.config.get("sides", ["right"]))
        blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        if not blueprints:
            return []
        world_map = self.world.get_map()
        spawned = 0
        for offset_m in offsets:
            if maximum is not None and spawned >= maximum:
                break
            index = self._spawn_sequence
            self._spawn_sequence += 1
            point = route_point_at(
                route_manager, route_manager.progress_m + float(offset_m)
            )
            road_waypoint = world_map.get_waypoint(
                carla.Location(x=point["x"], y=point["y"], z=point["z"]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if road_waypoint is None:
                continue
            side = str(sides[index % len(sides)]).lower()
            sidewalk = find_sidewalk_waypoint(road_waypoint, side)
            if sidewalk is None:
                sidewalk_transform = self._roadside_transform(road_waypoint, side)
            else:
                sidewalk_transform = sidewalk.transform
            blueprint = blueprints[(self.seed + index) % len(blueprints)]
            transform = carla.Transform(
                carla.Location(
                    x=sidewalk_transform.location.x,
                    y=sidewalk_transform.location.y,
                    z=sidewalk_transform.location.z + 0.35,
                ),
                sidewalk_transform.rotation,
            )
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            forward = road_waypoint.transform.get_forward_vector()
            direction = carla.Vector3D(
                x=forward.x * (-1.0 if index % 2 else 1.0),
                y=forward.y * (-1.0 if index % 2 else 1.0),
                z=0.0,
            )
            self.walkers.append({
                "actor": actor,
                "direction": direction,
                "speed": float(self.config.get("speed_mps", 1.2)) + 0.1 * (index % 3),
                "start_location": carla.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=transform.location.z,
                ),
                "rotation": transform.rotation,
                "travelled_m": 0.0,
            })
            spawned += 1
        return spawned

    def tick(self, route_manager, fixed_delta_s=0.1, ego_vehicle=None):
        self._despawn_behind_ego(ego_vehicle)
        for item in self.walkers:
            actor = item["actor"]
            if actor is not None and actor.is_alive:
                actor.apply_control(carla.WalkerControl(
                    direction=item["direction"], speed=item["speed"]
                ))
                if self.motion_mode == "kinematic":
                    item["travelled_m"] += item["speed"] * float(fixed_delta_s)
                    maximum = float(self.config.get("walk_distance_m", 40.0))
                    if item["travelled_m"] > maximum:
                        item["start_location"] = actor.get_location()
                        item["direction"] = carla.Vector3D(
                            x=-item["direction"].x,
                            y=-item["direction"].y,
                            z=0.0,
                        )
                        item["travelled_m"] = 0.0
                    actor.set_transform(carla.Transform(
                        carla.Location(
                            x=item["start_location"].x + item["direction"].x * item["travelled_m"],
                            y=item["start_location"].y + item["direction"].y * item["travelled_m"],
                            z=item["start_location"].z,
                        ),
                        item["rotation"],
                    ))
        interval_m = float(self.config.get("route_replenish_every_m", 0.0))
        if interval_m <= 0.0 or route_manager.progress_m - self._last_replenish_progress_m < interval_m:
            return
        offsets = list(self.config.get("route_replenish_offsets_m", []))
        if not offsets:
            return
        active_count = sum(
            1 for item in self.walkers
            if item["actor"] is not None and item["actor"].is_alive
        )
        available = max(0, int(self.config.get("max_walker_count", active_count + len(offsets))) - active_count)
        if available > 0:
            self._spawn_offsets(route_manager, offsets, available)
        self._last_replenish_progress_m = route_manager.progress_m

    def _despawn_behind_ego(self, ego_vehicle):
        if ego_vehicle is None or not ego_vehicle.is_alive:
            return
        behind_m = float(self.config.get("despawn_behind_m", 90.0))
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        kept = []
        for item in self.walkers:
            actor = item["actor"]
            if actor is None or not actor.is_alive:
                continue
            delta_x = actor.get_location().x - ego_location.x
            delta_y = actor.get_location().y - ego_location.y
            longitudinal_m = delta_x * forward.x + delta_y * forward.y
            if longitudinal_m < -behind_m:
                actor.destroy()
                continue
            kept.append(item)
        self.walkers = kept

    def snapshot(self):
        active = [item["actor"].id for item in self.walkers if item["actor"].is_alive]
        return {
            "enabled": bool(self.config.get("enabled", False)),
            "motion_mode": self.motion_mode,
            "walker_ids": active,
            "walker_count": len(active),
        }

    def destroy(self):
        for item in self.walkers:
            actor = item["actor"]
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.walkers = []

    def _roadside_transform(self, road_waypoint, side):
        forward = road_waypoint.transform.get_forward_vector()
        right = carla.Vector3D(x=forward.y, y=-forward.x, z=0.0)
        lateral = float(self.config.get("fallback_roadside_offset_m", 7.0))
        sign = 1.0 if str(side).lower() == "right" else -1.0
        location = carla.Location(
            x=road_waypoint.transform.location.x + right.x * lateral * sign,
            y=road_waypoint.transform.location.y + right.y * lateral * sign,
            z=road_waypoint.transform.location.z,
        )
        return carla.Transform(location, road_waypoint.transform.rotation)
