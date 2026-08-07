"""Deterministic background traffic for the basic urban voice demonstration."""

from __future__ import annotations

import carla
import math
import random


class FixedRouteTraffic:
    """Fixed actor pool with out-of-view recycling and no runtime spawning."""

    def __init__(self, world, client, route_manager, config):
        self.world = world
        self.client = client
        self.route_manager = route_manager
        self.config = dict(config or {})
        self.port = int(self.config.get("traffic_manager_port", 8000))
        self.traffic_manager = None
        self.actors = []
        self.spawn_log = []
        self._scripted = []
        self._actor_specs = {}
        self._tick_count = 0
        self._last_recycle_tick = -100000
        self._last_actor_recycle_tick = {}
        self._visible_count = 0
        self._recycle_count = 0

    def setup(self):
        if not self.config.get("enabled", True):
            return []
        self.traffic_manager = self.client.get_trafficmanager(self.port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(self.config.get("seed", 20260728)))
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            float(self.config.get("following_distance_m", 12.0))
        )
        resume_offset_m = (
            float(self.route_manager.progress_m)
            if float(self.route_manager.progress_m) > 0.0
            else 0.0
        )
        for spec in self.config.get("vehicles", []):
            resolved_spec = dict(spec)
            if resume_offset_m > 0.0:
                resolved_spec["route_distance_m"] = min(
                    self.route_manager.route_length_m - 20.0,
                    resume_offset_m + float(spec["route_distance_m"]),
                )
            self._spawn_vehicle(resolved_spec)
        return list(self.actors)

    def _spawn_vehicle(self, spec):
        distance_m = float(spec["route_distance_m"])
        lane_from_right = int(spec.get("lane_from_right", 1))
        waypoint = self._waypoint_at(distance_m, lane_from_right)
        if waypoint is None:
            self.spawn_log.append({"id": spec["id"], "status": "skipped", "reason": "route_lane_unavailable"})
            return
        blueprint_id = str(spec.get("vehicle_type", "vehicle.audi.tt"))
        blueprint = self._resolve_vehicle_blueprint(blueprint_id, spec["id"])
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "background_" + str(spec["id"]))
        if blueprint.has_attribute("color"):
            colors = list(blueprint.get_attribute("color").recommended_values)
            if colors:
                blueprint.set_attribute("color", colors[sum(ord(char) for char in str(spec["id"])) % len(colors)])
        transform = carla.Transform(
            carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                # Match the ego clearance on generated OpenDRIVE roads.
                z=waypoint.transform.location.z + 1.0,
            ),
            waypoint.transform.rotation,
        )
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            self.spawn_log.append({"id": spec["id"], "status": "skipped", "reason": "occupied_spawn"})
            return
        actor.set_autopilot(True, self.port)
        self._configure_lane_change(actor, spec)
        if bool(spec.get("ignore_traffic_lights", self.config.get("ignore_traffic_lights", False))):
            self.traffic_manager.ignore_lights_percentage(actor, 100.0)
        if bool(spec.get("ignore_traffic_signs", self.config.get("ignore_traffic_signs", False))):
            self.traffic_manager.ignore_signs_percentage(actor, 100.0)
        self.traffic_manager.distance_to_leading_vehicle(
            actor, float(spec.get("following_distance_m", self.config.get("following_distance_m", 12.0)))
        )
        desired_speed = float(spec.get("speed_kmh", self.config.get("default_speed_kmh", 48.0)))
        speed_limit = max(float(actor.get_speed_limit()), 1.0)
        self.traffic_manager.vehicle_percentage_speed_difference(
            actor, max(-80.0, min(80.0, 100.0 * (speed_limit - desired_speed) / speed_limit))
        )
        if bool(self.config.get("traffic_manager_set_path", True)):
            route = self._traffic_manager_path(
                waypoint,
                distance_m=distance_m,
                lane_from_right=lane_from_right,
            )
            if len(route) >= 2:
                self.traffic_manager.set_path(actor, route)
        self.actors.append(actor)
        self._actor_specs[actor.id] = dict(spec)
        scripted = spec.get("scripted_slowdown")
        if isinstance(scripted, dict):
            self._scripted.append({
                "id": spec["id"], "actor": actor, "triggered": False,
                "trigger_at_progress_m": float(scripted["trigger_at_progress_m"]),
                "speed_kmh": float(scripted["speed_kmh"]),
            })
        self.spawn_log.append({
            "id": spec["id"], "status": "spawned", "actor_id": actor.id,
            "route_distance_m": distance_m, "lane_from_right": lane_from_right,
            "speed_kmh": desired_speed,
            "auto_lane_change": bool(spec.get("auto_lane_change", False)),
        })

    def _configure_lane_change(self, actor, spec):
        """Enable sparse Traffic Manager lane changes for selected NPCs."""
        enabled = bool(spec.get("auto_lane_change", False))
        self.traffic_manager.auto_lane_change(actor, enabled)
        if not enabled:
            return
        probability = float(
            spec.get(
                "lane_change_probability",
                self.config.get("lane_change_probability", 6.0),
            )
        )
        probability = max(0.0, min(20.0, probability))
        self.traffic_manager.random_left_lanechange_percentage(actor, probability)
        self.traffic_manager.random_right_lanechange_percentage(actor, probability)
        self.traffic_manager.keep_slow_lane_rule_percentage(
            actor,
            float(self.config.get("keep_slow_lane_rule_percentage", 35.0)),
        )

    def _route_point(self, distance_m):
        for point in self.route_manager.route:
            if float(point["distance_m"]) >= distance_m:
                return point
        return None

    def _waypoint_at(self, distance_m, lane_from_right):
        point = self._route_point(distance_m)
        if point is None:
            return None
        base = self.world.get_map().get_waypoint(
            carla.Location(x=point["x"], y=point["y"], z=point.get("z", 0.0)),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if base is None:
            return None
        lanes = [base]
        candidate = base.get_right_lane()
        while self._same_direction(base, candidate):
            lanes.insert(0, candidate)
            candidate = candidate.get_right_lane()
        candidate = base.get_left_lane()
        while self._same_direction(base, candidate):
            lanes.append(candidate)
            candidate = candidate.get_left_lane()
        return lanes[lane_from_right - 1] if 0 < lane_from_right <= len(lanes) else None

    def _path_from(self, start_distance_m, lane_from_right):
        points = []
        sampling_m = float(self.config.get("path_sampling_m", 25.0))
        distance_m = start_distance_m + sampling_m
        while distance_m < self.route_manager.route_length_m:
            waypoint = self._waypoint_at(distance_m, lane_from_right)
            if waypoint is not None:
                points.append(waypoint.transform.location)
            distance_m += sampling_m
        return points

    @staticmethod
    def _same_direction(reference, candidate):
        return bool(
            candidate is not None
            and candidate.lane_type == carla.LaneType.Driving
            and candidate.road_id == reference.road_id
            and candidate.lane_id * reference.lane_id > 0
        )

    def _resolve_vehicle_blueprint(self, requested_id, stable_id):
        library = self.world.get_blueprint_library()
        try:
            return library.find(requested_id)
        except RuntimeError:
            choices = sorted(library.filter("vehicle.*"), key=lambda item: item.id)
            if not choices:
                raise RuntimeError("CARLA blueprint library has no vehicle blueprints")
            return choices[sum(ord(char) for char in str(stable_id)) % len(choices)]

    def tick(self, ego_vehicle, ego_progress_m=0.0):
        """Update scripted behavior and keep a stable visible traffic buffer."""
        self._tick_count += 1
        for item in self._scripted:
            if item["triggered"] or float(ego_progress_m) < item["trigger_at_progress_m"]:
                continue
            actor = item["actor"]
            if not actor.is_alive:
                continue
            speed_limit = max(float(actor.get_speed_limit()), 1.0)
            self.traffic_manager.vehicle_percentage_speed_difference(
                actor,
                max(-80.0, min(80.0, 100.0 * (speed_limit - item["speed_kmh"]) / speed_limit)),
            )
            item["triggered"] = True
            self.spawn_log.append({
                "id": item["id"], "status": "scripted_slowdown",
                "ego_progress_m": round(float(ego_progress_m), 1),
                "speed_kmh": item["speed_kmh"],
            })
        self._maintain_visible_pool(ego_vehicle, float(ego_progress_m))

    def _maintain_visible_pool(self, ego_vehicle, ego_progress_m):
        if not bool(self.config.get("recycle_enabled", True)):
            return
        visible = self._visible_actors(ego_vehicle)
        self._visible_count = len(visible)
        if self._in_turn_clearance(ego_progress_m):
            return
        minimum = int(self.config.get("minimum_visible_vehicles", 3))
        target = max(minimum, int(self.config.get("target_visible_vehicles", 5)))
        cooldown = max(1, int(self.config.get("recycle_cooldown_ticks", 20)))
        urgent = len(visible) < minimum
        if urgent:
            cooldown = max(
                1,
                int(self.config.get("urgent_recycle_cooldown_ticks", cooldown)),
            )
        if len(visible) >= target:
            return
        if self._tick_count - self._last_recycle_tick < cooldown:
            return

        # Add at most one vehicle per maintenance cycle. Repositioning several
        # actors on one frame creates an artificial traffic cluster even when
        # every individual target is clear.
        required = 1
        protected_ids = {actor.id for actor in visible}
        candidates = [
            actor for actor in self.actors
            if actor.is_alive
            and actor.id not in protected_ids
            and self._is_safe_recycle_source(actor, ego_vehicle)
            and self._tick_count - self._last_actor_recycle_tick.get(
                actor.id, -100000
            ) >= int(self.config.get("actor_recycle_cooldown_ticks", 120))
        ]
        candidates.sort(
            key=lambda actor: actor.get_location().distance(ego_vehicle.get_location()),
            reverse=True,
        )
        recycled = 0
        for slot in range(max(0, required)):
            if not candidates:
                break
            target_data = self._find_recycle_target(
                ego_vehicle, ego_progress_m, len(visible) + recycled + slot,
            )
            if target_data is None:
                continue
            actor = candidates.pop(0)
            distance_m, lane_from_right, waypoint = target_data
            replacement = self._replace_actor(
                actor,
                distance_m,
                lane_from_right,
                waypoint,
                ego_speed_kmh=self._vehicle_speed_kmh(ego_vehicle),
            )
            if replacement is None:
                continue
            self._last_actor_recycle_tick.pop(actor.id, None)
            self._last_actor_recycle_tick[replacement.id] = self._tick_count
            recycled += 1
        if recycled:
            self._last_recycle_tick = self._tick_count
            self._recycle_count += recycled
            # Recycled actors are deliberately placed outside the immediate
            # camera range. Count them only after they physically enter the
            # forward visibility cone on a later tick.
            self._visible_count = len(self._visible_actors(ego_vehicle))

    def _in_turn_clearance(self, ego_progress_m):
        for window in self.config.get("traffic_clear_windows_m", []):
            if (
                len(window) == 2
                and float(window[0]) <= ego_progress_m <= float(window[1])
            ):
                return True
        before = float(self.config.get("turn_clear_before_m", 180.0))
        after = float(self.config.get("turn_clear_after_m", 120.0))
        for directive in self.route_manager.applied_directives:
            if str(directive.get("action", "")) not in {"turn_left", "turn_right", "u_turn"}:
                continue
            center = float(directive.get("applied_distance_m", 0.0))
            if center - before <= ego_progress_m <= center + after:
                return True
        return False

    def _visible_actors(self, ego_vehicle):
        maximum = float(self.config.get("visibility_max_distance_m", 120.0))
        half_fov = math.radians(float(self.config.get("visibility_horizontal_fov_deg", 110.0)) * 0.5)
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        result = []
        for actor in self.actors:
            if not actor.is_alive:
                continue
            delta = actor.get_location() - ego_location
            distance = math.sqrt(delta.x * delta.x + delta.y * delta.y)
            if distance < 3.0 or distance > maximum:
                continue
            projection = delta.x * forward.x + delta.y * forward.y
            if projection <= 0.0:
                continue
            lateral = abs(delta.x * forward.y - delta.y * forward.x)
            if math.atan2(lateral, projection) <= half_fov:
                result.append(actor)
        return result

    def _is_safe_recycle_source(self, actor, ego_vehicle):
        minimum = float(self.config.get("recycle_source_min_distance_m", 150.0))
        delta = actor.get_location() - ego_vehicle.get_location()
        distance = math.sqrt(delta.x * delta.x + delta.y * delta.y)
        forward = ego_vehicle.get_transform().get_forward_vector()
        projection = delta.x * forward.x + delta.y * forward.y
        behind_clearance = float(self.config.get("recycle_behind_clearance_m", 45.0))
        if projection <= -behind_clearance:
            return True
        lateral = abs(delta.x * forward.y - delta.y * forward.x)
        angle = math.atan2(lateral, max(projection, 0.001))
        offscreen_minimum = float(self.config.get("recycle_offscreen_min_distance_m", 60.0))
        half_fov = math.radians(
            float(self.config.get("visibility_horizontal_fov_deg", 110.0)) * 0.5
        )
        if distance >= offscreen_minimum and angle > half_fov:
            return True
        return distance >= minimum

    def _find_recycle_target(self, ego_vehicle, ego_progress_m, slot):
        offsets = self.config.get(
            "recycle_relative_offsets_m", [-95, 130, -125, 165]
        )
        lanes = self.config.get("recycle_lane_from_right", [2, 3, 4])
        minimum_clearance = float(self.config.get("recycle_target_clearance_m", 16.0))
        ego_location = ego_vehicle.get_location()
        if bool(self.config.get("recycle_use_planned_route", False)):
            for offset_index in range(len(offsets)):
                offset = float(offsets[(slot + offset_index) % len(offsets)])
                if bool(self.config.get("recycle_forward_only", False)) and offset <= 0.0:
                    continue
                distance_m = max(
                    0.0,
                    min(
                        ego_progress_m + offset,
                        self.route_manager.route_length_m - 10.0,
                    ),
                )
                preferred = [
                    int(lanes[(slot + offset_index + lane_index) % len(lanes)])
                    for lane_index in range(len(lanes))
                ]
                for lane in preferred:
                    waypoint = self._waypoint_at(distance_m, lane)
                    if waypoint is None:
                        continue
                    location = waypoint.transform.location
                    minimum_target_distance = float(
                        self.config.get("recycle_target_min_distance_m", 120.0)
                    )
                    if location.distance(ego_location) < minimum_target_distance:
                        continue
                    occupied = any(
                        other.is_alive
                        and other.get_location().distance(location) < minimum_clearance
                        for other in self.actors
                    )
                    if not occupied:
                        return distance_m, lane, waypoint
            return None

        ego_waypoint = self.world.get_map().get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_waypoint is None:
            return None
        for offset_index in range(len(offsets)):
            offset = float(offsets[(slot + offset_index) % len(offsets)])
            if bool(self.config.get("recycle_forward_only", False)) and offset <= 0.0:
                continue
            base = (
                self._forward_waypoint(ego_waypoint, offset)
                if offset >= 0.0
                else self._backward_waypoint(ego_waypoint, -offset)
            )
            if base is None:
                continue
            available_lanes = self._same_direction_lanes(base)
            preferred = [
                int(lanes[(slot + offset_index + lane_index) % len(lanes)])
                for lane_index in range(len(lanes))
            ]
            lane_order = preferred
            for lane in lane_order:
                if not 0 < lane <= len(available_lanes):
                    continue
                waypoint = available_lanes[lane - 1]
                location = waypoint.transform.location
                minimum_target_distance = float(
                    self.config.get("recycle_target_min_distance_m", 120.0)
                )
                if location.distance(ego_location) < minimum_target_distance:
                    continue
                occupied = any(
                    other.is_alive and other.get_location().distance(location) < minimum_clearance
                    for other in self.actors
                )
                if not occupied:
                    distance_m = max(
                        0.0,
                        min(
                            ego_progress_m + offset,
                            self.route_manager.route_length_m - 10.0,
                        ),
                    )
                    return distance_m, lane, waypoint
        return None

    def _forward_waypoint(self, start, distance_m):
        """Follow the current road topology instead of stale route projection."""
        current = start
        remaining = max(0.0, float(distance_m))
        step_m = max(2.0, float(self.config.get("recycle_topology_step_m", 10.0)))
        while remaining > 0.5:
            step = min(step_m, remaining)
            choices = list(current.next(step))
            if not choices:
                return None
            current_yaw = math.radians(float(current.transform.rotation.yaw))
            current = min(
                choices,
                key=lambda candidate: abs(math.atan2(
                    math.sin(math.radians(float(candidate.transform.rotation.yaw)) - current_yaw),
                    math.cos(math.radians(float(candidate.transform.rotation.yaw)) - current_yaw),
                )),
            )
            remaining -= step
        return current

    def _backward_waypoint(self, start, distance_m):
        current = start
        remaining = max(0.0, float(distance_m))
        step_m = max(2.0, float(self.config.get("recycle_topology_step_m", 10.0)))
        while remaining > 0.5:
            step = min(step_m, remaining)
            choices = list(current.previous(step))
            if not choices:
                return None
            current_yaw = math.radians(float(current.transform.rotation.yaw))
            current = min(
                choices,
                key=lambda candidate: abs(math.atan2(
                    math.sin(math.radians(float(candidate.transform.rotation.yaw)) - current_yaw),
                    math.cos(math.radians(float(candidate.transform.rotation.yaw)) - current_yaw),
                )),
            )
            remaining -= step
        return current

    def _same_direction_lanes(self, base):
        lanes = [base]
        candidate = base.get_right_lane()
        while self._same_direction(base, candidate):
            lanes.insert(0, candidate)
            candidate = candidate.get_right_lane()
        candidate = base.get_left_lane()
        while self._same_direction(base, candidate):
            lanes.append(candidate)
            candidate = candidate.get_left_lane()
        return lanes

    def _path_forward_from_waypoint(self, start):
        points = []
        current = start
        sampling_m = max(5.0, float(self.config.get("path_sampling_m", 25.0)))
        horizon_m = max(sampling_m, float(self.config.get("recycle_path_horizon_m", 800.0)))
        distance_m = sampling_m
        while distance_m <= horizon_m:
            next_waypoint = self._forward_waypoint(current, sampling_m)
            if next_waypoint is None:
                break
            points.append(next_waypoint.transform.location)
            current = next_waypoint
            distance_m += sampling_m
        return points

    def _traffic_manager_path(self, waypoint, *, distance_m, lane_from_right):
        """Build a lane-continuous NPC path without joining distant lane samples."""
        mode = str(self.config.get("traffic_manager_path_mode", "planned")).lower()
        if mode == "local":
            return self._path_forward_from_waypoint(waypoint)
        return self._path_from(distance_m, lane_from_right)

    @staticmethod
    def _vehicle_speed_kmh(actor):
        velocity = actor.get_velocity()
        return 3.6 * math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )

    def _replace_actor(
        self,
        actor,
        distance_m,
        lane_from_right,
        waypoint,
        *,
        ego_speed_kmh=None,
    ):
        """Replace an off-screen actor instead of teleporting its live TM state."""
        spec = self._actor_specs.get(actor.id, {})
        blueprint = self._resolve_vehicle_blueprint(
            str(spec.get("vehicle_type", actor.type_id)),
            str(spec.get("id", actor.id)),
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute(
                "role_name",
                "background_" + str(spec.get("id", actor.id)),
            )
        if blueprint.has_attribute("color"):
            colors = list(blueprint.get_attribute("color").recommended_values)
            if colors:
                stable_id = str(spec.get("id", actor.id))
                blueprint.set_attribute(
                    "color",
                    colors[sum(ord(char) for char in stable_id) % len(colors)],
                )
        transform = carla.Transform(
            carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                z=waypoint.transform.location.z + 1.0,
            ),
            waypoint.transform.rotation,
        )
        replacement = self.world.try_spawn_actor(blueprint, transform)
        if replacement is None:
            return None

        replacement.set_autopilot(True, self.port)
        self._configure_lane_change(replacement, spec)
        if bool(spec.get("ignore_traffic_lights", self.config.get("ignore_traffic_lights", False))):
            self.traffic_manager.ignore_lights_percentage(replacement, 100.0)
        if bool(spec.get("ignore_traffic_signs", self.config.get("ignore_traffic_signs", False))):
            self.traffic_manager.ignore_signs_percentage(replacement, 100.0)
        self.traffic_manager.distance_to_leading_vehicle(
            replacement,
            float(spec.get("following_distance_m", self.config.get("following_distance_m", 12.0))),
        )
        minimum_speed = float(self.config.get("recycle_min_speed_kmh", 40.0))
        maximum_speed = max(
            minimum_speed,
            float(self.config.get("recycle_max_speed_kmh", 48.0)),
        )
        stable_id = str(spec.get("id", replacement.id))
        speed_fraction = (sum(ord(char) for char in stable_id) % 101) / 100.0
        if (
            bool(self.config.get("recycle_follow_ego_speed", False))
            and ego_speed_kmh is not None
        ):
            minimum_delta = float(
                self.config.get("recycle_speed_delta_min_kmh", -4.0)
            )
            maximum_delta = max(
                minimum_delta,
                float(self.config.get("recycle_speed_delta_max_kmh", 4.0)),
            )
            desired_speed = float(ego_speed_kmh) + minimum_delta + speed_fraction * (
                maximum_delta - minimum_delta
            )
            desired_speed = max(minimum_speed, min(maximum_speed, desired_speed))
        else:
            desired_speed = minimum_speed + speed_fraction * (
                maximum_speed - minimum_speed
            )
        speed_limit = max(float(replacement.get_speed_limit()), 1.0)
        self.traffic_manager.vehicle_percentage_speed_difference(
            replacement,
            max(-80.0, min(80.0, 100.0 * (speed_limit - desired_speed) / speed_limit)),
        )
        if bool(self.config.get("traffic_manager_set_path", True)):
            route = self._traffic_manager_path(
                waypoint,
                distance_m=distance_m,
                lane_from_right=lane_from_right,
            )
            if len(route) >= 2:
                self.traffic_manager.set_path(replacement, route)

        old_actor_id = actor.id
        actor_index = self.actors.index(actor)
        self.actors[actor_index] = replacement
        self._actor_specs.pop(old_actor_id, None)
        self._actor_specs[replacement.id] = dict(spec)
        for item in self._scripted:
            if item["actor"].id == old_actor_id:
                item["actor"] = replacement
        actor.set_autopilot(False, self.port)
        actor.destroy()
        self.spawn_log.append({
            "id": spec.get("id", "actor_{0}".format(replacement.id)),
            "status": "respawned_out_of_view",
            "old_actor_id": old_actor_id,
            "actor_id": replacement.id,
            "route_distance_m": round(distance_m, 1),
            "lane_from_right": lane_from_right,
            "tick": self._tick_count,
        })
        return replacement

    def snapshot(self):
        history_limit = max(10, int(self.config.get("snapshot_event_limit", 40)))
        return {
            "mode": "fixed_route_traffic_manager",
            "active_actor_count": sum(1 for actor in self.actors if actor.is_alive),
            "visible_actor_count": self._visible_count,
            "recycle_count": self._recycle_count,
            "traffic_event_count": len(self.spawn_log),
            "recent_traffic_events": list(self.spawn_log[-history_limit:]),
            "scripted_event_count": len(self._scripted),
        }

    def destroy(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors = []


class EgoCentricTraffic:
    """Keep a bounded, varied Traffic Manager flow around the ego vehicle.

    Replenishment deliberately happens behind the ego or well beyond the
    camera's useful range.  This keeps the forward view populated without
    actors visibly popping into the frame.
    """

    def __init__(self, world, client, route_manager, config):
        self.world = world
        self.client = client
        self.route_manager = route_manager
        self.config = dict(config or {})
        self.port = int(self.config.get("traffic_manager_port", 8000))
        self.traffic_manager = None
        self.actors = []
        self.spawn_log = []
        self._rng = random.Random(int(self.config.get("seed", 20260728)))
        self._serial = 0

    def setup(self, ego_vehicle):
        if not self.config.get("enabled", True):
            return []
        self.traffic_manager = self.client.get_trafficmanager(self.port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(self.config.get("seed", 20260728)))
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            float(self.config.get("following_distance_m", 10.0))
        )
        self._replenish(ego_vehicle, initial=True)
        return list(self.actors)

    def tick(self, ego_vehicle, ego_progress_m=0.0):
        if self._in_turn_clearance(float(ego_progress_m)):
            self._clear_for_turn()
            return
        self._remove_distant(ego_vehicle)
        self._replenish(ego_vehicle, initial=False)

    def _in_turn_clearance(self, ego_progress_m):
        for window in self.config.get("traffic_clear_windows_m", []):
            if len(window) == 2 and float(window[0]) <= ego_progress_m <= float(window[1]):
                return True
        before = float(self.config.get("turn_clear_before_m", 180.0))
        after = float(self.config.get("turn_clear_after_m", 120.0))
        for directive in self.route_manager.applied_directives:
            if str(directive.get("action", "")) in {"turn_left", "turn_right", "u_turn"}:
                center = float(directive.get("applied_distance_m", 0.0))
                return center - before <= ego_progress_m <= center + after
        return False

    def _clear_for_turn(self):
        if not self.actors:
            return
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.spawn_log.append({"status": "cleared", "reason": "turn_conflict_window"})
        self.actors = []

    def _remove_distant(self, ego_vehicle):
        maximum = float(self.config.get("despawn_distance_m", 320.0))
        ego_location = ego_vehicle.get_location()
        survivors = []
        for actor in self.actors:
            if not actor.is_alive:
                continue
            if actor.get_location().distance(ego_location) > maximum:
                actor.destroy()
                self.spawn_log.append({"status": "despawned", "actor_id": actor.id, "reason": "outside_ego_window"})
            else:
                survivors.append(actor)
        self.actors = survivors

    def _replenish(self, ego_vehicle, initial):
        desired = int(self.config.get("desired_actor_count", 20))
        needed = max(0, desired - len(self.actors))
        if not needed:
            return
        attempts = int(self.config.get("spawn_attempts_per_tick", 30 if initial else 8))
        ego_progress = float(self.route_manager.progress_m)
        ego_lane = self.world.get_map().get_waypoint(
            ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        ego_lane_id = ego_lane.lane_id if ego_lane is not None else None
        candidates = []
        offsets = (55, 75, 100, 130, 165, 200, 240, 280, 320) if initial else (120, 160, 200, 240, 280, 320)
        for offset in offsets:
            base = self._route_waypoint(ego_progress + offset)
            if base is None:
                continue
            for lane in self._same_direction_lanes(base):
                # Keep the ego lane clear enough to demonstrate commands.
                # Side lanes carry most visible traffic; a sparse lead vehicle
                # is allowed only far enough ahead for safe following.
                if lane.lane_id == ego_lane_id and offset < 180:
                    continue
                transform = carla.Transform(
                    carla.Location(x=lane.transform.location.x, y=lane.transform.location.y, z=lane.transform.location.z + 0.6),
                    lane.transform.rotation,
                )
                candidates.append((transform, ego_progress + offset, self._lane_rank(base, lane)))
        self._rng.shuffle(candidates)
        for transform, route_distance, lane_rank in candidates[:attempts]:
            if len(self.actors) >= desired:
                break
            self._spawn(transform, route_distance, lane_rank)

    def _spawn(self, transform, route_distance, lane_rank):
        blueprints = [item for item in self.world.get_blueprint_library().filter("vehicle.*")
                      if not item.has_attribute("number_of_wheels") or int(item.get_attribute("number_of_wheels").as_int()) >= 4]
        if not blueprints:
            return
        blueprint = self._rng.choice(blueprints)
        self._serial += 1
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "background_flow_{0}".format(self._serial))
        if blueprint.has_attribute("color"):
            colors = list(blueprint.get_attribute("color").recommended_values)
            if colors:
                blueprint.set_attribute("color", self._rng.choice(colors))
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            return
        actor.set_autopilot(True, self.port)
        self.traffic_manager.auto_lane_change(actor, False)
        self.traffic_manager.distance_to_leading_vehicle(
            actor, float(self.config.get("following_distance_m", 10.0))
        )
        target_speed = self._rng.uniform(
            float(self.config.get("min_speed_kmh", 42.0)),
            float(self.config.get("max_speed_kmh", 58.0)),
        )
        limit = max(float(actor.get_speed_limit()), 1.0)
        self.traffic_manager.vehicle_percentage_speed_difference(
            actor, max(-80.0, min(80.0, 100.0 * (limit - target_speed) / limit))
        )
        path = self._path_from(route_distance, lane_rank)
        if len(path) >= 2:
            self.traffic_manager.set_path(actor, path)
        self.actors.append(actor)
        self.spawn_log.append({"status": "spawned", "actor_id": actor.id, "speed_kmh": round(target_speed, 1), "blueprint": blueprint.id, "route_distance_m": round(route_distance, 1), "lane_rank": lane_rank})

    def _route_waypoint(self, distance_m):
        for point in self.route_manager.route:
            if float(point["distance_m"]) >= distance_m:
                return self.world.get_map().get_waypoint(
                    carla.Location(x=point["x"], y=point["y"], z=point.get("z", 0.0)),
                    project_to_road=True, lane_type=carla.LaneType.Driving,
                )
        return None

    @staticmethod
    def _same_direction(reference, candidate):
        return bool(candidate is not None and candidate.lane_type == carla.LaneType.Driving and candidate.road_id == reference.road_id and candidate.lane_id * reference.lane_id > 0)

    def _same_direction_lanes(self, base):
        lanes = [base]
        candidate = base.get_right_lane()
        while self._same_direction(base, candidate):
            lanes.insert(0, candidate)
            candidate = candidate.get_right_lane()
        candidate = base.get_left_lane()
        while self._same_direction(base, candidate):
            lanes.append(candidate)
            candidate = candidate.get_left_lane()
        return lanes

    def _lane_rank(self, base, lane):
        for index, candidate in enumerate(self._same_direction_lanes(base), start=1):
            if candidate.road_id == lane.road_id and candidate.lane_id == lane.lane_id:
                return index
        return 1

    def _path_from(self, distance_m, lane_rank):
        path = []
        step = float(self.config.get("path_sampling_m", 20.0))
        cursor = distance_m + step
        while cursor < self.route_manager.route_length_m:
            base = self._route_waypoint(cursor)
            if base is not None:
                lanes = self._same_direction_lanes(base)
                if lane_rank <= len(lanes):
                    path.append(lanes[lane_rank - 1].transform.location)
            cursor += step
        return path

    def snapshot(self):
        return {
            "mode": "ego_centric_traffic_manager",
            "active_actor_count": sum(1 for actor in self.actors if actor.is_alive),
            "spawn_log": list(self.spawn_log[-80:]),
        }

    def destroy(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        self.actors = []
