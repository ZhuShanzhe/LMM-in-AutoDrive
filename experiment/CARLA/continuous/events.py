"""Deterministic vehicle and pedestrian events owned by a continuous scenario."""

import math

import carla

from continuous.traffic import find_sidewalk_waypoint, route_point_at


class AdjacentLeadBrakeEvent:
    """A visible controlled vehicle that brakes in the adjacent lane.

    It demonstrates event triggering and actor lifecycle without placing the
    ego vehicle in an unavoidable collision state before a learned policy is
    integrated.
    """

    def __init__(self, world, ego_vehicle, event, route_manager):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.event = event
        self.route_manager = route_manager
        self.vehicle = None
        self.status = "INITIALIZED"
        self.elapsed_s = 0.0
        self.brake_started_s = None
        self.fixed_delta_s = float(event.get("fixed_delta_s", 0.1))
        self.brake_at_route_progress_m = event.get("brake_at_route_progress_m")

    def setup(self):
        blueprint = self.world.get_blueprint_library().find(
            self.event.get("vehicle_type", "vehicle.audi.tt")
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "controlled_brake_vehicle")
        self.vehicle = self._spawn_vehicle(blueprint)
        if self.vehicle is None:
            raise RuntimeError("Could not spawn controlled brake vehicle")
        self.status = "CRUISING"

    def tick(self):
        if self.vehicle is None or not self.vehicle.is_alive:
            self.status = "FAILED"
            return
        self.elapsed_s += self.fixed_delta_s
        brake_after_s = float(self.event.get("brake_after_s", 2.0))
        brake_hold_s = float(self.event.get("brake_hold_s", 2.0))
        route_trigger_reached = (
            self.brake_at_route_progress_m is None
            or self.route_manager.progress_m >= float(self.brake_at_route_progress_m)
        )
        if self.elapsed_s < brake_after_s or not route_trigger_reached:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=float(self.event.get("cruise_throttle", 0.28)),
                brake=0.0,
            ))
            return
        if self.brake_started_s is None:
            self.brake_started_s = self.elapsed_s
            self.status = "BRAKING"
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        if self.elapsed_s - self.brake_started_s >= brake_hold_s:
            self.status = "COMPLETED"

    def finished(self):
        return self.status in ("COMPLETED", "FAILED")

    def get_status(self):
        return {
            "status": self.status,
            "actor_id": self.vehicle.id if self.vehicle is not None else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "brake_started_s": self.brake_started_s,
        }

    def destroy(self):
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        self.vehicle = None

    def _route_point(self, distance_m):
        for point in self.route_manager.route[self.route_manager.current_index:]:
            if point["distance_m"] >= distance_m:
                return point
        return self.route_manager.route[-1]

    def _spawn_vehicle(self, blueprint):
        preferred_side = str(self.event.get("lane", "right")).lower()
        alternate_side = "left" if preferred_side == "right" else "right"
        base_offset = float(self.event.get("spawn_offset_m", 36.0))
        retry_step = float(self.event.get("spawn_retry_step_m", 18.0))
        retry_count = int(self.event.get("spawn_retry_count", 6))
        world_map = self.world.get_map()
        for attempt in range(retry_count):
            point = self._route_point(
                self.route_manager.progress_m + base_offset + retry_step * attempt
            )
            waypoint = world_map.get_waypoint(
                carla.Location(x=point["x"], y=point["y"], z=point["z"]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            sides = ("same",) if preferred_side == "same" else (preferred_side, alternate_side)
            for side in sides:
                if side == "same":
                    adjacent = waypoint
                else:
                    adjacent = (
                        waypoint.get_right_lane()
                        if side == "right"
                        else waypoint.get_left_lane()
                    )
                if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
                    continue
                transform = carla.Transform(
                    carla.Location(
                        x=adjacent.transform.location.x,
                        y=adjacent.transform.location.y,
                        z=adjacent.transform.location.z + 0.25,
                    ),
                    adjacent.transform.rotation,
                )
                vehicle = self.world.try_spawn_actor(blueprint, transform)
                if vehicle is not None:
                    return vehicle
        return None


class CutInVehicleEvent:
    """A Traffic Manager vehicle that visibly merges into the ego lane."""

    def __init__(self, world, ego_vehicle, event, route_manager, traffic_manager):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.event = event
        self.route_manager = route_manager
        self.traffic_manager = traffic_manager
        self.vehicle = None
        self.status = "INITIALIZED"
        self.elapsed_s = 0.0
        self.lane_change_requested = False
        self.lane_change_requested_s = None
        self.source_lane_id = None
        self.target_lane_id = None
        self.target_road_id = None
        self.merge_observed_s = None
        self.fixed_delta_s = float(event.get("fixed_delta_s", 0.1))
        self.merge_at_route_progress_m = event.get("merge_at_route_progress_m")

    def setup(self):
        blueprint = self.world.get_blueprint_library().find(
            self.event.get("vehicle_type", "vehicle.lincoln.mkz_2020")
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "controlled_cut_in_vehicle")
        self.vehicle, direction_right = self._spawn_vehicle(blueprint)
        if self.vehicle is None:
            raise RuntimeError("Could not spawn controlled cut-in vehicle")
        self.direction_right = direction_right
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        if hasattr(self.traffic_manager, "auto_lane_change"):
            self.traffic_manager.auto_lane_change(self.vehicle, False)
        if hasattr(self.traffic_manager, "set_desired_speed"):
            self.traffic_manager.set_desired_speed(
                self.vehicle, float(self.event.get("speed_kmh", 35.0))
            )
        self.status = "FOLLOWING"

    def tick(self):
        if self.vehicle is None or not self.vehicle.is_alive:
            self.status = "FAILED"
            return
        self.elapsed_s += self.fixed_delta_s
        route_trigger_reached = (
            self.merge_at_route_progress_m is None
            or self.route_manager.progress_m >= float(self.merge_at_route_progress_m)
        )
        if (
            not self.lane_change_requested
            and self.elapsed_s >= float(self.event.get("lane_change_after_s", 2.5))
            and route_trigger_reached
        ):
            self.traffic_manager.force_lane_change(self.vehicle, self.direction_right)
            self.lane_change_requested = True
            self.lane_change_requested_s = self.elapsed_s
            self.status = "MERGING"
        if self.lane_change_requested and self._in_target_lane():
            if self.merge_observed_s is None:
                self.merge_observed_s = self.elapsed_s
            self.status = "MERGED"
        if (
            self.merge_observed_s is not None
            and self.elapsed_s - self.merge_observed_s
            >= float(self.event.get("post_merge_hold_s", 2.0))
        ):
            self.status = "COMPLETED"
            return
        if (
            self.lane_change_requested
            and self.lane_change_requested_s is not None
            and self.elapsed_s - self.lane_change_requested_s
            >= float(self.event.get("merge_timeout_s", 12.0))
        ):
            self.status = "FAILED"

    def finished(self):
        return self.status in ("COMPLETED", "FAILED")

    def get_status(self):
        return {
            "status": self.status,
            "actor_id": self.vehicle.id if self.vehicle is not None else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "lane_change_requested": self.lane_change_requested,
            "lane_change_requested_s": self.lane_change_requested_s,
            "source_lane_id": self.source_lane_id,
            "target_lane_id": self.target_lane_id,
            "current_lane_id": self._current_lane_id(),
            "merge_observed_s": self.merge_observed_s,
        }

    def destroy(self):
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        self.vehicle = None

    def _spawn_vehicle(self, blueprint):
        side = str(self.event.get("lane", "left")).lower()
        alternates = (side, "right" if side == "left" else "left")
        base_offset = float(self.event.get("spawn_offset_m", 58.0))
        world_map = self.world.get_map()
        for offset_index in range(int(self.event.get("spawn_retry_count", 5))):
            point = route_point_at(
                self.route_manager,
                self.route_manager.progress_m + base_offset + offset_index * 15.0,
            )
            waypoint = world_map.get_waypoint(
                carla.Location(x=point["x"], y=point["y"], z=point["z"]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            for candidate_side in alternates:
                adjacent = (
                    waypoint.get_left_lane()
                    if candidate_side == "left"
                    else waypoint.get_right_lane()
                )
                if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
                    continue
                transform = carla.Transform(
                    carla.Location(
                        x=adjacent.transform.location.x,
                        y=adjacent.transform.location.y,
                        z=adjacent.transform.location.z + 0.25,
                    ),
                    adjacent.transform.rotation,
                )
                vehicle = self.world.try_spawn_actor(blueprint, transform)
                if vehicle is not None:
                    direction_right = self._direction_to_target(adjacent, waypoint)
                    if direction_right is None:
                        vehicle.destroy()
                        continue
                    self.source_lane_id = adjacent.lane_id
                    self.target_lane_id = waypoint.lane_id
                    self.target_road_id = waypoint.road_id
                    return vehicle, direction_right
        return None, False

    @staticmethod
    def _direction_to_target(source_waypoint, target_waypoint):
        for direction_right, candidate in (
            (False, source_waypoint.get_left_lane()),
            (True, source_waypoint.get_right_lane()),
        ):
            if (
                candidate is not None
                and candidate.lane_type == carla.LaneType.Driving
                and candidate.road_id == target_waypoint.road_id
                and candidate.lane_id == target_waypoint.lane_id
            ):
                return direction_right
        return None

    def _in_target_lane(self):
        if self.target_lane_id is None or self.vehicle is None or not self.vehicle.is_alive:
            return False
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        return (
            waypoint is not None
            and waypoint.lane_id == self.target_lane_id
        )

    def _current_lane_id(self):
        if self.vehicle is None or not self.vehicle.is_alive:
            return None
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        return waypoint.lane_id if waypoint is not None else None


class PedestrianCrossingEvent:
    """A walker that leaves a sidewalk and crosses part of the carriageway."""

    def __init__(self, world, ego_vehicle, event, route_manager):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.event = event
        self.route_manager = route_manager
        self.walker = None
        self.status = "INITIALIZED"
        self.elapsed_s = 0.0
        self.distance_m = 0.0
        self.direction = None
        self.start_location = None
        self.start_rotation = None
        self.crossing_observed_s = None
        self.motion_mode = str(event.get("motion_mode", "kinematic")).lower()
        self.fixed_delta_s = float(event.get("fixed_delta_s", 0.1))
        self.cross_at_route_progress_m = event.get("cross_at_route_progress_m")

    def setup(self):
        blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        if not blueprints:
            raise RuntimeError("No pedestrian blueprint is available")
        point = route_point_at(
            self.route_manager,
            self.route_manager.progress_m + float(self.event.get("spawn_offset_m", 55.0)),
        )
        road_waypoint = self.world.get_map().get_waypoint(
            carla.Location(x=point["x"], y=point["y"], z=point["z"]),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if road_waypoint is None:
            raise RuntimeError("Could not locate a road lane for pedestrian crossing")
        side = str(self.event.get("side", "right")).lower()
        sidewalk = find_sidewalk_waypoint(road_waypoint, side)
        if sidewalk is None:
            forward = road_waypoint.transform.get_forward_vector()
            right = carla.Vector3D(x=forward.y, y=-forward.x, z=0.0)
            lateral = float(self.event.get("fallback_roadside_offset_m", 7.0))
            sign = 1.0 if side == "right" else -1.0
            location = carla.Location(
                x=road_waypoint.transform.location.x + right.x * lateral * sign,
                y=road_waypoint.transform.location.y + right.y * lateral * sign,
                z=road_waypoint.transform.location.z + 0.35,
            )
            transform = carla.Transform(location, road_waypoint.transform.rotation)
        else:
            transform = carla.Transform(
                carla.Location(
                    x=sidewalk.transform.location.x,
                    y=sidewalk.transform.location.y,
                    z=sidewalk.transform.location.z + 0.35,
                ),
                sidewalk.transform.rotation,
            )
        self.walker = self.world.try_spawn_actor(blueprints[0], transform)
        if self.walker is None:
            raise RuntimeError("Could not spawn crossing pedestrian")
        self.start_location = carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
        )
        self.start_rotation = transform.rotation
        yaw = road_waypoint.transform.rotation.yaw
        radians = math.radians(yaw)
        lateral_left = carla.Vector3D(
            x=-math.sin(radians), y=math.cos(radians), z=0.0
        )
        self.direction = lateral_left if side == "right" else carla.Vector3D(
            x=-lateral_left.x, y=-lateral_left.y, z=0.0
        )
        self.status = "WAITING"

    def tick(self):
        if self.walker is None or not self.walker.is_alive:
            self.status = "FAILED"
            return
        if (
            self.cross_at_route_progress_m is not None
            and self.route_manager.progress_m < float(self.cross_at_route_progress_m)
        ):
            return
        if self.status == "WAITING":
            self.status = "CROSSING"
        speed = float(self.event.get("speed_mps", 1.35))
        self.walker.apply_control(carla.WalkerControl(direction=self.direction, speed=speed))
        self.elapsed_s += self.fixed_delta_s
        required_distance = float(self.event.get("cross_distance_m", 5.0))
        if self.motion_mode == "kinematic":
            self.distance_m = min(required_distance, speed * self.elapsed_s)
            self.walker.set_transform(carla.Transform(
                carla.Location(
                    x=self.start_location.x + self.direction.x * self.distance_m,
                    y=self.start_location.y + self.direction.y * self.distance_m,
                    z=self.start_location.z,
                ),
                self.start_rotation,
            ))
        else:
            self.distance_m = self.walker.get_location().distance(self.start_location)
        if self.distance_m >= required_distance:
            if self.crossing_observed_s is None:
                self.crossing_observed_s = self.elapsed_s
            self.status = "CLEARED"
        if (
            self.crossing_observed_s is not None
            and self.elapsed_s - self.crossing_observed_s
            >= float(self.event.get("post_cross_hold_s", 2.0))
        ):
            self.status = "COMPLETED"
        elif self.elapsed_s >= float(self.event.get("cross_timeout_s", 15.0)):
            self.status = "FAILED"

    def finished(self):
        return self.status in ("COMPLETED", "FAILED")

    def get_status(self):
        return {
            "status": self.status,
            "actor_id": self.walker.id if self.walker is not None else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "distance_m": round(self.distance_m, 3),
            "required_distance_m": float(self.event.get("cross_distance_m", 5.0)),
            "crossing_observed_s": self.crossing_observed_s,
            "motion_mode": self.motion_mode,
        }

    def destroy(self):
        if self.walker is not None and self.walker.is_alive:
            self.walker.destroy()
        self.walker = None
