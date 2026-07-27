"""Continuous, voice-command demonstration for the basic competition track."""

import json
import os

import carla

from continuous.route_manager import RouteManager
from control.voice_schedule_policy import VoiceSchedulePolicy
from scenarios.base import BaseScenario
from scenarios.utils.vehicle import VehicleSpawner


class BasicVoiceControl5KmScenario(BaseScenario):
    """One ego vehicle, one route, and a deterministic temporary voice policy."""

    default_map = "Town04_Opt"
    default_duration_s = 420.0

    def __init__(self, world, external_control=True, config_path=None):
        super().__init__(world, external_control)
        self.scenario_id = "basic_voice_control_5km"
        self.scenario_name = "Basic Voice Control 5km"
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "basic_voice_control_5km.json",
        )
        self.config = self._load_config(self.config_path)
        self.vehicle_spawner = VehicleSpawner(world)
        self.route_manager = RouteManager(world)
        self.ego_vehicle = None
        self.route_preflight = {}
        self.command_ids = {item["id"] for item in self.config["commands"]}
        self.emitted_command_ids = set()
        self.active_action = "keep_lane"
        self.goal_tolerance_m = float(self.config["route"].get("goal_tolerance_m", 15.0))
        self.success_condition = dict(self.config["success"])
        self.failure_conditions = ["collision", "illegal_lane_invasion", "route_planning_incomplete"]
        self.trigger = {
            "type": "route_progress_schedule",
            "commands": [item["id"] for item in self.config["commands"]],
        }

    def setup(self):
        self._apply_weather()
        directives = []
        for command in self.config["commands"]:
            directive = command.get("route_directive")
            if directive:
                item = dict(directive)
                item["id"] = command["id"]
                directives.append(item)
        spawn_point = self._select_spawn_point_and_route(directives)
        self.ego_vehicle = self.vehicle_spawner.spawn_ego_vehicle(spawn_point)
        self.add_actor(self.ego_vehicle, "ego")

        if not self.route:
            raise RuntimeError("Route planner produced no route points")

        last = self.route[-1]
        self.goal_location = carla.Location(x=last["x"], y=last["y"], z=last["z"])
        self.status = "RUNNING"

    def tick(self):
        if self._finished:
            return
        self.metrics["simulation_time"] += self.fixed_delta_s
        self._update_route_progress()

    def create_temporary_policy(
        self,
        default_speed_kmh,
        parser_model_path=None,
        parser_device="cuda",
    ):
        return VoiceSchedulePolicy(
            self.config["commands"],
            default_speed_kmh,
            parser_model_path,
            parser_device,
        )

    def get_policy_context(self):
        return {
            "progress_m": self.route_manager.progress_m,
            "route_target": self.route_manager.target_point(7.0),
            "turn_route_target": self.route_manager.target_point(5.0),
        }

    def report_intent(self, intent):
        self.active_action = str(intent.get("action", "keep_lane"))
        command_id = intent.get("command_id")
        if command_id in self.command_ids:
            self.emitted_command_ids.add(command_id)

    def report_events(self, events):
        if self._finished:
            return
        self._update_route_progress()
        self.metrics["collision_count"] = int(events.get("collision_count", 0))
        self.metrics["lane_invasion_count"] = int(events.get("lane_invasion_count", 0))
        if events.get("collision"):
            self.failure("collision_detected")
            return

        illegal_events = [
            item for item in events.get("new_lane_invasion_events", [])
            if self._is_illegal_lane_invasion(item)
        ]
        self.metrics["illegal_lane_invasion_count"] = self.metrics.get(
            "illegal_lane_invasion_count", 0
        ) + len(illegal_events)
        if illegal_events:
            self.failure("illegal_lane_invasion")
            return

        if self.route_manager.is_finished(self.goal_tolerance_m):
            if self.emitted_command_ids != self.command_ids:
                self.failure("route_finished_before_all_commands_emitted")
            else:
                self.success("route_completed_without_collision_or_illegal_lane_invasion")

    def get_status(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "actors": self.get_actor_ids(),
            "metrics": self.metrics,
            "route_progress_m": round(self.route_manager.progress_m, 3),
            "route_length_m": round(self.route_manager.route_length_m, 3),
            "emitted_command_ids": sorted(self.emitted_command_ids),
            "applied_route_directives": self.route_manager.applied_directives,
            "route_preflight": dict(self.route_preflight),
        }

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def _update_route_progress(self):
        if self.ego_vehicle is None:
            return
        progress_m = self.route_manager.update(self.ego_vehicle)
        self.metrics["route_progress_m"] = round(progress_m, 3)
        self.metrics["route_length_m"] = round(self.route_manager.route_length_m, 3)

    def _select_spawn_point_and_route(self, directives):
        minimum_lanes = int(self.config["route"].get("minimum_same_direction_lanes", 3))
        max_uncommanded_turn = float(self.config["route"].get(
            "max_uncommanded_heading_delta_deg", 18.0
        ))
        directive_tolerance = float(self.config["route"].get(
            "directive_turn_tolerance_m", 180.0
        ))
        continuity_horizon = float(self.config["route"].get(
            "route_continuity_horizon_m", 400.0
        ))
        world_map = self.world.get_map()
        spawn_points = world_map.get_spawn_points()
        requested_index = self.config["route"].get("start_spawn_index")
        candidates = []
        rejected = {"lane": 0, "directive": 0, "continuity": 0}
        if requested_index is not None and int(requested_index) < len(spawn_points):
            candidates.append(spawn_points[int(requested_index)])
        candidates.extend(spawn_points)
        seen = set()
        for candidate_index, transform in enumerate(candidates):
            key = (round(transform.location.x, 2), round(transform.location.y, 2))
            if key in seen:
                continue
            seen.add(key)
            waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
            if waypoint is None or self._same_direction_lane_count(waypoint) < minimum_lanes:
                rejected["lane"] += 1
                continue
            spawn_transform = self._offset_spawn_transform(transform)
            route = self.route_manager.build_route(
                start_location=spawn_transform.location,
                length_m=self.config["route"]["length_m"],
                step_m=self.config["route"].get("step_m", 5.0),
                directives=directives,
            )
            if not route or self.route_manager.unapplied_directives:
                rejected["directive"] += 1
                continue
            if self._has_uncommanded_sharp_turn(
                route,
                self.route_manager.applied_directives,
                max_uncommanded_turn,
                directive_tolerance,
                continuity_horizon,
            ):
                rejected["continuity"] += 1
                continue
            self.route = route
            self.route_preflight = {
                "selected_candidate_position": candidate_index,
                "selected_spawn_location": {
                    "x": round(float(spawn_transform.location.x), 3),
                    "y": round(float(spawn_transform.location.y), 3),
                    "z": round(float(spawn_transform.location.z), 3),
                },
                "applied_directives": list(self.route_manager.applied_directives),
                "continuity_horizon_m": continuity_horizon,
                "max_uncommanded_heading_delta_deg": max_uncommanded_turn,
            }
            return spawn_transform
        raise RuntimeError(
            "No spawn point satisfies lane, route-directive, and route-continuity requirements: {0}".format(
                rejected
            )
        )

    def _offset_spawn_transform(self, transform):
        """Optionally move a spawn point into the road before physics starts."""
        backoff_m = float(self.config["route"].get("spawn_backoff_m", 0.0))
        if backoff_m <= 0.0:
            return transform
        forward = transform.get_forward_vector()
        location = carla.Location(
            x=transform.location.x + forward.x * backoff_m,
            y=transform.location.y + forward.y * backoff_m,
            z=transform.location.z,
        )
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)
        if waypoint is None:
            return transform
        snapped_location = waypoint.transform.location
        snapped_location.z = transform.location.z
        return carla.Transform(snapped_location, transform.rotation)

    @staticmethod
    def _has_uncommanded_sharp_turn(
        route,
        directives,
        maximum_delta_deg,
        tolerance_m,
        horizon_m,
    ):
        allowed_turns = [float(item.get("applied_distance_m", -1.0)) for item in directives]
        for current, following in zip(route, route[1:]):
            if float(current["distance_m"]) > horizon_m:
                break
            delta = (float(following["yaw"]) - float(current["yaw"]) + 180.0) % 360.0 - 180.0
            if abs(delta) <= maximum_delta_deg:
                continue
            if any(abs(float(current["distance_m"]) - point) <= tolerance_m for point in allowed_turns):
                continue
            return True
        return False

    def _apply_weather(self):
        weather_name = self.config["environment"].get("weather", "ClearNoon")
        weather = getattr(carla.WeatherParameters, weather_name, None)
        if weather is None:
            raise ValueError("Unknown CARLA weather preset: {0}".format(weather_name))
        self.world.set_weather(weather)

    @staticmethod
    def _same_direction_lane_count(waypoint):
        lanes = [waypoint]
        for getter in ("get_left_lane", "get_right_lane"):
            candidate = getattr(waypoint, getter)()
            while candidate is not None:
                if (
                    candidate.lane_type != carla.LaneType.Driving
                    or candidate.road_id != waypoint.road_id
                    or candidate.lane_id * waypoint.lane_id <= 0
                ):
                    break
                lanes.append(candidate)
                candidate = getattr(candidate, getter)()
        return len(lanes)

    def _is_illegal_lane_invasion(self, event):
        if self.active_action in ("lane_change_left", "lane_change_right"):
            return False
        markings = " ".join(event.get("markings", [])).lower()
        if not any(token in markings for token in ("solid", "double", "curb")):
            return False
        waypoint = self.world.get_map().get_waypoint(
            self.ego_vehicle.get_location(), project_to_road=True
        )
        # CARLA may report a solid connector marking while a planned turn is
        # traversing an intersection. That is not a lane-departure violation.
        return waypoint is None or not bool(getattr(waypoint, "is_junction", False))

    @staticmethod
    def _load_config(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        required = {"environment", "route", "commands", "success"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("Scenario config missing keys: {0}".format(", ".join(missing)))
        return payload
