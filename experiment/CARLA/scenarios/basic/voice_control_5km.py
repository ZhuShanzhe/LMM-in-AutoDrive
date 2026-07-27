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
        self.command_ids = {item["id"] for item in self.config["commands"]}
        self.emitted_command_ids = set()
        self.goal_tolerance_m = float(self.config["route"].get("goal_tolerance_m", 15.0))
        self.success_condition = dict(self.config["success"])
        self.failure_conditions = ["collision", "illegal_lane_invasion", "route_planning_incomplete"]
        self.trigger = {
            "type": "route_progress_schedule",
            "commands": [item["id"] for item in self.config["commands"]],
        }

    def setup(self):
        self._apply_weather()
        spawn_point = self._select_spawn_point()
        self.ego_vehicle = self.vehicle_spawner.spawn_ego_vehicle(spawn_point)
        self.add_actor(self.ego_vehicle, "ego")

        directives = []
        for command in self.config["commands"]:
            directive = command.get("route_directive")
            if directive:
                item = dict(directive)
                item["id"] = command["id"]
                directives.append(item)
        self.route = self.route_manager.build_route(
            start_location=spawn_point.location,
            length_m=self.config["route"]["length_m"],
            step_m=self.config["route"].get("step_m", 5.0),
            directives=directives,
        )
        if not self.route:
            raise RuntimeError("Route planner produced no route points")
        if self.route_manager.unapplied_directives:
            names = [item.get("id", item.get("action", "unknown")) for item in self.route_manager.unapplied_directives]
            raise RuntimeError("Route planner could not apply directives: {0}".format(", ".join(names)))

        last = self.route[-1]
        self.goal_location = carla.Location(x=last["x"], y=last["y"], z=last["z"])
        self.status = "RUNNING"

    def tick(self):
        if self._finished:
            return
        self.metrics["simulation_time"] += self.fixed_delta_s
        self._update_route_progress()

    def create_temporary_policy(self, default_speed_kmh):
        return VoiceSchedulePolicy(self.config["commands"], default_speed_kmh)

    def get_policy_context(self):
        return {
            "progress_m": self.route_manager.progress_m,
            "route_target": self.route_manager.target_point(22.0),
        }

    def report_intent(self, intent):
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
        }

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def _update_route_progress(self):
        if self.ego_vehicle is None:
            return
        progress_m = self.route_manager.update(self.ego_vehicle)
        self.metrics["route_progress_m"] = round(progress_m, 3)
        self.metrics["route_length_m"] = round(self.route_manager.route_length_m, 3)

    def _select_spawn_point(self):
        minimum_lanes = int(self.config["route"].get("minimum_same_direction_lanes", 3))
        world_map = self.world.get_map()
        for transform in world_map.get_spawn_points():
            waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
            if waypoint is not None and self._same_direction_lane_count(waypoint) >= minimum_lanes:
                return transform
        raise RuntimeError(
            "No spawn point satisfies the configured same-direction lane requirement"
        )

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

    @staticmethod
    def _is_illegal_lane_invasion(event):
        markings = " ".join(event.get("markings", [])).lower()
        return any(token in markings for token in ("solid", "double", "curb"))

    @staticmethod
    def _load_config(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        required = {"environment", "route", "commands", "success"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("Scenario config missing keys: {0}".format(", ".join(missing)))
        return payload
