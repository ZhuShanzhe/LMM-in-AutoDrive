"""Configurable 5 km urban test environment for the basic competition track."""

import json
import os

import carla

from continuous.events import (
    AdjacentLeadBrakeEvent,
    CutInVehicleEvent,
    PedestrianCrossingEvent,
)
from continuous.route_manager import RouteManager
from continuous.scenario_manager import ScenarioManager
from continuous.traffic import (
    PedestrianFlowManager,
    TrafficFlowManager,
    TrafficManagerRouteController,
)
from scenarios.base import BaseScenario
from scenarios.utils.vehicle import VehicleSpawner


class RouteAutopilotPolicy:
    """Temporary route policy used while the VLA decision module is integrated."""

    def __init__(self, scenario):
        self.scenario = scenario

    def decide(self, world_state):
        del world_state
        return {
            "action": "keep_lane",
            "target_speed_kmh": self.scenario.target_speed_kmh,
            "reason": "traffic_manager_route_following",
        }

    def telemetry(self):
        return {
            "mode": "traffic_manager_route_following",
            "route_progress_m": round(self.scenario.route_manager.progress_m, 3),
            "route_length_m": round(self.scenario.route_manager.route_length_m, 3),
            "active_events": self.scenario.event_manager.snapshot()["active"],
        }


class BasicTrack5KmScenario(BaseScenario):
    """A continuous urban route with traffic flow and independently owned events."""

    default_map = "Town05_Opt"
    default_duration_s = 650.0

    def __init__(self, world, external_control=True, config_path=None):
        super().__init__(world, external_control)
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "basic_track_5km_demo.json",
        )
        self.config = self._load_config(self.config_path)
        self.scenario_id = self.config["scenario_id"]
        self.scenario_name = self.config["scenario_name"]
        self.default_map = self.config.get("map", self.default_map)
        self.default_duration_s = float(self.config.get("duration_s", self.default_duration_s))
        self.target_speed_kmh = float(self.config["ego"]["target_speed_kmh"])
        self.goal_tolerance_m = float(self.config["route"].get("goal_tolerance_m", 20.0))
        self.resume_progress_m = float(self.config["route"].get("resume_progress_m", 0.0))
        self.vehicle_spawner = VehicleSpawner(world)
        self.route_manager = RouteManager(world)
        self.event_manager = ScenarioManager(world, self.route_manager)
        self.traffic = TrafficFlowManager(world, self.config.get("traffic"))
        self.pedestrians = PedestrianFlowManager(world, self.config.get("pedestrians"))
        self.ego_vehicle = None
        self.client = None
        self.lifecycle_log = []
        self.success_condition = {
            "route_length_m": self.config["route"]["length_m"],
            "route_completion_tolerance_m": self.goal_tolerance_m,
            "no_collision": True,
            "no_illegal_lane_invasion": True,
            "all_configured_events_completed": True,
        }
        self.failure_conditions = ["collision", "illegal_lane_invasion", "route_timeout"]
        self.trigger = {
            "type": "route_progress_schedule",
            "event_ids": [item["id"] for item in self.config.get("events", [])],
        }

    def setup(self):
        if self.client is None:
            raise RuntimeError("Scenario requires the CARLA client before setup")
        self._apply_weather()
        spawn_point = self._select_spawn_point()
        self.route = self.route_manager.build_route(
            start_location=spawn_point.location,
            length_m=self.config["route"]["length_m"],
            step_m=self.config["route"].get("step_m", 5.0),
        )
        if self.route_manager.route_length_m < self.config["route"]["length_m"] - 5.0:
            raise RuntimeError("Route ended before the configured continuous distance")
        if self.resume_progress_m > 0.0:
            self.route_manager.seek(self.resume_progress_m)
            spawn_point = self._route_transform_at_current_progress()
        self.ego_vehicle = self.vehicle_spawner.spawn_ego_vehicle(spawn_point)
        self.add_actor(self.ego_vehicle, "ego")
        last = self.route[-1]
        self.goal_location = carla.Location(x=last["x"], y=last["y"], z=last["z"])

        self.traffic.start(self.client)
        clear_at_m = self.config.get("traffic", {}).get("clear_at_m")
        if clear_at_m is not None and self.route_manager.progress_m >= float(clear_at_m):
            self.traffic.deactivated_at_m = self.route_manager.progress_m
        else:
            self.traffic.spawn_background(self.ego_vehicle, self.route_manager)
        self.pedestrians.spawn(self.route_manager)
        events = []
        for item in self.config.get("events", []):
            event = dict(item)
            event["fixed_delta_s"] = self.fixed_delta_s
            events.append(event)
        self.event_manager.register(
            "adjacent_lead_brake",
            lambda world, ego, event: AdjacentLeadBrakeEvent(
                world, ego, event, self.route_manager
            ),
        )
        self.event_manager.register(
            "cut_in_vehicle",
            lambda world, ego, event: CutInVehicleEvent(
                world, ego, event, self.route_manager, self.traffic.traffic_manager
            ),
        )
        self.event_manager.register(
            "pedestrian_crossing",
            lambda world, ego, event: PedestrianCrossingEvent(
                world, ego, event, self.route_manager
            ),
        )
        self.event_manager.set_events(events)
        self._mark_completed_events_before_resume()
        self.status = "RUNNING"

    def tick(self):
        if self._finished:
            return
        self.metrics["simulation_time"] += self.fixed_delta_s
        self.event_manager.tick(self.ego_vehicle)
        self._update_route_metrics()
        self.traffic.tick(self.route_manager)
        self.pedestrians.tick(self.route_manager, self.fixed_delta_s)
        self._deactivate_traffic_if_due()

    def create_decision_policy(self):
        return RouteAutopilotPolicy(self)

    def create_controller(self):
        return TrafficManagerRouteController(
            self.ego_vehicle,
            self.traffic.traffic_manager,
            self.route_manager,
            self.target_speed_kmh,
        )

    def report_events(self, events):
        if self._finished:
            return
        self._update_route_metrics()
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
            snapshot = self.event_manager.snapshot()
            if snapshot["failed"]:
                self.failure("configured_event_failed")
            elif snapshot["active"] or snapshot["pending"]:
                self.failure("route_finished_before_events_completed")
            else:
                self.success("continuous_route_completed")

    def drain_event_log(self):
        result = self.lifecycle_log + self.event_manager.drain_event_log()
        self.lifecycle_log = []
        return result

    def get_status(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "actors": self.get_actor_ids(),
            "metrics": self.metrics,
            "route_progress_m": round(self.route_manager.progress_m, 3),
            "route_length_m": round(self.route_manager.route_length_m, 3),
            "traffic": self.traffic.snapshot(),
            "pedestrians": self.pedestrians.snapshot(),
            "scenario_events": self.event_manager.snapshot(),
        }

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def get_goal_distance_m(self):
        return self.route_manager.route_length_m

    def get_scenario_info(self):
        info = super().get_scenario_info()
        info.update({
            "config_path": self.config_path,
            "environment": self.config["environment"],
            "traffic": self.config.get("traffic", {}),
            "pedestrians": self.config.get("pedestrians", {}),
            "events": self.config.get("events", []),
        })
        return info

    def destroy(self):
        self.event_manager.destroy()
        self.pedestrians.destroy()
        self.traffic.destroy()
        super().destroy()

    def restore_runtime(self):
        self.traffic.restore_runtime()

    def _update_route_metrics(self):
        progress_m = self.route_manager.update(self.ego_vehicle)
        self.metrics["route_progress_m"] = round(progress_m, 3)
        self.metrics["route_length_m"] = round(self.route_manager.route_length_m, 3)

    def _deactivate_traffic_if_due(self):
        clear_at_m = self.config.get("traffic", {}).get("clear_at_m")
        if clear_at_m is None or self.traffic.deactivated_at_m is not None:
            return
        if self.route_manager.progress_m < float(clear_at_m):
            return
        actor_count = self.traffic.deactivate_background(self.route_manager.progress_m)
        self.lifecycle_log.append({
            "type": "traffic_flow_deactivated",
            "route_progress_m": self.traffic.deactivated_at_m,
            "background_actor_count": actor_count,
        })

    def _select_spawn_point(self):
        world_map = self.world.get_map()
        spawn_points = world_map.get_spawn_points()
        desired_index = self.config["route"].get("start_spawn_index")
        candidates = []
        if desired_index is not None and int(desired_index) < len(spawn_points):
            candidates.append(spawn_points[int(desired_index)])
        candidates.extend(spawn_points)
        minimum_lanes = int(self.config["route"].get("minimum_same_direction_lanes", 3))
        seen = set()
        for transform in candidates:
            key = (round(transform.location.x, 2), round(transform.location.y, 2))
            if key in seen:
                continue
            seen.add(key)
            waypoint = world_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None and self._same_direction_lane_count(waypoint) >= minimum_lanes:
                return transform
        raise RuntimeError("No spawn point meets the configured lane-count requirement")

    def _route_transform_at_current_progress(self):
        waypoint = self.route_manager.route[self.route_manager.current_index]
        return carla.Transform(
            carla.Location(x=waypoint["x"], y=waypoint["y"], z=waypoint["z"] + 0.3),
            carla.Rotation(yaw=waypoint["yaw"]),
        )

    def _mark_completed_events_before_resume(self):
        if self.route_manager.progress_m <= 0.0:
            return
        for event in self.event_manager.events:
            if float(event["distance_m"]) > self.route_manager.progress_m:
                continue
            event["triggered"] = True
            event["status"] = "COMPLETED"
            self.lifecycle_log.append({
                "type": "scenario_event",
                "transition": "resumed_completed",
                "event_id": event["id"],
                "event_type": event["scenario"],
                "route_progress_m": self.route_manager.progress_m,
            })

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
        required = {"scenario_id", "scenario_name", "environment", "route", "ego"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("Scenario config missing keys: {0}".format(", ".join(missing)))
        return payload
