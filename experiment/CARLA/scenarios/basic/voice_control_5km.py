"""Continuous, voice-command demonstration for the basic competition track."""

import json
import math
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
        self.active_command_id = None
        self.turn_executions = {}
        self.turn_commands = {
            item["id"]: str(item.get("action", ""))
            for item in self.config["commands"]
            if str(item.get("action", "")).startswith("turn_")
        }
        self.goal_tolerance_m = float(self.config["route"].get("goal_tolerance_m", 15.0))
        self.max_route_error_m = float(
            self.config["route"].get("max_route_error_m", 12.0)
        )
        self.route_error_grace_frames = int(
            self.config["route"].get("route_error_grace_frames", 60)
        )
        self.resume_progress_m = float(
            self.config["route"].get("resume_progress_m", 0.0)
        )
        self._route_error_frames = 0
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
        if self.resume_progress_m > 0.0:
            self.route_manager.seek(self.resume_progress_m)
            point = self.route[self.route_manager.current_index]
            spawn_point = carla.Transform(
                carla.Location(
                    x=point["x"], y=point["y"], z=point["z"] + 1.0
                ),
                carla.Rotation(yaw=point["yaw"]),
            )
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
        policy = VoiceSchedulePolicy(
            self.config["commands"],
            default_speed_kmh,
            parser_model_path,
            parser_device,
            prefer_configured_execution=True,
        )
        if self.resume_progress_m > 0.0:
            policy.resume_to(self.resume_progress_m)
            self.emitted_command_ids.update(policy.emitted_command_ids)
        return policy

    def get_policy_context(self):
        velocity = self.ego_vehicle.get_velocity()
        speed_mps = math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )
        route_lookahead_m = max(12.0, min(26.0, 10.0 + 0.85 * speed_mps))
        turn_lookahead_m = max(8.0, min(12.0, 7.0 + 0.30 * speed_mps))
        turn_completion_progress_m = None
        for directive in self.route_manager.applied_directives:
            if str(directive.get("action", "")) in {"turn_left", "turn_right", "u_turn"}:
                turn_completion_progress_m = float(
                    directive.get("applied_distance_m", 0.0)
                ) + 150.0
                break
        return {
            "progress_m": self.route_manager.progress_m,
            "simulation_time_s": self.metrics["simulation_time"],
            # This is deliberately longer than the controller fallback but
            # shorter than Town04's tight radius curves. The point is written
            # to ControlDecision JSON so the consumer holds the planned branch
            # through a multi-exit junction without cutting across a curve.
            "route_target": self.route_manager.target_point(route_lookahead_m),
            "route_reference": self.route_manager.target_point(0.0),
            "turn_route_target": self.route_manager.target_point(turn_lookahead_m),
            "route_lookahead_m": round(route_lookahead_m, 3),
            "turn_lookahead_m": round(turn_lookahead_m, 3),
            # Route geometry is refreshed on every scene-decision tick and
            # persisted in control_decision.json. The controller therefore
            # follows the selected junction branch without reading the route
            # manager directly.
            "turn_uses_local_branch": False,
            "turn_completion_progress_m": turn_completion_progress_m,
        }

    def report_intent(self, intent):
        self.active_action = str(intent.get("action", "keep_lane"))
        command_id = intent.get("command_id")
        self.active_command_id = command_id
        if command_id in self.command_ids:
            self.emitted_command_ids.add(command_id)
        if command_id in self.turn_commands:
            self._update_turn_execution(command_id)

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
            elif not all(
                self.turn_executions.get(command_id, {}).get("completed", False)
                for command_id in self.turn_commands
            ):
                self.failure("required_intersection_turn_not_completed")
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
            "route_cross_track_error_m": round(
                self.route_manager.cross_track_error_m, 3
            ),
            "emitted_command_ids": sorted(self.emitted_command_ids),
            "applied_route_directives": self.route_manager.applied_directives,
            "route_preflight": dict(self.route_preflight),
            "turn_executions": dict(self.turn_executions),
        }

    def get_ego_vehicle(self):
        return self.ego_vehicle

    def _update_route_progress(self):
        if self.ego_vehicle is None:
            return
        progress_m = self.route_manager.update(self.ego_vehicle)
        self.metrics["route_progress_m"] = round(progress_m, 3)
        self.metrics["route_length_m"] = round(self.route_manager.route_length_m, 3)
        self.metrics["route_cross_track_error_m"] = round(
            self.route_manager.cross_track_error_m, 3
        )
        if self.route_manager.cross_track_error_m > self.max_route_error_m:
            self._route_error_frames += 1
        else:
            self._route_error_frames = 0
        if (
            not self._finished
            and self._route_error_frames >= self.route_error_grace_frames
        ):
            self.failure("ego_departed_planned_route")

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
        if not bool(self.config["route"].get("strict_start_spawn", False)):
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
            waypoint = self._select_initial_lane(waypoint)
            if waypoint is None:
                rejected["lane"] += 1
                continue
            transform = carla.Transform(
                carla.Location(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    # Generated OpenDRIVE roads report a centerline at z=0.
                    # Leave clearance for CARLA physics to settle the chassis.
                    z=waypoint.transform.location.z + 1.0,
                ),
                waypoint.transform.rotation,
            )
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
            turn_evidence = self._route_turn_evidence(
                route, self.route_manager.applied_directives
            )
            if len(turn_evidence) != len([
                item for item in self.route_manager.applied_directives
                if str(item.get("action", "")).startswith("turn_")
            ]):
                rejected["directive"] += 1
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
                "turn_evidence": turn_evidence,
            }
            return spawn_transform
        raise RuntimeError(
            "No spawn point satisfies lane, route-directive, and route-continuity requirements: {0}".format(
                rejected
            )
        )

    def _select_initial_lane(self, waypoint):
        """Honor a stable lane rank when the scenario needs both lane changes."""
        requested = self.config["route"].get("initial_lane_from_right")
        if requested is None:
            return waypoint
        requested = max(1, int(requested))
        lanes = [waypoint]
        candidate = waypoint.get_right_lane()
        while self._is_same_direction_driving_lane(waypoint, candidate):
            lanes.insert(0, candidate)
            candidate = candidate.get_right_lane()
        candidate = waypoint.get_left_lane()
        while self._is_same_direction_driving_lane(waypoint, candidate):
            lanes.append(candidate)
            candidate = candidate.get_left_lane()
        return lanes[requested - 1] if requested <= len(lanes) else None

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

    def _update_turn_execution(self, command_id):
        """Require an actual junction traversal into a differently aligned road."""
        if self.ego_vehicle is None:
            return
        waypoint = self.world.get_map().get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return
        yaw = float(self.ego_vehicle.get_transform().rotation.yaw)
        state = self.turn_executions.get(command_id)
        if state is None:
            state = {
                "action": self.turn_commands[command_id],
                "start_road_id": int(waypoint.road_id),
                "start_yaw_deg": round(yaw, 3),
                "entered_junction": bool(waypoint.is_junction),
                "exited_junction": False,
                "final_road_id": int(waypoint.road_id),
                "signed_heading_change_deg": 0.0,
                "completed": False,
            }
            self.turn_executions[command_id] = state
        state["entered_junction"] = bool(
            state["entered_junction"] or waypoint.is_junction
        )
        if state["entered_junction"] and not waypoint.is_junction:
            state["exited_junction"] = True
        state["final_road_id"] = int(waypoint.road_id)
        heading_change = (
            yaw - float(state["start_yaw_deg"]) + 180.0
        ) % 360.0 - 180.0
        state["signed_heading_change_deg"] = round(heading_change, 3)
        direction_ok = (
            heading_change <= -45.0
            if state["action"] == "turn_left"
            else heading_change >= 45.0
        )
        state["completed"] = bool(
            state["entered_junction"]
            and state["exited_junction"]
            and state["final_road_id"] != state["start_road_id"]
            and direction_ok
        )

    @staticmethod
    def _route_turn_evidence(route, directives):
        """Prove configured turns change roads and heading in route geometry."""
        evidence = []
        for directive in directives:
            action = str(directive.get("action", ""))
            if action not in ("turn_left", "turn_right"):
                continue
            applied_m = float(directive.get("applied_distance_m", 0.0))
            entry = min(route, key=lambda item: abs(item["distance_m"] - applied_m))
            exits = [
                item for item in route
                if applied_m + 40.0 <= item["distance_m"] <= applied_m + 160.0
                and not item.get("is_junction", False)
                and item.get("road_id") != entry.get("road_id")
            ]
            if not exits:
                continue
            exit_point = exits[0]
            delta = (
                float(exit_point["yaw"]) - float(entry["yaw"]) + 180.0
            ) % 360.0 - 180.0
            direction_ok = delta <= -45.0 if action == "turn_left" else delta >= 45.0
            if not direction_ok:
                continue
            evidence.append({
                "id": directive.get("id"),
                "action": action,
                "entry_distance_m": applied_m,
                "entry_road_id": entry.get("road_id"),
                "exit_road_id": exit_point.get("road_id"),
                "heading_change_deg": round(delta, 3),
                "junction_observed": any(
                    item.get("is_junction", False)
                    for item in route
                    if applied_m <= item["distance_m"] <= exit_point["distance_m"]
                ),
            })
        return evidence

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
    def _is_same_direction_driving_lane(reference, candidate):
        return bool(
            candidate is not None
            and candidate.lane_type == carla.LaneType.Driving
            and candidate.road_id == reference.road_id
            and candidate.lane_id * reference.lane_id > 0
        )

    def _is_illegal_lane_invasion(self, event):
        if self.active_action in (
            "lane_change_left", "lane_change_right", "turn_left", "turn_right",
        ):
            return False
        markings = " ".join(event.get("markings", [])).lower()
        if not any(token in markings for token in ("solid", "double", "curb")):
            return False
        waypoint = self.world.get_map().get_waypoint(
            self.ego_vehicle.get_location(), project_to_road=True
        )
        # CARLA may report a solid connector marking while a planned turn is
        # traversing an intersection. That is not a lane-departure violation.
        if waypoint is None or bool(getattr(waypoint, "is_junction", False)):
            return waypoint is None
        transform = self.ego_vehicle.get_transform()
        lane_transform = waypoint.transform
        lane_yaw = math.radians(lane_transform.rotation.yaw)
        dx = transform.location.x - lane_transform.location.x
        dy = transform.location.y - lane_transform.location.y
        lateral_error_m = dx * -math.sin(lane_yaw) + dy * math.cos(lane_yaw)
        vehicle_half_width_m = float(self.ego_vehicle.bounding_box.extent.y)
        lane_half_width_m = 0.5 * float(waypoint.lane_width)
        # The invasion sensor can emit a Solid event at a Broken/Solid marking
        # transition even when the vehicle body remains inside the lane. Only
        # classify it as illegal when the body geometrically reaches the edge.
        return (
            abs(lateral_error_m) + vehicle_half_width_m
            >= lane_half_width_m - 0.05
        )

    @staticmethod
    def _load_config(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        required = {"environment", "route", "commands", "success"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("Scenario config missing keys: {0}".format(", ".join(missing)))
        return payload
