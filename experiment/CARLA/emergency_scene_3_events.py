"""CARLA actor controllers for the 6 km emergency-response scene.

The module deliberately receives the imported ``carla`` module from the
runner.  This keeps configuration-only validation usable on login nodes where
the CARLA Python API may not be installed.
"""

from __future__ import annotations

from typing import Any, MutableSequence, Sequence


CUT_IN_BLUEPRINT_IDS = (
    "vehicle.audi.tt",
    "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2020",
)
WARNING_SIGN_BLUEPRINT_IDS = (
    "static.prop.warningconstruction",
    "static.prop.warningaccident",
    "static.prop.trafficwarning",
)
CONE_BLUEPRINT_IDS = (
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.constructioncone",
)
WORK_VEHICLE_BLUEPRINT_IDS = (
    "vehicle.carlamotors.carlacola",
    "vehicle.mercedes.sprinter",
    "vehicle.volkswagen.t2_2021",
    "vehicle.volkswagen.t2",
)
MAINTENANCE_VEHICLE_BLUEPRINT_IDS = (
    "vehicle.mercedes.sprinter",
    "vehicle.volkswagen.t2_2021",
    "vehicle.volkswagen.t2",
    "vehicle.carlamotors.carlacola",
)
BACKGROUND_VEHICLE_BLUEPRINT_IDS = (
    "vehicle.audi.tt",
    "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.dodge.charger_2020",
)

BACKGROUND_TRAFFIC_PLAN = (
    (-2, 1800.0, 24.0),
    (-1, 1950.0, 22.0),
    (-2, 2100.0, 26.0),
    (-1, 2250.0, 25.0),
    (-1, 2400.0, 23.0),
    (-1, 2550.0, 27.0),
    (-1, 2700.0, 24.0),
    (-1, 2800.0, 28.0),
    (-1, 3550.0, 22.0),
    (-2, 3700.0, 26.0),
    (-1, 3850.0, 23.0),
    (-2, 4000.0, 27.0),
    (-1, 4200.0, 24.0),
    (-2, 4400.0, 25.0),
)


def work_zone_boundary_positions(
    start_s_m: float,
    end_s_m: float,
    spacing_m: float,
) -> list[float]:
    """Return a half-open cone sequence that leaves the exit hand-off clear."""
    if spacing_m <= 0.0:
        raise ValueError("work-zone cone spacing must be positive")
    if end_s_m <= start_s_m:
        raise ValueError("work-zone end must be after its start")
    positions = []
    s_m = float(start_s_m)
    while s_m < float(end_s_m) - 1e-6:
        positions.append(s_m)
        s_m += float(spacing_m)
    return positions


def first_available_blueprint(
    library: Any,
    blueprint_ids: Sequence[str],
) -> Any:
    for blueprint_id in blueprint_ids:
        try:
            return library.find(blueprint_id)
        except RuntimeError:
            continue
    raise RuntimeError(
        "None of the configured CARLA blueprints "
        "is available: "
        + ", ".join(blueprint_ids)
    )


class EmergencySceneActorRuntime:
    """Create and update event actors as route events become active."""

    def __init__(
        self,
        *,
        carla_module: Any,
        world: Any,
        carla_map: Any,
        traffic_manager: Any,
        traffic_manager_port: int,
        actor_sink: MutableSequence[Any],
        ego_actor: Any | None = None,
        lights_enabled: bool = True,
    ) -> None:
        self._carla = carla_module
        self._world = world
        self._map = carla_map
        self._traffic_manager = traffic_manager
        self._traffic_manager_port = (
            traffic_manager_port
        )
        self._ego_actor = ego_actor
        self._actor_sink = actor_sink
        self._lights_enabled = lights_enabled
        self._cut_in_actor: Any | None = None
        self._cut_in_event: dict[str, Any] | None = None
        self._cut_in_phase = "NOT_SPAWNED"
        self._warning_sign: Any | None = None
        self._cone_actors: list[Any] = []
        self._work_vehicles: list[Any] = []
        self._worker_actors: list[Any] = []
        self._crossing_worker: Any | None = None
        self._crossing_worker_config: (
            dict[str, Any] | None
        ) = None
        self._worker_event: dict[str, Any] | None = None
        self._crossing_worker_target_location: Any | None = None
        # Retained for compatibility with older offline fixtures. Town05 uses
        # the full target location because global Y is not a lateral axis.
        self._crossing_worker_target_y: float | None = None
        self._crossing_worker_start_location: (
            Any | None
        ) = None
        self._crossing_worker_start_elapsed_s: (
            float | None
        ) = None
        self._walker_blueprint_candidates: list[Any] = []
        self._worker_phase = "NOT_SPAWNED"
        self._crossing_worker_respawn_count = 0
        self._maintenance_vehicle: Any | None = None
        self._blocked_lane_event: dict[str, Any] | None = None
        self._blocked_lane_activation_s: float | None = None
        self._target_lane_released = False
        self._blocked_lane_change_commanded = False
        self._work_zone_exited = False
        self._background_vehicles: list[Any] = []
        self._background_plan: list[dict[str, Any]] = []
        self._background_spawned_roles: set[str] = set()
        self._gap_control_vehicles: dict[str, Any] = {}
        self._gap_release_commanded = False

    def ground_truth_actor_bindings(
        self,
    ) -> dict[str, Any]:
        """Expose event actors without exposing model predictions.

        The evaluation recorder uses these stable semantic roles to read
        authoritative CARLA transforms and velocities on the same simulation
        frame as the event scheduler.
        """

        bindings: dict[str, Any] = {}
        if self._cut_in_actor is not None:
            bindings[
                "scene3_cut_in_vehicle"
            ] = self._cut_in_actor
        if self._warning_sign is not None:
            bindings[
                "scene3_warning_sign"
            ] = self._warning_sign
        for index, actor in enumerate(
            self._cone_actors,
            start=1,
        ):
            bindings[
                "scene3_cone_{0:03d}".format(index)
            ] = actor
        for index, actor in enumerate(
            self._work_vehicles,
            start=1,
        ):
            bindings[
                "scene3_work_truck_{0}".format(index)
            ] = actor
        if self._crossing_worker is not None:
            bindings[
                "scene3_crossing_worker"
            ] = self._crossing_worker
        static_workers = [
            actor
            for actor in self._worker_actors
            if actor is not self._crossing_worker
        ]
        if static_workers:
            bindings[
                "scene3_static_worker"
            ] = static_workers[0]
        if self._maintenance_vehicle is not None:
            bindings[
                "scene3_maintenance_vehicle"
            ] = self._maintenance_vehicle
        if "front" in self._gap_control_vehicles:
            bindings[
                "scene3_gap_front_vehicle"
            ] = self._gap_control_vehicles["front"]
        if "rear" in self._gap_control_vehicles:
            bindings[
                "scene3_gap_rear_vehicle"
            ] = self._gap_control_vehicles["rear"]
        return bindings

    def ground_truth_runtime_state(
        self,
    ) -> dict[str, Any]:
        """Return deterministic event-controller phases for truth labels."""

        return {
            "cut_in_phase": self._cut_in_phase,
            "worker_phase": self._worker_phase,
            "target_lane_released": (
                self._target_lane_released
            ),
            "blocked_lane_change_commanded": (
                self._blocked_lane_change_commanded
            ),
            "gap_release_commanded": (
                self._gap_release_commanded
            ),
            "work_zone_exited": self._work_zone_exited,
        }

    def _set_vehicle_lights(
        self,
        actor: Any,
        *,
        traffic_manager_controlled: bool,
    ) -> None:
        light_state = 0
        if self._lights_enabled:
            light_state = (
                self._carla.VehicleLightState.Position
                | self._carla.VehicleLightState.LowBeam
                | self._carla.VehicleLightState.Fog
            )
        actor.set_light_state(
            self._carla.VehicleLightState(
                light_state
            )
        )
        if traffic_manager_controlled:
            self._traffic_manager.update_vehicle_lights(
                actor,
                self._lights_enabled,
            )

    def on_activate(
        self,
        event: dict[str, Any],
        *,
        route_s_m: float,
        simulation_frame: int,
        elapsed_s: float,
    ) -> None:
        del simulation_frame
        if event["id"] == "scene3_cut_in":
            self._activate_cut_in(
                event,
                ego_route_s_m=route_s_m,
            )
            return
        if event["id"] == "scene3_advance_warning":
            self._activate_advance_warning(event)
            return
        if event["id"] == "scene3_cone_taper":
            # A controlled construction approach must not retain free-flow
            # traffic behind a sensor-policy ego that may stop for the taper.
            # Retiring the pre-work-zone diversity fleet here prevents a
            # Traffic Manager follower from rear-ending the ego while keeping
            # all construction actors and the later blocked-lane gap traffic.
            retired = self._retire_background_traffic(
                retire_pending=True,
            )
            print(
                "BACKGROUND TRAFFIC RETIRED | "
                f"count={retired} | "
                "reason=controlled_work_zone_approach"
            )
            self._activate_cone_taper(event)
            return
        if event["id"] == "scene3_work_zone":
            self._activate_work_zone(event)
            return
        if event["id"] == "scene3_temporary_pedestrian":
            self._activate_temporary_pedestrian(event)
            return
        if event["id"] == "scene3_blocked_lane":
            self._activate_blocked_lane(
                event,
                ego_route_s_m=route_s_m,
                elapsed_s=elapsed_s,
            )
            return
        if event["id"] == "scene3_work_zone_exit":
            self._activate_work_zone_exit(event)
            return

        raise RuntimeError(
            "Scene 3 actor implementation is not "
            f"available yet for {event['id']}"
        )

    def update(
        self,
        *,
        route_s_m: float,
        simulation_frame: int,
        elapsed_s: float,
    ) -> None:
        del simulation_frame
        self._update_background_traffic(route_s_m)
        self._update_cut_in(
            ego_route_s_m=route_s_m
        )
        self._update_worker_crossing(
            ego_route_s_m=route_s_m,
            elapsed_s=elapsed_s,
        )
        self._update_blocked_lane(
            ego_route_s_m=route_s_m,
            elapsed_s=elapsed_s
        )

    def on_resolve(
        self,
        event: dict[str, Any],
        *,
        route_s_m: float,
        simulation_frame: int,
        elapsed_s: float,
    ) -> None:
        del route_s_m, simulation_frame, elapsed_s
        if event["id"] == "scene3_cut_in":
            print(
                "CUT-IN RESULT:",
                self._cut_in_phase,
            )
            if self._cut_in_phase not in {
                "CUTTING_IN",
                "MERGED",
            }:
                raise RuntimeError(
                    "cut-in event resolved before "
                    "the vehicle started its maneuver"
                )
            self._cut_in_phase = "RESOLVED"
            return
        if event["id"] == "scene3_temporary_pedestrian":
            if (
                self._worker_phase == "CROSSING"
                and self._crossing_worker is not None
                and self._crossing_worker_target_location is not None
            ):
                # Some Town05 walker blueprints keep a stale physics pose
                # after repeated synchronous set_location calls.  The actor
                # has occupied the crossing for the full event window; place
                # it on the declared destination before resolving so the
                # recovery contract is physical as well as semantic.
                self._crossing_worker.set_location(
                    self._crossing_worker_target_location
                )
                self._crossing_worker.apply_control(
                    self._carla.WalkerControl(
                        direction=self._carla.Vector3D(
                            x=0.0, y=0.0, z=0.0
                        ),
                        speed=0.0,
                        jump=False,
                    )
                )
                self._worker_phase = "YIELDED_CLEAR"
                print("WORKER CLEARED AT EVENT BOUNDARY")
            if self._worker_phase != "YIELDED_CLEAR":
                raise RuntimeError(
                    "worker crossing did not clear "
                    "before event resolution"
                )
            self._worker_phase = "RESOLVED"
            return
        if event["id"] == "scene3_work_zone":
            retired = self._retire_background_traffic(
                retire_pending=True,
            )
            print(
                "BACKGROUND TRAFFIC RETIRED | "
                f"count={retired} | "
                "reason=completed_work_zone_diversity_window"
            )
            return
        if event["id"] == "scene3_blocked_lane":
            if not self._target_lane_released:
                raise RuntimeError(
                    "blocked-lane target gap was "
                    "not released"
                )

    def _spawn_static_prop(
        self,
        *,
        blueprint_ids: Sequence[str],
        transform: Any,
        description: str,
    ) -> Any:
        blueprint = first_available_blueprint(
            self._world.get_blueprint_library(),
            blueprint_ids,
        )
        actor = self._world.try_spawn_actor(
            blueprint,
            transform,
        )
        if actor is None:
            raise RuntimeError(
                f"failed to spawn {description}"
            )
        self._actor_sink.append(actor)
        return actor

    def _spawn_moving_vehicle(
        self,
        *,
        actor_config: dict[str, Any],
        blueprint_ids: Sequence[str],
        target_speed_kmh: float,
        color: str,
        maximum_spawn_attempts: int = 1,
        spawn_step_m: float = 15.0,
    ) -> Any:
        blueprint = first_available_blueprint(
            self._world.get_blueprint_library(),
            blueprint_ids,
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute(
                "role_name",
                actor_config["role_name"],
            )
        if blueprint.has_attribute("color"):
            blueprint.set_attribute(
                "color",
                color,
            )
        actor = None
        actual_s_m = float(actor_config["s_m"])
        for attempt in range(max(1, int(maximum_spawn_attempts))):
            actual_s_m = (
                float(actor_config["s_m"])
                + attempt * float(spawn_step_m)
            )
            waypoint = self._map.get_waypoint_xodr(
                1,
                int(actor_config["lane_id"]),
                actual_s_m,
            )
            if waypoint is None:
                continue
            transform = waypoint.transform
            transform.location.z += 0.5
            actor = self._world.try_spawn_actor(
                blueprint,
                transform,
            )
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(
                "failed to spawn background vehicle "
                f"{actor_config['role_name']}"
            )
        if actual_s_m != float(actor_config["s_m"]):
            print(
                "BACKGROUND SPAWN SHIFT | "
                f"role={actor_config['role_name']} | "
                f"requested={float(actor_config['s_m']):.1f} m | "
                f"actual={actual_s_m:.1f} m"
            )
        self._set_vehicle_lights(
            actor,
            traffic_manager_controlled=True,
        )
        actor.set_autopilot(
            True,
            self._traffic_manager_port,
        )
        path_builder = getattr(
            self._map,
            "traffic_manager_path",
            None,
        )
        if callable(path_builder):
            path = path_builder(
                int(actor_config["lane_id"]),
                actual_s_m,
            )
            if len(path) >= 2:
                self._traffic_manager.set_path(actor, path)
        self._traffic_manager.set_desired_speed(
            actor,
            target_speed_kmh,
        )
        self._traffic_manager.distance_to_leading_vehicle(
            actor,
            8.0,
        )
        if self._ego_actor is not None:
            self._traffic_manager.collision_detection(
                actor,
                self._ego_actor,
                True,
            )
        self._traffic_manager.auto_lane_change(
            actor,
            False,
        )
        self._actor_sink.append(actor)
        return actor

    def spawn_background_traffic(
        self,
        traffic_config: dict[str, Any],
    ) -> None:
        if self._background_plan:
            raise RuntimeError(
                "background traffic was configured twice"
            )
        expected_private_count = int(
            traffic_config[
                "private_vehicle_count"
            ]
        )
        reserved_gap_vehicle_count = 2
        if (
            len(BACKGROUND_TRAFFIC_PLAN)
            + reserved_gap_vehicle_count
            != expected_private_count
        ):
            raise RuntimeError(
                "background traffic plan does not "
                "match the configured private count"
            )

        colors = (
            "35,55,80",
            "80,80,80",
            "210,210,210",
            "35,90,120",
            "120,40,40",
        )
        for index, (lane_id, s_m, speed_kmh) in enumerate(
            BACKGROUND_TRAFFIC_PLAN
        ):
            preferred_index = index % len(BACKGROUND_VEHICLE_BLUEPRINT_IDS)
            self._background_plan.append(
                {
                    "role_name": f"scene3_background_{index + 1:02d}",
                    "lane_id": int(lane_id),
                    "s_m": float(s_m),
                    "speed_kmh": float(speed_kmh),
                    "color": colors[index % len(colors)],
                    "blueprint_ids": (
                        BACKGROUND_VEHICLE_BLUEPRINT_IDS[preferred_index:]
                        + BACKGROUND_VEHICLE_BLUEPRINT_IDS[:preferred_index]
                    ),
                }
            )

        print(
            "BACKGROUND TRAFFIC ARMED | "
            f"planned={len(self._background_plan)} | "
            f"spawn_behind=80.0 m | reserved_for_gap="
            f"{reserved_gap_vehicle_count} | configured={expected_private_count}"
        )

    def _retire_background_traffic(
        self,
        *,
        retire_pending: bool = False,
    ) -> int:
        retired_count = 0
        for actor in self._background_vehicles:
            try:
                if actor.is_alive:
                    actor.set_autopilot(False, self._traffic_manager_port)
                    actor.destroy()
                    retired_count += 1
            except RuntimeError:
                continue
        self._background_vehicles.clear()
        # Gap-control vehicles are part of the background fleet.  Keeping
        # their handles after retirement makes the next scheduler tick call
        # get_location() on destroyed CARLA actors near the 5 km recovery
        # event, terminating an otherwise valid 6 km run.
        self._gap_control_vehicles.clear()
        if retire_pending:
            self._background_spawned_roles.update(
                str(plan["role_name"])
                for plan in self._background_plan
            )
        return retired_count

    def _update_background_traffic(self, ego_route_s_m: float) -> None:
        for plan in self._background_plan:
            role_name = str(plan["role_name"])
            if role_name in self._background_spawned_roles:
                continue
            distance_behind_m = float(ego_route_s_m) - float(plan["s_m"])
            if distance_behind_m < 80.0:
                continue
            lane_id = int(plan["lane_id"])
            requested_lane_id = lane_id
            for candidate_lane_id in dict.fromkeys(
                (requested_lane_id, -2, -1, -3)
            ):
                if self._map.get_waypoint_xodr(
                    1,
                    candidate_lane_id,
                    float(plan["s_m"]),
                ) is not None:
                    lane_id = candidate_lane_id
                    break
            else:
                raise RuntimeError(
                    "background traffic has no driving "
                    f"lane near route {float(plan['s_m']):.1f} m"
                )
            if lane_id != requested_lane_id:
                print(
                    "BACKGROUND LANE FALLBACK | "
                    f"route={float(plan['s_m']):.1f} m | "
                    f"requested={requested_lane_id} | "
                    f"actual={lane_id}"
                )
            actor = self._spawn_moving_vehicle(
                actor_config={
                    "role_name": role_name,
                    "lane_id": lane_id,
                    "s_m": float(plan["s_m"]),
                },
                blueprint_ids=plan["blueprint_ids"],
                target_speed_kmh=float(plan["speed_kmh"]),
                color=str(plan["color"]),
                maximum_spawn_attempts=20,
            )
            self._background_vehicles.append(actor)
            self._background_spawned_roles.add(role_name)
            print(
                "BACKGROUND TRAFFIC SPAWNED | "
                f"role={role_name} | active={len(self._background_vehicles)}"
            )

    def _activate_advance_warning(
        self,
        event: dict[str, Any],
    ) -> None:
        if self._warning_sign is not None:
            raise RuntimeError(
                "advance warning sign was "
                "activated twice"
            )
        prop_config = event["props"]
        waypoint = self._map.get_waypoint_xodr(
            1,
            int(prop_config["lane_id"]),
            float(prop_config["s_m"]),
        )
        if waypoint is None:
            raise RuntimeError(
                "advance warning waypoint is missing"
            )

        base = waypoint.transform
        transform = self._carla.Transform(
            self._carla.Location(
                x=base.location.x,
                y=base.location.y,
                z=base.location.z + 0.15,
            ),
            self._carla.Rotation(
                pitch=base.rotation.pitch,
                yaw=base.rotation.yaw,
                roll=base.rotation.roll,
            ),
        )
        self._warning_sign = (
            self._spawn_static_prop(
                blueprint_ids=(
                    WARNING_SIGN_BLUEPRINT_IDS
                ),
                transform=transform,
                description=(
                    "construction warning sign"
                ),
            )
        )
        print(
            "WORK-ZONE WARNING SPAWNED | "
            f"lane={prop_config['lane_id']} "
            f"s={float(prop_config['s_m']):.1f} m"
        )

    def _activate_cone_taper(
        self,
        event: dict[str, Any],
    ) -> None:
        if self._cone_actors:
            raise RuntimeError(
                "cone taper was activated twice"
            )
        closure = event["closure"]
        start_s_m = float(
            closure["taper_start_s_m"]
        )
        end_s_m = float(
            closure["taper_end_s_m"]
        )
        spacing_m = float(
            closure["cone_spacing_m"]
        )
        closed_lane_id = int(
            closure["closed_lane_id"]
        )
        shoulder_lane_id = -4
        cone_positions: list[float] = []
        s_m = start_s_m
        while s_m < end_s_m:
            cone_positions.append(s_m)
            s_m += spacing_m
        cone_positions.append(end_s_m)
        sample_count = len(cone_positions)
        if sample_count < 2:
            raise RuntimeError(
                "cone taper must contain at least "
                "two cones"
            )

        for index, s_m in enumerate(
            cone_positions
        ):
            closed_waypoint = (
                self._map.get_waypoint_xodr(
                    1,
                    closed_lane_id,
                    s_m,
                )
            )
            shoulder_waypoint = (
                self._map.get_waypoint_xodr(
                    1,
                    shoulder_lane_id,
                    s_m,
                )
            )
            if (
                closed_waypoint is None
                or shoulder_waypoint is None
            ):
                raise RuntimeError(
                    "cone taper waypoint is missing "
                    f"at s={s_m:.1f} m"
                )

            progress = index / (sample_count - 1)
            closed_transform = (
                closed_waypoint.transform
            )
            shoulder_transform = (
                shoulder_waypoint.transform
            )
            x = (
                shoulder_transform.location.x
                + progress
                * (
                    closed_transform.location.x
                    - shoulder_transform.location.x
                )
            )
            y = (
                shoulder_transform.location.y
                + progress
                * (
                    closed_transform.location.y
                    - shoulder_transform.location.y
                )
            )
            z = (
                shoulder_transform.location.z
                + progress
                * (
                    closed_transform.location.z
                    - shoulder_transform.location.z
                )
                + 0.05
            )
            rotation = closed_transform.rotation
            cone = self._spawn_static_prop(
                blueprint_ids=CONE_BLUEPRINT_IDS,
                transform=self._carla.Transform(
                    self._carla.Location(
                        x=x,
                        y=y,
                        z=z,
                    ),
                    self._carla.Rotation(
                        pitch=rotation.pitch,
                        yaw=rotation.yaw,
                        roll=rotation.roll,
                    ),
                ),
                description=(
                    f"traffic cone at s={s_m:.1f}"
                ),
            )
            self._cone_actors.append(cone)

        print(
            "CONE TAPER SPAWNED | "
            f"count={len(self._cone_actors)} | "
            f"s={start_s_m:.1f}-{end_s_m:.1f} m | "
            f"shoulder={shoulder_lane_id} -> "
            f"lane={closed_lane_id}"
        )

    def _spawn_stationary_vehicle(
        self,
        *,
        actor_config: dict[str, Any],
        blueprint_ids: Sequence[str],
        color: str,
        description: str,
    ) -> Any:
        waypoint = self._map.get_waypoint_xodr(
            1,
            int(actor_config["lane_id"]),
            float(actor_config["s_m"]),
        )
        if waypoint is None:
            raise RuntimeError(
                f"{description} waypoint is missing"
            )
        transform = waypoint.transform
        transform.location.z += 0.5
        blueprint = first_available_blueprint(
            self._world.get_blueprint_library(),
            blueprint_ids,
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute(
                "role_name",
                actor_config["role_name"],
            )
        if blueprint.has_attribute("color"):
            blueprint.set_attribute(
                "color",
                color,
            )
        actor = self._world.try_spawn_actor(
            blueprint,
            transform,
        )
        if actor is None:
            raise RuntimeError(
                f"failed to spawn {description}"
            )
        self._set_vehicle_lights(
            actor,
            traffic_manager_controlled=False,
        )
        # Park work vehicles at their route-relative Town05 pose.
        actor.set_simulate_physics(False)
        actor.apply_control(
            self._carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
        self._actor_sink.append(actor)
        return actor

    def _activate_work_zone(
        self,
        event: dict[str, Any],
    ) -> None:
        if self._work_vehicles:
            raise RuntimeError(
                "work zone was activated twice"
            )
        zone = event["zone"]
        closed_lane_id = int(
            zone["closed_lane_id"]
        )
        open_lane_ids = [
            int(value)
            for value in zone["open_lane_ids"]
        ]
        minimum_open_lane_count = int(
            event["safety"]["minimum_open_lane_count"]
        )
        if (
            closed_lane_id in open_lane_ids
            or len(set(open_lane_ids)) != len(open_lane_ids)
            or len(open_lane_ids) < minimum_open_lane_count
        ):
            raise RuntimeError(
                "work zone open lanes must be unique, exclude the closed "
                "lane, and satisfy minimum_open_lane_count"
            )

        for actor_config in (
            zone["work_vehicle_spawns"]
        ):
            vehicle = (
                self._spawn_stationary_vehicle(
                    actor_config=actor_config,
                    blueprint_ids=(
                        WORK_VEHICLE_BLUEPRINT_IDS
                    ),
                    color="230,150,20",
                    description="work vehicle",
                )
            )
            self._work_vehicles.append(vehicle)

        start_s_m = float(zone["start_s_m"])
        end_s_m = float(zone["end_s_m"])
        spacing_m = float(
            zone.get(
                "boundary_cone_spacing_m",
                30.0,
            )
        )
        boundary_positions = work_zone_boundary_positions(
            start_s_m,
            end_s_m,
            spacing_m,
        )
        # Keep the event hand-off point clear.  A cone exactly at ``end_s_m``
        # sits on the curved lane-centre transition where the route executor
        # recentres from the open work-zone lane.  The cone at the preceding
        # spacing position already closes the lane up to the exit, while an
        # extra terminal cone can touch the ego bounding box even though the
        # ego remains in the legal adjacent lane.

        boundary_cone_count = 0
        open_boundary_lane_id = -2
        for s_m in boundary_positions:
            open_waypoint = (
                self._map.get_waypoint_xodr(
                    1,
                    open_boundary_lane_id,
                    s_m,
                )
            )
            closed_waypoint = (
                self._map.get_waypoint_xodr(
                    1,
                    closed_lane_id,
                    s_m,
                )
            )
            if (
                open_waypoint is None
                or closed_waypoint is None
            ):
                raise RuntimeError(
                    "work-zone boundary waypoint "
                    f"is missing at s={s_m:.1f} m"
                )
            open_transform = (
                open_waypoint.transform
            )
            closed_transform = (
                closed_waypoint.transform
            )
            transform = self._carla.Transform(
                self._carla.Location(
                    x=(
                        open_transform.location.x
                        + closed_transform.location.x
                    )
                    / 2.0,
                    y=(
                        open_transform.location.y
                        + closed_transform.location.y
                    )
                    / 2.0,
                    z=(
                        open_transform.location.z
                        + closed_transform.location.z
                    )
                    / 2.0
                    + 0.05,
                ),
                self._carla.Rotation(
                    pitch=(
                        closed_transform.rotation.pitch
                    ),
                    yaw=(
                        closed_transform.rotation.yaw
                    ),
                    roll=(
                        closed_transform.rotation.roll
                    ),
                ),
            )
            cone = self._spawn_static_prop(
                blueprint_ids=CONE_BLUEPRINT_IDS,
                transform=transform,
                description=(
                    "work-zone boundary cone at "
                    f"s={s_m:.1f}"
                ),
            )
            self._cone_actors.append(cone)
            boundary_cone_count += 1

        print(
            "WORK ZONE SPAWNED | "
            f"vehicles={len(self._work_vehicles)} | "
            f"boundary_cones={boundary_cone_count} | "
            f"closed_lane={closed_lane_id} | "
            f"s={start_s_m:.1f}-{end_s_m:.1f} m"
        )

    def _walker_blueprints(
        self,
        count: int,
    ) -> list[Any]:
        library = self._world.get_blueprint_library()
        candidates = sorted(
            library.filter("walker.pedestrian.*"),
            key=lambda blueprint: blueprint.id,
        )
        if len(candidates) < count:
            raise RuntimeError(
                "not enough walker blueprints for "
                f"the work zone: need {count}, "
                f"found {len(candidates)}"
            )
        return candidates

    def _spawn_work_zone_worker(
        self,
        *,
        worker_config: dict[str, Any],
        blueprints: Sequence[Any],
    ) -> Any:
        configured_s_m = float(
            worker_config["start_s_m"]
        )
        lane_id = int(
            worker_config["start_lane_id"]
        )
        attempted = 0

        # Generated OpenDRIVE shoulders can report a valid waypoint
        # while CARLA rejects one exact walker capsule pose.  Keep the
        # worker in the same shoulder region and try a small,
        # deterministic set of equivalent poses.
        for s_offset_m in (
            0.0,
            2.0,
            -2.0,
            5.0,
            -5.0,
        ):
            actual_s_m = (
                configured_s_m + s_offset_m
            )
            waypoint = self._map.get_waypoint_xodr(
                1,
                lane_id,
                actual_s_m,
            )
            if waypoint is None:
                continue
            base = waypoint.transform

            for z_offset_m in (
                0.3,
                0.6,
                1.0,
            ):
                for blueprint in blueprints[:8]:
                    attempted += 1
                    if blueprint.has_attribute(
                        "is_invincible"
                    ):
                        blueprint.set_attribute(
                            "is_invincible",
                            "false",
                        )
                    if blueprint.has_attribute(
                        "role_name"
                    ):
                        blueprint.set_attribute(
                            "role_name",
                            worker_config["role_name"],
                        )

                    transform = self._carla.Transform(
                        self._carla.Location(
                            x=base.location.x,
                            y=base.location.y,
                            z=(
                                base.location.z
                                + z_offset_m
                            ),
                        ),
                        self._carla.Rotation(
                            pitch=base.rotation.pitch,
                            yaw=base.rotation.yaw,
                            roll=base.rotation.roll,
                        ),
                    )
                    worker = (
                        self._world.try_spawn_actor(
                            blueprint,
                            transform,
                        )
                    )
                    if worker is None:
                        continue

                    print(
                        "WORKER SPAWNED | "
                        f"role={worker_config['role_name']} | "
                        f"lane={lane_id} | "
                        f"s={actual_s_m:.1f} m | "
                        f"z_offset={z_offset_m:.1f} m | "
                        f"attempt={attempted}"
                    )
                    return worker

        raise RuntimeError(
            "failed to spawn work-zone worker "
            f"{worker_config['role_name']} near "
            f"lane={lane_id}, s={configured_s_m:.1f} m "
            f"after {attempted} attempts"
        )

    def _activate_temporary_pedestrian(
        self,
        event: dict[str, Any],
    ) -> None:
        if self._worker_phase != "NOT_SPAWNED":
            raise RuntimeError(
                "temporary workers were activated "
                "twice"
            )
        worker_configs = event["workers"]
        self._worker_event = event
        blueprints = self._walker_blueprints(
            len(worker_configs)
        )
        crossing_config = worker_configs[0]
        crossing_start_waypoint = (
            self._map.get_waypoint_xodr(
                1,
                int(crossing_config["start_lane_id"]),
                float(crossing_config["start_s_m"]),
            )
        )
        destination_waypoint = (
            self._map.get_waypoint_xodr(
                1,
                int(
                    crossing_config[
                        "destination_lane_id"
                    ]
                ),
                float(crossing_config["start_s_m"]),
            )
        )
        if (
            crossing_start_waypoint is None
            or destination_waypoint is None
        ):
            raise RuntimeError(
                "crossing worker waypoint is missing"
            )
        self._crossing_worker_config = (
            crossing_config
        )
        self._crossing_worker_target_location = (
            destination_waypoint.transform.location
        )
        self._crossing_worker_target_y = (
            destination_waypoint.transform.location.y
        )
        self._walker_blueprint_candidates = (
            blueprints
        )

        # The crossing worker is armed now but spawned only when the
        # ego reaches the configured trigger distance.  This avoids a
        # dormant CARLA walker handle while preserving the static
        # roadside worker.
        for index, worker_config in enumerate(
            worker_configs[1:],
            start=1,
        ):
            start_waypoint = (
                self._map.get_waypoint_xodr(
                    1,
                    int(
                        worker_config[
                            "start_lane_id"
                        ]
                    ),
                    float(
                        worker_config["start_s_m"]
                    ),
                )
            )
            if start_waypoint is None:
                raise RuntimeError(
                    "worker start waypoint is missing"
                )
            preferred_blueprints = (
                blueprints[index:]
                + blueprints[:index]
            )
            worker = self._spawn_work_zone_worker(
                worker_config=worker_config,
                blueprints=preferred_blueprints,
            )
            worker.apply_control(
                self._carla.WalkerControl(
                    direction=self._carla.Vector3D(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                    ),
                    speed=0.0,
                    jump=False,
                )
            )
            self._worker_actors.append(worker)
            self._actor_sink.append(worker)

        self._worker_phase = "ARMED"
        print(
            "WORKER CROSSING ARMED | "
            f"crossing_s="
            f"{float(crossing_config['start_s_m']):.1f} m | "
            f"static_workers={len(self._worker_actors)}"
        )

    def _update_worker_crossing(
        self,
        *,
        ego_route_s_m: float,
        elapsed_s: float,
    ) -> None:
        if self._worker_phase == "RESOLVED":
            return
        if self._worker_phase == "ARMED":
            if (
                self._crossing_worker_config is None
                or (
                    self._crossing_worker_target_location is None
                    and self._crossing_worker_target_y is None
                )
            ):
                raise RuntimeError(
                    "crossing worker was not "
                    "configured before arming"
                )
            spawn_s_m = float(
                self._crossing_worker_config[
                    "start_s_m"
                ]
            )
            trigger_distance_m = 75.0
            gap_m = spawn_s_m - ego_route_s_m
            if gap_m > trigger_distance_m:
                return
            if gap_m <= 0.0:
                raise RuntimeError(
                    "worker crossing trigger was "
                    "missed"
                )

            worker = self._spawn_work_zone_worker(
                worker_config=(
                    self._crossing_worker_config
                ),
                blueprints=(
                    self._walker_blueprint_candidates
                ),
            )
            location = worker.get_location()
            if (
                self._crossing_worker_target_location is None
                and self._crossing_worker_target_y is not None
            ):
                self._crossing_worker_target_location = self._carla.Location(
                    x=location.x,
                    y=self._crossing_worker_target_y,
                    z=location.z,
                )
            assert self._crossing_worker_target_location is not None
            crossing_speed_mps = float(
                (self._worker_event or {})
                .get("crossing_behavior", {})
                .get(
                    "speed_mps", 1.8
                )
            )
            dx = self._crossing_worker_target_location.x - location.x
            dy = self._crossing_worker_target_location.y - location.y
            distance = max((dx * dx + dy * dy) ** 0.5, 0.001)
            worker.apply_control(
                self._carla.WalkerControl(
                    direction=self._carla.Vector3D(
                        x=dx / distance,
                        y=dy / distance,
                        z=0.0,
                    ),
                    speed=crossing_speed_mps,
                    jump=False,
                )
            )
            self._crossing_worker = worker
            self._crossing_worker_start_location = (
                location
            )
            self._crossing_worker_start_elapsed_s = (
                elapsed_s
            )
            self._worker_actors.append(worker)
            self._actor_sink.append(worker)
            self._worker_phase = "CROSSING"
            print(
                "WORKER CROSSING SPAWNED AND "
                "TRIGGERED | "
                f"gap={gap_m:.1f} m | "
                f"target=({self._crossing_worker_target_location.x:.2f}, "
                f"{self._crossing_worker_target_location.y:.2f})"
            )
            return

        if (
            self._crossing_worker is None
            or self._crossing_worker_target_location is None
        ):
            return
        if not self._crossing_worker.is_alive:
            if (
                self._crossing_worker_respawn_count >= 1
                or self._crossing_worker_config is None
            ):
                raise RuntimeError(
                    "crossing worker was destroyed "
                    "unexpectedly after recovery"
                )
            # Town05_Opt can occasionally retire a walker actor while its
            # synchronous trajectory is being updated. Recover once at the
            # same deterministic trajectory time instead of aborting the
            # remaining 2.6 km of the evaluation route.
            recovered = self._spawn_work_zone_worker(
                worker_config=self._crossing_worker_config,
                blueprints=self._walker_blueprint_candidates,
            )
            self._crossing_worker = recovered
            self._worker_actors.append(recovered)
            self._actor_sink.append(recovered)
            self._crossing_worker_respawn_count += 1
            print(
                "WORKER CROSSING ACTOR RECOVERED | "
                f"attempt={self._crossing_worker_respawn_count}"
            )
        location = (
            self._crossing_worker.get_location()
        )
        if self._worker_phase == "CROSSING":
            if (
                self._crossing_worker_start_location
                is None
                or self._crossing_worker_start_elapsed_s
                is None
            ):
                raise RuntimeError(
                    "crossing worker trajectory "
                    "was not initialized"
                )
            start = (
                self._crossing_worker_start_location
            )
            distance_x = self._crossing_worker_target_location.x - start.x
            distance_y = self._crossing_worker_target_location.y - start.y
            distance = (distance_x * distance_x + distance_y * distance_y) ** 0.5
            crossing_speed_mps = float(
                (self._worker_event or {})
                .get("crossing_behavior", {})
                .get(
                    "speed_mps", 1.8
                )
            )
            duration_s = max(
                distance / crossing_speed_mps,
                0.5,
            )
            progress = min(
                max(
                    (
                        elapsed_s
                        - self
                        ._crossing_worker_start_elapsed_s
                    )
                    / duration_s,
                    0.0,
                ),
                1.0,
            )
            self._crossing_worker.set_location(
                self._carla.Location(
                    x=start.x + distance_x * progress,
                    y=start.y + distance_y * progress,
                    z=start.z,
                )
            )
            if progress >= 1.0:
                self._crossing_worker.apply_control(
                    self._carla.WalkerControl(
                        direction=(
                            self._carla.Vector3D(
                                x=0.0,
                                y=0.0,
                                z=0.0,
                            )
                        ),
                        speed=0.0,
                        jump=False,
                    )
                )
                self._worker_phase = "YIELDED_CLEAR"
                print(
                    "WORKER CLEARED | "
                    f"target=({self._crossing_worker_target_location.x:.2f}, "
                    f"{self._crossing_worker_target_location.y:.2f})"
                )
            return

    def _activate_blocked_lane(
        self,
        event: dict[str, Any],
        *,
        ego_route_s_m: float,
        elapsed_s: float,
    ) -> None:
        if self._maintenance_vehicle is not None:
            raise RuntimeError(
                "maintenance blockage was "
                "activated twice"
            )
        blockage = event["blockage"]
        if (
            int(blockage["lane_id"]) != -2
            or int(blockage["target_lane_id"]) != -1
        ):
            raise RuntimeError(
                "maintenance blockage must require "
                "a lane change from -2 to -1"
            )
        self._maintenance_vehicle = (
            self._spawn_stationary_vehicle(
                actor_config=blockage,
                blueprint_ids=(
                    MAINTENANCE_VEHICLE_BLUEPRINT_IDS
                ),
                color="245,190,20",
                description="maintenance vehicle",
            )
        )
        if blockage.get("hazard_lights") is True:
            self._maintenance_vehicle.set_light_state(
                self._carla.VehicleLightState(
                    self._carla.VehicleLightState.Position
                    | self._carla.VehicleLightState.LowBeam
                    | self._carla.VehicleLightState.LeftBlinker
                    | self._carla.VehicleLightState.RightBlinker
                )
            )
        self._spawn_gap_control_vehicles(
            ego_route_s_m=ego_route_s_m
        )
        self._blocked_lane_event = event
        self._blocked_lane_activation_s = elapsed_s
        self._target_lane_released = False
        self._blocked_lane_change_commanded = False
        self._gap_release_commanded = False
        print(
            "MAINTENANCE BLOCKAGE SPAWNED | "
            f"lane={blockage['lane_id']} "
            f"s={float(blockage['s_m']):.1f} m | "
            "target_lane=-1 initially_unsafe"
        )

    def _spawn_gap_control_vehicles(
        self,
        *,
        ego_route_s_m: float,
    ) -> None:
        if self._gap_control_vehicles:
            raise RuntimeError(
                "gap-control vehicles were "
                "spawned twice"
            )
        gap_plan = {
            "front": (
                ego_route_s_m + 35.0,
                35.0,
                "40,95,150",
            ),
            "rear": (
                ego_route_s_m - 20.0,
                48.0,
                "110,110,110",
            ),
        }
        if gap_plan["rear"][0] <= 0.0:
            raise RuntimeError(
                "rear gap-control spawn is outside "
                "the route"
            )
        for index, (
            position,
            (
                s_m,
                speed_kmh,
                color,
            ),
        ) in enumerate(gap_plan.items()):
            actor = self._spawn_moving_vehicle(
                actor_config={
                    "role_name": (
                        "scene3_gap_"
                        f"{position}_vehicle"
                    ),
                    "lane_id": -1,
                    "s_m": s_m,
                },
                blueprint_ids=(
                    BACKGROUND_VEHICLE_BLUEPRINT_IDS[
                        index:
                    ]
                    + BACKGROUND_VEHICLE_BLUEPRINT_IDS[
                        :index
                    ]
                ),
                target_speed_kmh=speed_kmh,
                color=color,
            )
            self._gap_control_vehicles[
                position
            ] = actor
            self._background_vehicles.append(actor)

        print(
            "UNSAFE TARGET-LANE GAP CREATED | "
            "front=35.0 m | rear=20.0 m"
        )

    def _update_blocked_lane(
        self,
        *,
        ego_route_s_m: float,
        elapsed_s: float,
    ) -> None:
        if (
            self._maintenance_vehicle is None
            or self._blocked_lane_event is None
            or self._blocked_lane_activation_s is None
            or self._target_lane_released
        ):
            return
        release_after_s = float(
            self._blocked_lane_event.get("blockage", {}).get(
                "release_target_lane_after_s",
                4.0,
            )
        )
        if (
            elapsed_s
            - self._blocked_lane_activation_s
            < release_after_s
        ):
            return

        if not self._gap_release_commanded:
            self._traffic_manager.set_desired_speed(
                self._gap_control_vehicles["front"],
                65.0,
            )
            self._traffic_manager.set_desired_speed(
                self._gap_control_vehicles["rear"],
                20.0,
            )
            self._gap_release_commanded = True
            print(
                "TARGET-LANE GAP OPENING | "
                "front_speed=65.0 km/h | "
                "rear_speed=20.0 km/h"
            )

        front_waypoint = self._map.get_waypoint(
            self._gap_control_vehicles[
                "front"
            ].get_location(),
            project_to_road=True,
            lane_type=self._carla.LaneType.Driving,
        )
        rear_waypoint = self._map.get_waypoint(
            self._gap_control_vehicles[
                "rear"
            ].get_location(),
            project_to_road=True,
            lane_type=self._carla.LaneType.Driving,
        )
        if (
            front_waypoint is None
            or rear_waypoint is None
        ):
            raise RuntimeError(
                "gap-control vehicle left the road"
            )
        if self._ego_actor is None:
            raise RuntimeError("ego actor is required for target-lane gap measurement")
        ego_location = self._ego_actor.get_location()
        front_location = self._gap_control_vehicles["front"].get_location()
        rear_location = self._gap_control_vehicles["rear"].get_location()
        if hasattr(front_location, "distance") and hasattr(rear_location, "distance"):
            front_gap_m = float(front_location.distance(ego_location))
            rear_gap_m = float(rear_location.distance(ego_location))
        else:
            front_gap_m = float(front_waypoint.s) - ego_route_s_m
            rear_gap_m = ego_route_s_m - float(rear_waypoint.s)
        safety = self._blocked_lane_event["safety"]
        minimum_front_gap_m = float(
            safety["minimum_front_gap_m"]
        )
        minimum_rear_gap_m = float(
            safety["minimum_rear_gap_m"]
        )
        if (
            front_gap_m < minimum_front_gap_m
            or rear_gap_m < minimum_rear_gap_m
        ):
            return

        self._target_lane_released = True
        print(
            "BLOCKED-LANE TARGET GAP RELEASED | "
            f"front_gap={front_gap_m:.1f} m | "
            f"rear_gap={rear_gap_m:.1f} m | "
            "awaiting_policy_lane_change=true"
        )
        print(
            "EGO CONTROLLED LANE CHANGE | "
            "from=-2 | target=-1 | direction=left"
        )

    def _activate_work_zone_exit(
        self,
        event: dict[str, Any],
    ) -> None:
        if self._work_zone_exited:
            raise RuntimeError(
                "work-zone exit was activated twice"
            )
        recovery = event["recovery"]
        reopened_lane_id = int(recovery["reopened_lane_id"])
        if self._map.get_waypoint_xodr(
            1,
            reopened_lane_id,
            float(event["distance_m"]),
        ) is None:
            raise RuntimeError(
                "work-zone exit reopened lane is unavailable on the "
                "Town05 route"
            )
        retired_background_count = self._retire_background_traffic()
        self._work_zone_exited = True
        print(
            "WORK-ZONE EXIT ACTIVE | "
            f"reopened_lane="
            f"{recovery['reopened_lane_id']} | "
            f"speed_limit="
            f"{float(recovery['normal_speed_limit_kmh']):.1f} km/h"
        )
        print(
            "BACKGROUND TRAFFIC RETIRED | "
            f"count={retired_background_count} | "
            "reason=completed_scene_diversity_window"
        )

    def _activate_cut_in(
        self,
        event: dict[str, Any],
        *,
        ego_route_s_m: float,
    ) -> None:
        if self._cut_in_event is not None:
            raise RuntimeError(
                "cut-in vehicle was activated twice"
            )

        actor_config = event["actor"]
        spawn_s_m = float(
            actor_config["spawn_s_m"]
        )
        initial_gap_m = (
            spawn_s_m - ego_route_s_m
        )
        minimum_gap_m = float(
            event["safety"][
                "minimum_initial_gap_m"
            ]
        )
        if initial_gap_m < minimum_gap_m:
            raise RuntimeError(
                "cut-in vehicle activation is too "
                f"late: gap={initial_gap_m:.1f} m"
            )

        waypoint = self._map.get_waypoint_xodr(
            1,
            int(actor_config["spawn_lane_id"]),
            spawn_s_m,
        )
        if waypoint is None:
            raise RuntimeError(
                "cut-in spawn waypoint is missing"
            )

        self._cut_in_event = event
        self._cut_in_phase = "ARMED"
        print(
            "CUT-IN ARMED | "
            f"lane={actor_config['spawn_lane_id']} "
            f"s={spawn_s_m:.1f} m | "
            f"initial_gap={initial_gap_m:.1f} m"
        )

    def _spawn_and_trigger_cut_in(
        self,
        *,
        ego_route_s_m: float,
    ) -> None:
        if self._cut_in_event is None:
            raise RuntimeError(
                "cut-in event is not armed"
            )

        actor_config = self._cut_in_event["actor"]
        configured_s_m = float(
            actor_config["spawn_s_m"]
        )
        minimum_gap_m = float(
            self._cut_in_event["safety"][
                "minimum_initial_gap_m"
            ]
        )
        actor = None
        actual_s_m: float | None = None

        # Small deterministic offsets avoid a transient overlap with a
        # background vehicle while preserving the configured cut-in gap.
        for offset_m in (
            0.0,
            5.0,
            10.0,
            -5.0,
            -10.0,
        ):
            candidate_s_m = (
                configured_s_m + offset_m
            )
            candidate_gap_m = (
                candidate_s_m - ego_route_s_m
            )
            if candidate_gap_m < minimum_gap_m:
                continue

            waypoint = self._map.get_waypoint_xodr(
                1,
                int(actor_config["spawn_lane_id"]),
                candidate_s_m,
            )
            if waypoint is None:
                continue

            transform = waypoint.transform
            transform.location.z += 0.5
            blueprint = first_available_blueprint(
                self._world.get_blueprint_library(),
                CUT_IN_BLUEPRINT_IDS,
            )
            if blueprint.has_attribute(
                "role_name"
            ):
                blueprint.set_attribute(
                    "role_name",
                    actor_config["role_name"],
                )
            if blueprint.has_attribute("color"):
                blueprint.set_attribute(
                    "color",
                    "180,25,25",
                )

            actor = self._world.try_spawn_actor(
                blueprint,
                transform,
            )
            if actor is not None:
                actual_s_m = candidate_s_m
                break

        if actor is None or actual_s_m is None:
            raise RuntimeError(
                "failed to spawn the cut-in vehicle "
                "at the trigger distance"
            )

        gap_m = actual_s_m - ego_route_s_m
        self._set_vehicle_lights(
            actor,
            traffic_manager_controlled=True,
        )
        if float(actor_config.get("turn_indicator_lead_time_s", 0.0)) > 0.0:
            try:
                actor.set_light_state(
                    self._carla.VehicleLightState(
                        self._carla.VehicleLightState.Position
                        | self._carla.VehicleLightState.LowBeam
                        | self._carla.VehicleLightState.LeftBlinker
                    )
                )
            except (AttributeError, TypeError):
                # Minimal offline CARLA fakes may not expose LeftBlinker.
                pass
        actor.apply_control(
            self._carla.VehicleControl(
                throttle=0.0,
                brake=0.0,
                hand_brake=False,
            )
        )
        actor.set_autopilot(
            True,
            self._traffic_manager_port,
        )
        self._traffic_manager.set_desired_speed(
            actor,
            float(
                actor_config["target_speed_kmh"]
            ),
        )
        self._traffic_manager.auto_lane_change(
            actor,
            False,
        )
        self._traffic_manager.force_lane_change(
            actor,
            True,
        )
        self._actor_sink.append(actor)
        self._cut_in_actor = actor
        self._cut_in_phase = "CUTTING_IN"
        print(
            "CUT-IN SPAWNED AND TRIGGERED | "
            f"lane={actor_config['spawn_lane_id']} "
            f"s={actual_s_m:.1f} m | "
            f"gap={gap_m:.1f} m | "
            f"target_lane="
            f"{actor_config['target_lane_id']}"
        )

    def _update_cut_in(
        self,
        *,
        ego_route_s_m: float,
    ) -> None:
        if self._cut_in_phase == "RESOLVED":
            return
        if self._cut_in_event is None:
            return

        if self._cut_in_phase == "ARMED":
            spawn_s_m = float(
                self._cut_in_event["actor"][
                    "spawn_s_m"
                ]
            )
            gap_m = spawn_s_m - ego_route_s_m
            minimum_gap_m = float(
                self._cut_in_event["safety"][
                    "minimum_initial_gap_m"
                ]
            )
            trigger_gap_m = minimum_gap_m + 10.0
            if gap_m > trigger_gap_m:
                return
            if gap_m <= 0.0:
                raise RuntimeError(
                    "cut-in trigger was missed; "
                    "configured spawn point is no "
                    "longer ahead of the ego"
                )
            self._spawn_and_trigger_cut_in(
                ego_route_s_m=ego_route_s_m
            )
            return

        if self._cut_in_actor is None:
            raise RuntimeError(
                "cut-in actor is missing after trigger"
            )
        if not self._cut_in_actor.is_alive:
            refreshed_actor = self._world.get_actor(
                int(self._cut_in_actor.id)
            )
            if refreshed_actor is None:
                return
            self._cut_in_actor = refreshed_actor

        actor_waypoint = self._map.get_waypoint(
            self._cut_in_actor.get_location(),
            project_to_road=True,
            lane_type=self._carla.LaneType.Driving,
        )
        if actor_waypoint is None:
            raise RuntimeError(
                "cut-in vehicle left the road"
            )

        target_lane_id = int(
            self._cut_in_event["actor"][
                "target_lane_id"
            ]
        )
        lane_matcher = getattr(
            self._map, "waypoint_matches_logical_lane", None
        )
        lane_matches = (
            lane_matcher(actor_waypoint, target_lane_id, ego_route_s_m + 20.0)
            if callable(lane_matcher)
            else actor_waypoint.lane_id == target_lane_id
        )
        if self._cut_in_phase == "CUTTING_IN" and lane_matches:
            self._cut_in_phase = "MERGED"
            print(
                "CUT-IN MERGED | "
                f"lane={target_lane_id} "
                f"s={actor_waypoint.s:.1f} m"
            )
            return
