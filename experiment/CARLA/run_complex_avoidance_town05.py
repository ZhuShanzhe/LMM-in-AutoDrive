"""Run the Scene 2 environment on packaged CARLA Town05 assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_bootstrap import setup_carla_api
from evaluation.camera import ExperimentCamera
from scenarios.complex.town05_scene2 import (
    ActorRegistry,
    DeterministicSceneEvents,
    RouteProgressTracker,
    TownTrafficFlow,
    build_repeated_route,
    speed_kmh,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "scene_2_town05_runtime.json"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "scene2_town05"
EGO_BLUEPRINTS = (
    "vehicle.lincoln.mkz_2020",
    "vehicle.tesla.model3",
    "vehicle.audi.etron",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the competition Scene 2 route on Town05 with stable "
            "Traffic Manager flow and deterministic special actors."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--duration",
        type=float,
        default=90.0,
        help="Simulation seconds; 0 runs until 8 km route completion.",
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
    )
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--video-overlay", action="store_true")
    parser.add_argument(
        "--start-progress-m",
        type=float,
        default=0.0,
        help="Diagnostic start along the route; normal runs use 0.",
    )
    parser.add_argument(
        "--external-ego-control",
        action="store_true",
        help="Spawn and instrument ego but do not apply preview-agent control.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the runtime contract without connecting to CARLA.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "scene_2_town05_runtime/v1":
        raise ValueError("unsupported Town05 Scene 2 config schema")
    if payload.get("map") != "Town05_Opt":
        raise ValueError("Town05 Scene 2 must use Town05_Opt")
    route = payload["route"]
    if float(route["target_length_m"]) < 8000.0:
        raise ValueError("Scene 2 route must be at least 8 km")
    if len(payload.get("commands", [])) != 15:
        raise ValueError("Scene 2 requires exactly 15 demonstration commands")
    kinds = {
        str(event["kind"])
        for event in payload.get("special_events", [])
    }
    required = {
        "slow_vehicle",
        "crossing_pedestrian",
        "bus_stop",
        "cyclist",
    }
    if not required.issubset(kinds):
        raise ValueError(
            "special events must include slow vehicle, pedestrian, "
            "bus stop, and cyclist"
        )
    traffic = payload["traffic"]
    if bool(traffic["hybrid_physics"]):
        raise ValueError(
            "hybrid physics is disabled to prevent visible NPC teleporting"
        )
    if bool(traffic["respawn_dormant_vehicles"]):
        raise ValueError(
            "runtime respawn is disabled to prevent visible NPC pop-in"
        )
    return payload


def setup_navigation_agents(carla_root: str | None) -> None:
    try:
        __import__("agents.navigation.behavior_agent")
        return
    except ModuleNotFoundError as error:
        if error.name and error.name != "agents":
            raise RuntimeError(
                "CARLA navigation agent dependency is missing: "
                + error.name
            ) from error
    configured = carla_root or os.environ.get("CARLA_ROOT")
    if configured:
        api_path = Path(configured) / "PythonAPI" / "carla"
        if api_path.exists() and str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))
    try:
        __import__("agents.navigation.behavior_agent")
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("agents"):
            raise RuntimeError(
                "CARLA navigation agent dependency is missing: "
                + error.name
            ) from error
        raise RuntimeError(
            "CARLA navigation agents are required for preview control; "
            "set CARLA_ROOT to the extracted CARLA package"
        ) from error


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass
class SafetyMonitor:
    world: Any
    ego: Any
    registry: ActorRegistry

    def __post_init__(self) -> None:
        self.collisions: list[dict[str, Any]] = []
        self.lane_invasions: list[dict[str, Any]] = []
        self.simulation_time_s = 0.0

    def start(self) -> None:
        import carla

        library = self.world.get_blueprint_library()
        collision = self.world.spawn_actor(
            library.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=self.ego,
        )
        lane = self.world.spawn_actor(
            library.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=self.ego,
        )
        collision.listen(self._on_collision)
        lane.listen(self._on_lane_invasion)
        self.registry.add(collision)
        self.registry.add(lane)

    def _on_collision(self, event: Any) -> None:
        other = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        impulse_vector = {
            "x": float(getattr(impulse, "x", 0.0)),
            "y": float(getattr(impulse, "y", 0.0)),
            "z": float(getattr(impulse, "z", 0.0)),
        }
        self.collisions.append(
            {
                "event_id": f"collision_{int(event.frame)}_{len(self.collisions)}",
                "frame": int(event.frame),
                "timestamp_s": self.simulation_time_s,
                "simulation_time_s": self.simulation_time_s,
                "other_actor_id": getattr(other, "id", None),
                "other_actor_type": getattr(
                    other,
                    "type_id",
                    "unknown",
                ),
                "normal_impulse_ns": impulse_vector,
                "impulse_magnitude_ns": math.sqrt(
                    sum(value * value for value in impulse_vector.values())
                ),
            }
        )

    def _on_lane_invasion(self, event: Any) -> None:
        self.lane_invasions.append(
            {
                "event_id": (
                    f"lane_invasion_{int(event.frame)}_"
                    f"{len(self.lane_invasions)}"
                ),
                "frame": int(event.frame),
                "timestamp_s": self.simulation_time_s,
                "simulation_time_s": self.simulation_time_s,
                "markings": [
                    str(marking.type)
                    for marking in event.crossed_lane_markings
                ],
                "crossed_lane_markings": [
                    str(marking.type)
                    for marking in event.crossed_lane_markings
                ],
            }
        )


def apply_weather(world: Any, config: Mapping[str, Any]) -> None:
    import carla

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=float(config["cloudiness"]),
            precipitation=float(config["precipitation"]),
            precipitation_deposits=float(
                config["precipitation_deposits"]
            ),
            wind_intensity=float(config["wind_intensity"]),
            sun_azimuth_angle=float(config["sun_azimuth_angle"]),
            sun_altitude_angle=float(config["sun_altitude_angle"]),
            fog_density=float(config["fog_density"]),
            fog_distance=float(config["fog_distance"]),
            fog_falloff=float(config["fog_falloff"]),
            wetness=float(config["wetness"]),
        )
    )


def spawn_ego(world: Any, transform: Any) -> Any:
    library = world.get_blueprint_library()
    for identifier in EGO_BLUEPRINTS:
        try:
            blueprint = library.find(identifier)
        except (IndexError, RuntimeError):
            continue
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")
        candidate = transform
        candidate.location.z += 0.35
        actor = world.try_spawn_actor(blueprint, candidate)
        if actor is not None:
            return actor
    raise RuntimeError("failed to spawn Town05 Scene 2 ego")


def overlay_payload(
    progress_m: float,
    route_length_m: float,
    ego: Any,
    current_command: Mapping[str, Any] | None,
    event_states: Mapping[str, str],
    nearby: Mapping[str, int],
) -> dict[str, Any]:
    command_text = (
        str(current_command["text"])
        if current_command is not None
        else "WAITING"
    )
    return {
        "status": "RUNNING",
        "route_progress_m": progress_m,
        "route_length_m": route_length_m,
        "speed_kmh": speed_kmh(ego),
        "target_speed_kmh": 45.0,
        "asr_text": command_text,
        "source_step_action": (
            "COMPOSITE_COMMAND"
            if current_command is not None
            else "WAITING"
        ),
        "active_step_id": (
            str(current_command["id"])
            if current_command is not None
            else "WAITING"
        ),
        "parse_status": "PRESET_TEXT",
        "risk_level": (
            "MEDIUM"
            if any(state == "ACTIVE" for state in event_states.values())
            else "LOW"
        ),
        "reason": (
            "Town05 packaged map / deterministic event actors"
        ),
        "traffic_count": int(nearby["vehicles"]),
        "pedestrian_count": int(nearby["walkers"]),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    print("Validated runtime config: {0}".format(config_path))
    print("Map: {0}".format(config["map"]))
    print("Commands: {0}".format(len(config["commands"])))
    print(
        "Traffic: {0} vehicles, {1} ambient walkers".format(
            config["traffic"]["vehicles"],
            config["traffic"]["ambient_walkers"],
        )
    )
    if args.validate_only:
        return 0

    carla_root = setup_carla_api()
    setup_navigation_agents(carla_root)
    import carla
    from agents.navigation.behavior_agent import BehaviorAgent

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = (
        args.video_output.resolve()
        if args.video_output is not None
        else output_dir / "scene2_town05_preview.mp4"
    )
    runtime_log = JsonlWriter(output_dir / "runtime.jsonl")
    event_log = JsonlWriter(output_dir / "events.jsonl")
    command_log = JsonlWriter(output_dir / "commands.jsonl")
    registry = ActorRegistry()
    client = None
    world = None
    original_settings = None
    traffic_manager = None
    camera = None
    safety = None
    summary: dict[str, Any] = {}

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.load_world(str(config["map"]))
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(
            args.fixed_delta_seconds
        )
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        apply_weather(world, config["weather"])

        traffic_manager = client.get_trafficmanager(
            args.traffic_manager_port
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(
            int(config["traffic"]["seed"])
        )
        traffic_manager.set_global_distance_to_leading_vehicle(
            float(
                config["traffic"][
                    "global_distance_to_leading_vehicle_m"
                ]
            )
        )
        traffic_manager.global_percentage_speed_difference(
            float(
                config["traffic"]["global_speed_difference_pct"]
            )
        )
        traffic_manager.set_hybrid_physics_mode(False)
        traffic_manager.set_respawn_dormant_vehicles(False)

        route, route_distances = build_repeated_route(
            world.get_map(),
            int(config["route"]["start_spawn_index"]),
            int(config["route"]["turnaround_spawn_index"]),
            float(config["route"]["target_length_m"]),
            float(config["route"]["route_sampling_m"]),
        )
        route_length_m = float(route_distances[-1])
        start_progress_m = max(0.0, float(args.start_progress_m))
        start_route_index = min(
            range(len(route_distances)),
            key=lambda index: abs(
                route_distances[index] - start_progress_m
            ),
        )
        ego = spawn_ego(
            world,
            route[start_route_index][0].transform,
        )
        registry.add(ego)

        events = DeterministicSceneEvents(
            world,
            traffic_manager,
            registry,
            route,
            route_distances,
            config["special_events"],
            int(config["traffic"]["seed"]),
        )
        events.spawn()
        traffic = TownTrafficFlow(
            client,
            world,
            traffic_manager,
            registry,
            route,
            config["traffic"],
        )
        traffic.spawn(events.reserved_locations, ego.get_location())
        safety = SafetyMonitor(world, ego, registry)
        safety.start()

        agent = None
        if not args.external_ego_control:
            agent = BehaviorAgent(ego, behavior="normal")
            agent.set_target_speed(
                float(config["route"]["target_speed_kmh"])
            )
            agent.set_global_plan(
                list(route[start_route_index:]),
                stop_waypoint_creation=True,
                clean_queue=True,
            )

        for _ in range(30):
            ego.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()

        camera = ExperimentCamera(
            world,
            ego,
            str(output_dir / "camera_frames"),
            every_n_frames=1,
            width=args.camera_width,
            height=args.camera_height,
            save_images=args.record_images,
            video_output=str(video_output),
            video_fps=args.video_fps,
            ffmpeg_path=args.ffmpeg,
            video_overlay=args.video_overlay,
            camera_attributes={
                "gamma": "2.2",
                "exposure_mode": "histogram",
                "exposure_compensation": "0.3",
                "exposure_speed_up": "3.0",
                "exposure_speed_down": "1.0",
            },
            camera_pose=(-9.0, 0.0, 3.2, -9.0, 0.0),
        )
        camera.start()

        tracker = RouteProgressTracker(
            route,
            route_distances,
            index=start_route_index,
        )
        announced: set[str] = set()
        current_command = None
        start_snapshot = world.get_snapshot()
        start_time = float(start_snapshot.timestamp.elapsed_seconds)
        frame_counter = 0
        progress_m = 0.0
        while True:
            snapshot = world.get_snapshot()
            simulation_time_s = (
                float(snapshot.timestamp.elapsed_seconds) - start_time
            )
            if safety is not None:
                safety.simulation_time_s = simulation_time_s

            progress_m = tracker.update(ego.get_location())
            for command in config["commands"]:
                command_id = str(command["id"])
                if (
                    command_id not in announced
                    and progress_m >= float(command["announce_at_m"])
                ):
                    announced.add(command_id)
                    current_command = command
                    command_log.write(
                        {
                            "frame": int(snapshot.frame),
                            "simulation_time_s": simulation_time_s,
                            "route_progress_m": progress_m,
                            **command,
                        }
                    )
                    print(
                        "COMMAND | {0} | {1:.1f} m | {2}".format(
                            command_id,
                            progress_m,
                            command["text"],
                        )
                    )

            for change in events.update(progress_m):
                event_log.write(
                    {
                        "frame": int(snapshot.frame),
                        "simulation_time_s": simulation_time_s,
                        **change,
                    }
                )
                print(
                    "EVENT | {0} | {1}".format(
                        change["event_id"],
                        change["state"],
                    )
                )

            if agent is not None:
                control = agent.run_step()
                control.manual_gear_shift = False
                ego.apply_control(control)

            frame = world.tick()
            nearby = traffic.nearby_counts(ego.get_location())
            camera.save_frame(
                frame,
                overlay=overlay_payload(
                    progress_m,
                    route_length_m,
                    ego,
                    current_command,
                    events.states,
                    nearby,
                ),
            )
            if frame_counter % 20 == 0:
                runtime_log.write(
                    {
                        "frame": int(frame),
                        "simulation_time_s": simulation_time_s,
                        "route_progress_m": progress_m,
                        "route_length_m": route_length_m,
                        "ego_speed_kmh": speed_kmh(ego),
                        "nearby_vehicles_85m": nearby["vehicles"],
                        "nearby_walkers_85m": nearby["walkers"],
                        "event_states": dict(events.states),
                        "collisions": len(safety.collisions),
                        "lane_invasions": len(
                            safety.lane_invasions
                        ),
                    }
                )
            frame_counter += 1

            if progress_m >= float(config["route"]["target_length_m"]):
                print("Town05 Scene 2 route completed")
                break
            if (
                args.duration > 0.0
                and simulation_time_s >= args.duration
            ):
                print("Town05 Scene 2 preview duration completed")
                break

        summary = {
            "schema_version": "scene_2_town05_summary/v1",
            "map": world.get_map().name,
            "route_length_m": route_length_m,
            "route_progress_m": progress_m,
            "commands_announced": len(announced),
            "traffic_vehicles_spawned": len(traffic.vehicles),
            "ambient_walkers_spawned": len(traffic.walkers),
            "event_states": dict(events.states),
            "event_summary": events.summary(),
            "collision_count": len(safety.collisions),
            "lane_invasion_count": len(safety.lane_invasions),
            "video_output": str(video_output),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        runtime_log.close()
        event_log.close()
        command_log.close()
        if camera is not None:
            try:
                camera.destroy()
            except RuntimeError:
                pass
        if client is not None:
            registry.destroy(client)
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
