"""Run Town05 Scene 2 through perception, VLA/FSM, and ego control."""

from __future__ import annotations

import argparse
import copy
import json
import queue
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carla_bootstrap import setup_carla_api
from control.pid_controller import EgoPIDController
from control.route_adapter import attach_route_target, route_target_location
from control.safety_supervisor import (
    apply_adaptive_cruise_guard,
    apply_kinematic_conflict_guard,
    preserve_safe_lateral_maneuver,
)
from evaluation.camera import ExperimentCamera
from run_complex_avoidance_town05 import (
    JsonlWriter,
    SafetyMonitor,
    apply_weather,
    load_config,
    spawn_ego,
)
from scenarios.complex.town05_scene2 import (
    ActorRegistry,
    DeterministicSceneEvents,
    RouteProgressTracker,
    TownTrafficFlow,
    build_repeated_route,
    route_index_at,
    speed_kmh,
)
from scene2_closed_loop import Scene2ClosedLoop
from scene_understanding.core.carla_bbox_projection import (
    project_world_state_objects,
)
from scene_understanding.core.carla_world_state import CarlaWorldStateCollector
from scene_understanding.src.high_level_driving_actions import map_step_action


DEFAULT_CONFIG = ROOT / "configs" / "scene_2_town05_runtime.json"
DEFAULT_SUITE = ROOT / "configs" / "scene_2_command_suite.json"
DEFAULT_INTENTS = ROOT / "configs" / "scene_2_expected_driving_intents.json"
DEFAULT_TOKENS = ROOT / "outputs" / "scene2_intent_tokens.pt"
DEFAULT_MODELS = Path("D:/CARLA/models")


def apply_traffic_light_guard(
    decision: Mapping[str, Any],
    ego: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the ego vehicle's traffic-signal state before actuation."""

    is_at_light = bool(ego.is_at_traffic_light())
    state = str(ego.get_traffic_light_state()).split(".")[-1].lower()
    audit = {
        "source": "ego_vehicle_state",
        "is_at_traffic_light": is_at_light,
        "traffic_light_state": state,
        "override_applied": False,
    }
    result = copy.deepcopy(dict(decision))
    if not is_at_light or state not in {"red", "yellow"}:
        return result, audit
    if result.get("action") == "emergency_brake":
        return result, audit

    result["action"] = "stop"
    result["target_speed_kmh"] = 0.0
    result["target_lane"] = None
    result["target_location"] = None
    result["emergency"] = False
    result["reason"] = f"traffic_light_{state}_guard"
    reasons = list(result.get("blocked_reason_codes", []))
    reason_code = f"ego_traffic_light_{state}"
    if reason_code not in reasons:
        reasons.append(reason_code)
    result["blocked_reason_codes"] = reasons
    audit["override_applied"] = True
    return result, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--command-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS)
    parser.add_argument("--intent-tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "scene2_closed_loop",
    )
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--perception-stride", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--video-overlay", action="store_true")
    parser.add_argument("--start-progress-m", type=float, default=150.0)
    parser.add_argument(
        "--metadata-semantic-proxy",
        action="store_true",
        help="Debug only: enrich perception objects with CARLA actor metadata.",
    )
    parser.add_argument(
        "--allow-stuck-recovery",
        action="store_true",
        help="Debug only: teleport the ego forward after 20 seconds without progress.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


class SynchronizedPerceptionCamera:
    def __init__(self, world: Any, ego: Any, width: int = 960, height: int = 540):
        self.world = world
        self.ego = ego
        self.width = int(width)
        self.height = int(height)
        self.sensor = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._pending: dict[int, Any] = {}

    def start(self) -> None:
        import carla

        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self.width))
        blueprint.set_attribute("image_size_y", str(self.height))
        blueprint.set_attribute("fov", "90")
        blueprint.set_attribute("sensor_tick", "0.05")
        if blueprint.has_attribute("gamma"):
            blueprint.set_attribute("gamma", "2.2")
        transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0),
        )
        self.sensor = self.world.spawn_actor(
            blueprint, transform, attach_to=self.ego
        )
        self.sensor.listen(self._queue.put)

    def image_for_frame(self, frame: int, timeout_s: float = 1.0) -> Image.Image:
        frame = int(frame)
        image = self._pending.pop(frame, None)
        deadline = time.time() + timeout_s
        while image is None and time.time() < deadline:
            try:
                candidate = self._queue.get(
                    timeout=max(0.01, deadline - time.time())
                )
            except queue.Empty:
                break
            candidate_frame = int(candidate.frame)
            if candidate_frame < frame:
                continue
            if candidate_frame > frame:
                self._pending[candidate_frame] = candidate
                break
            image = candidate
        if image is None:
            raise TimeoutError(f"front RGB frame {frame} was not received")
        return Image.frombuffer(
            "RGBA",
            (self.width, self.height),
            bytes(image.raw_data),
            "raw",
            "BGRA",
            0,
            1,
        ).convert("RGB")

    def destroy(self) -> None:
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None


def _load_commands(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    commands = payload.get("commands", [])
    if len(commands) < 8:
        raise ValueError("Scene 2 command suite must contain at least 8 commands")
    return commands


def _color_name(value: str) -> str | None:
    try:
        red, green, blue = [int(part) for part in value.split(",")[:3]]
    except (TypeError, ValueError):
        return None
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    if minimum >= 210:
        return "white"
    if red >= 150 and red >= green * 1.35 and red >= blue * 1.35:
        return "red"
    if blue >= 130 and blue >= red * 1.25:
        return "blue"
    if maximum - minimum <= 25:
        return "gray"
    return None


def enrich_actor_metadata_semantics(
    world_state: dict[str, Any],
    actors: Any,
) -> dict[str, Any]:
    """Expose simulator actor labels as an explicit integration proxy."""

    actor_index = {str(actor.id): actor for actor in actors}
    applied = 0
    for obj in world_state.get("objects", []):
        actor = actor_index.get(str(obj.get("source_object_id")))
        if actor is None:
            continue
        attributes = getattr(actor, "attributes", {})
        role_name = str(attributes.get("role_name", "")).strip()
        color = _color_name(str(attributes.get("color", "")))
        terms = [term for term in (color, role_name) if term]
        if not terms:
            continue
        obj["semantic_matches"] = [
            {
                "camera_name": "carla_metadata_proxy",
                "visual_object_id": f"metadata_{actor.id}",
                "bbox_2d": [0.0, 0.0, 1.0, 1.0],
                "description": " ".join(terms),
                "confidence": 1.0,
            }
        ]
        applied += 1
    world_state["provenance"]["semantic_source"] = "carla_actor_metadata_proxy"
    world_state["provenance"]["camera_names"] = ["carla_metadata_proxy"]
    world_state["environment"]["scene_summary"] = (
        f"CARLA metadata proxy applied to {applied} actors"
    )
    return world_state


def world_state_sensor_events(
    safety: SafetyMonitor,
    cursor: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Adapt and consume new monitor records for one WorldState frame."""

    cursor = cursor if cursor is not None else {}
    collision_start = int(cursor.get("collisions", 0))
    lane_start = int(cursor.get("lane_invasions", 0))
    collisions = []
    for index, event in enumerate(
        safety.collisions[collision_start:],
        start=collision_start,
    ):
        actor_id = event.get("other_actor_id")
        collisions.append(
            {
                "event_id": str(
                    event.get("event_id")
                    or f"collision_{event.get('frame')}_{index}"
                ),
                "frame": int(event["frame"]),
                "timestamp_s": float(
                    event.get(
                        "timestamp_s",
                        event.get("simulation_time_s", 0.0),
                    )
                ),
                "other_actor_id": (
                    str(actor_id) if actor_id is not None else "unknown"
                ),
                "normal_impulse_ns": dict(
                    event.get("normal_impulse_ns")
                    or {"x": 0.0, "y": 0.0, "z": 0.0}
                ),
                "impulse_magnitude_ns": float(
                    event.get("impulse_magnitude_ns", 0.0)
                ),
            }
        )
    lane_invasions = []
    for index, event in enumerate(
        safety.lane_invasions[lane_start:],
        start=lane_start,
    ):
        lane_invasions.append(
            {
                "event_id": str(
                    event.get("event_id")
                    or f"lane_invasion_{event.get('frame')}_{index}"
                ),
                "frame": int(event["frame"]),
                "timestamp_s": float(
                    event.get(
                        "timestamp_s",
                        event.get("simulation_time_s", 0.0),
                    )
                ),
                "crossed_lane_markings": list(
                    event.get("crossed_lane_markings")
                    or event.get("markings")
                    or []
                ),
            }
        )
    cursor["collisions"] = len(safety.collisions)
    cursor["lane_invasions"] = len(safety.lane_invasions)
    return {
        "collisions": collisions,
        "lane_invasions": lane_invasions,
    }


def _planner_target(
    runtime: Scene2ClosedLoop,
    route: list[tuple[Any, Any]],
    route_index: int,
    current_speed_kmh: float,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    step = runtime.active_step()
    if step is None:
        return None, {"status": "UNAVAILABLE", "reason": "no_active_step"}
    action, _, _, _ = map_step_action(step, current_speed_kmh)
    if action not in {"turn_left", "turn_right"}:
        return None, {"status": "NOT_REQUIRED", "action": action}
    return route_target_location(
        route,
        route_index,
        action=action,
        maneuver_search_m=250.0,
    )


def _overlay(
    result: Mapping[str, Any],
    command: Mapping[str, Any],
    ego: Any,
    control: Any,
    simulation_time_s: float,
    nearby: Mapping[str, int],
    route_progress_m: float,
    route_length_m: float,
    collisions: int,
    lane_events: int,
    command_phase: str,
    command_text: str,
) -> dict[str, Any]:
    decision = result["control_decision"]
    state = result["control_plan_state"]
    tracks = result["perception_frame"].get("tracks", [])
    latency = result["latency_ms"]
    return {
        "status": command_phase,
        "speed_kmh": speed_kmh(ego),
        "target_speed_kmh": decision["target_speed_kmh"],
        "asr_text": command_text,
        "action": decision["action"],
        "emergency": decision["emergency"],
        "risk_level": str(result["risk_assessment"]["risk_level"]).upper(),
        "policy_state": state["plan_status"],
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "steer": float(control.steer),
        "sim_time_s": simulation_time_s,
        "traffic_count": int(nearby["vehicles"]),
        "pedestrian_count": int(nearby["walkers"]),
        "command_phase": command_phase,
        "route_progress_m": route_progress_m,
        "route_length_m": route_length_m,
        "detected_vehicle_count": sum(
            str(track.get("category", "")).lower() == "vehicle"
            for track in tracks
        ),
        "detected_pedestrian_count": sum(
            str(track.get("category", "")).lower()
            in {"pedestrian", "walker"}
            for track in tracks
        ),
        "parse_latency_ms": latency.get("parser_precomputed"),
        "scene_decision_latency_ms": (
            float(latency.get("semantic_alignment_ms", 0.0))
            + float(latency.get("risk_ms", 0.0))
        ),
        "end_to_end_ms": latency.get("frame_pipeline_ms"),
        "collisions": collisions,
        "lane_events": lane_events,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    commands = _load_commands(args.command_suite.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config": str(args.config.resolve()),
                    "commands": len(commands),
                    "intents": str(args.intents.resolve()),
                    "tokens": str(args.intent_tokens.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    setup_carla_api()
    import carla

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = (
        args.video_output.resolve()
        if args.video_output is not None
        else output_dir / "scene2_closed_loop.mp4"
    )
    frame_log = JsonlWriter(output_dir / "pipeline.jsonl")
    command_log = JsonlWriter(output_dir / "commands.jsonl")
    event_log = JsonlWriter(output_dir / "events.jsonl")
    registry = ActorRegistry()
    client = None
    world = None
    original_settings = None
    traffic_manager = None
    video_camera = None
    perception_camera = None
    safety = None
    runtime = None
    summary: dict[str, Any] = {}
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.load_world(str(config["map"]))
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta_seconds)
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        apply_weather(world, config["weather"])

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(config["traffic"]["seed"]))
        traffic_manager.set_global_distance_to_leading_vehicle(
            float(config["traffic"]["global_distance_to_leading_vehicle_m"])
        )
        traffic_manager.global_percentage_speed_difference(
            float(config["traffic"]["global_speed_difference_pct"])
        )
        traffic_manager.set_hybrid_physics_mode(False)
        traffic_manager.set_respawn_dormant_vehicles(False)

        route, distances = build_repeated_route(
            world.get_map(),
            int(config["route"]["start_spawn_index"]),
            int(config["route"]["turnaround_spawn_index"]),
            float(config["route"]["target_length_m"]),
            float(config["route"]["route_sampling_m"]),
        )
        route_length_m = float(distances[-1])
        start_index = min(
            range(len(distances)),
            key=lambda index: abs(
                distances[index] - max(0.0, args.start_progress_m)
            ),
        )
        start_progress_m = float(distances[start_index])
        ego = spawn_ego(world, route[start_index][0].transform)
        registry.add(ego)
        events = DeterministicSceneEvents(
            world,
            traffic_manager,
            registry,
            route,
            distances,
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

        for _ in range(20):
            ego.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()
        initial_speed_mps = (
            float(config["route"].get("initial_speed_kmh", 0.0)) / 3.6
        )
        if initial_speed_mps > 0.0:
            forward = ego.get_transform().get_forward_vector()
            ego.set_target_velocity(
                carla.Vector3D(
                    x=float(forward.x) * initial_speed_mps,
                    y=float(forward.y) * initial_speed_mps,
                    z=0.0,
                )
            )

        video_camera = ExperimentCamera(
            world,
            ego,
            str(output_dir / "camera_frames"),
            every_n_frames=1,
            width=args.camera_width,
            height=args.camera_height,
            save_images=False,
            video_output=str(video_output),
            video_fps=args.video_fps,
            ffmpeg_path=args.ffmpeg,
            video_overlay=args.video_overlay,
            camera_attributes={
                "gamma": "2.2",
                "exposure_mode": "histogram",
                "exposure_compensation": "0.0",
                "exposure_speed_up": "3.0",
                "exposure_speed_down": "1.0",
            },
            camera_pose=(-9.0, 0.0, 3.2, -9.0, 0.0),
        )
        video_camera.start()
        perception_camera = SynchronizedPerceptionCamera(world, ego)
        perception_camera.start()

        runtime = Scene2ClosedLoop(
            intents_path=args.intents.resolve(),
            intent_token_cache=args.intent_tokens.resolve(),
            yolop_root=args.models.resolve() / "external" / "YOLOP",
            yolo11_weights=args.models.resolve()
            / "scene_understanding"
            / "yolo11s_specialized_carla_v1"
            / "weights"
            / "best.pt",
            vla_dir=args.models.resolve() / "lightweight_vla_adapter" / "v10",
            perception_stride=args.perception_stride,
        )
        controller = EgoPIDController(
            ego,
            world.get_map(),
            float(config["route"]["target_speed_kmh"]),
        )
        collector = CarlaWorldStateCollector(world, ego, max_distance_m=100.0)
        progress_tracker = RouteProgressTracker(
            route, distances, index=start_index
        )
        announced: set[str] = set()
        current_command: dict[str, Any] | None = None
        command_started_s = 0.0
        command_completed_s: float | None = None
        safety_event_cursor = {"collisions": 0, "lane_invasions": 0}
        camera_sync_skips = 0
        start_time = float(world.get_snapshot().timestamp.elapsed_seconds)
        frame = world.tick()
        progress_m = float(distances[start_index])
        last_advance_progress_m = progress_m
        last_advance_simulation_time_s = 0.0
        stuck_recovery_count = 0
        last_result = None
        while True:
            snapshot = world.get_snapshot()
            simulation_time_s = (
                float(snapshot.timestamp.elapsed_seconds) - start_time
            )
            safety.simulation_time_s = simulation_time_s
            progress_m = progress_tracker.update(ego.get_location())
            for command in commands:
                command_id = str(command["id"])
                if (
                    command_id not in announced
                    and progress_m >= float(command["announce_at_m"])
                ):
                    announced.add(command_id)
                    current_command = command
                    command_started_s = simulation_time_s
                    command_completed_s = None
                    runtime.activate(command_id)
                    command_log.write(
                        {
                            "frame": int(frame),
                            "simulation_time_s": simulation_time_s,
                            "route_progress_m": progress_m,
                            **command,
                        }
                    )

            for change in events.update(progress_m):
                event_log.write(
                    {
                        "frame": int(frame),
                        "simulation_time_s": simulation_time_s,
                        **change,
                    }
                )

            if current_command is None:
                ego.apply_control(carla.VehicleControl(brake=1.0))
                frame = world.tick()
                continue
            try:
                image = perception_camera.image_for_frame(frame)
            except TimeoutError as exc:
                camera_sync_skips += 1
                event_log.write(
                    {
                        "frame": int(frame),
                        "simulation_time_s": simulation_time_s,
                        "event_id": "perception_camera_sync_skip",
                        "state": "SKIPPED",
                        "reason": str(exc),
                    }
                )
                ego.apply_control(carla.VehicleControl(brake=1.0))
                frame = world.tick()
                continue
            world_state = collector.collect(
                sensor_events=world_state_sensor_events(
                    safety,
                    safety_event_cursor,
                )
            )
            if args.metadata_semantic_proxy:
                world_state = enrich_actor_metadata_semantics(
                    world_state,
                    world.get_actors(),
                )
            projection_record = project_world_state_objects(
                world_state,
                world.get_actors(),
                perception_camera.sensor,
                camera_name="front_rgb",
                image_width=perception_camera.width,
                image_height=perception_camera.height,
                fov_deg=90.0,
            )
            planner_target, planner_diagnostics = _planner_target(
                runtime,
                route,
                progress_tracker.index,
                speed_kmh(ego),
            )
            result = runtime.process(
                world_state=world_state,
                image=image,
                projection_record=projection_record,
                planner_target_location=planner_target,
                lateral_diagnostics=controller.lateral_diagnostics,
                route_progress_m=progress_m,
                route_length_m=route_length_m,
            )
            lateral_guarded, lateral_audit = preserve_safe_lateral_maneuver(
                result["control_decision"],
                runtime.active_step(),
                result["risk_assessment"],
                speed_setpoint_kmh=result["speed_setpoint"][
                    "target_speed_kmh"
                ],
            )
            cruise_guarded, cruise_audit = apply_adaptive_cruise_guard(
                lateral_guarded,
                world_state,
                result["risk_assessment"],
                speed_limit_kmh=result["speed_setpoint"][
                    "target_speed_kmh"
                ],
            )
            conflict_guarded, conflict_audit = apply_kinematic_conflict_guard(
                cruise_guarded,
                world_state,
                result["risk_assessment"],
            )
            guarded_decision, traffic_light_audit = apply_traffic_light_guard(
                conflict_guarded,
                ego,
            )
            command_decision, route_diagnostics = attach_route_target(
                guarded_decision,
                route,
                progress_tracker.index,
            )
            control, normalized = controller.run_step(
                command_decision,
                float(args.fixed_delta_seconds),
            )
            control.manual_gear_shift = False
            ego.apply_control(control)
            nearby = traffic.nearby_counts(ego.get_location())
            if simulation_time_s - command_started_s < 1.0:
                presentation_phase = "EXECUTING"
            elif result["control_plan_state"]["plan_status"] == "COMPLETED":
                if command_completed_s is None:
                    command_completed_s = simulation_time_s
                presentation_phase = (
                    "SUCCESS"
                    if simulation_time_s - command_completed_s < 2.0
                    else "WAITING"
                )
            else:
                presentation_phase = "EXECUTING"
            presentation_text = (
                str(current_command["text"])
                if presentation_phase != "WAITING"
                else "WAITING"
            )
            video_camera.save_frame(
                frame,
                overlay=_overlay(
                    result,
                    current_command,
                    ego,
                    control,
                    simulation_time_s,
                    nearby,
                    progress_m,
                    route_length_m,
                    len(safety.collisions),
                    len(safety.lane_invasions),
                    presentation_phase,
                    presentation_text,
                ),
            )
            frame_log.write(
                {
                    "frame": int(frame),
                    "simulation_time_s": simulation_time_s,
                    "route_progress_m": progress_m,
                    "speed_kmh": speed_kmh(ego),
                    "command_id": current_command["id"],
                    "active_step_id": result["control_plan_state"].get(
                        "active_step_id"
                    ),
                    "plan_status": result["control_plan_state"]["plan_status"],
                    "semantic_alignment": result["semantic_alignment"],
                    "risk_assessment": result["risk_assessment"],
                    "perception_frame": result["perception_frame"],
                    "visual_fusion_audit": result[
                        "visual_fusion_audit"
                    ],
                    "vla_proposal": result["vla_proposal"],
                    "control_decision": result["control_decision"],
                    "controller_command": normalized,
                    "traffic_light_guard": traffic_light_audit,
                    "kinematic_conflict_guard": conflict_audit,
                    "adaptive_cruise_guard": cruise_audit,
                    "lateral_progress_gate": lateral_audit,
                    "vehicle_control": {
                        "throttle": float(control.throttle),
                        "brake": float(control.brake),
                        "steer": float(control.steer),
                    },
                    "lateral_diagnostics": controller.lateral_diagnostics,
                    "planner_target": planner_diagnostics,
                    "route_target": route_diagnostics,
                    "step_feedback": result["step_feedback"],
                    "speed_setpoint": result["speed_setpoint"],
                    "latency_ms": result["latency_ms"],
                    "nearby": nearby,
                    "collisions": len(safety.collisions),
                    "lane_invasions": len(safety.lane_invasions),
                    "sensor_events": world_state.get(
                        "sensor_events",
                        {},
                    ),
                }
            )
            if progress_m >= last_advance_progress_m + 5.0:
                last_advance_progress_m = progress_m
                last_advance_simulation_time_s = simulation_time_s
            elif (
                args.allow_stuck_recovery
                and
                simulation_time_s - last_advance_simulation_time_s >= 20.0
                and speed_kmh(ego) < 2.0
            ):
                recovery_index = route_index_at(
                    distances,
                    min(progress_m + 30.0, route_length_m - 2.0),
                )
                recovery_transform = route[recovery_index][0].transform
                recovery_transform.location.z += 0.5
                ego.set_transform(recovery_transform)
                forward = recovery_transform.get_forward_vector()
                ego.set_target_velocity(
                    carla.Vector3D(
                        x=float(forward.x) * 8.0,
                        y=float(forward.y) * 8.0,
                        z=0.0,
                    )
                )
                progress_tracker.index = recovery_index
                last_advance_progress_m = float(
                    distances[recovery_index]
                )
                last_advance_simulation_time_s = simulation_time_s
                stuck_recovery_count += 1
                event_log.write(
                    {
                        "event_id": (
                            f"stuck_recovery_{stuck_recovery_count}"
                        ),
                        "state": "APPLIED",
                        "simulation_time_s": simulation_time_s,
                        "from_progress_m": progress_m,
                        "to_progress_m": last_advance_progress_m,
                        "reason": "no_progress_for_20s",
                    }
                )
            last_result = result
            if progress_m >= float(config["route"]["target_length_m"]):
                break
            if args.duration > 0 and simulation_time_s >= args.duration:
                break
            frame = world.tick()

        summary = {
            "schema_version": "scene2_closed_loop_summary/v1",
            "map": world.get_map().name,
            "route_progress_m": progress_m,
            "route_length_m": route_length_m,
            "start_progress_m": start_progress_m,
            "distance_travelled_m": max(
                0.0,
                progress_m - start_progress_m,
            ),
            "commands_announced": len(announced),
            "last_plan_status": last_result["control_plan_state"]["plan_status"]
            if last_result
            else None,
            "collision_count": len(safety.collisions),
            "lane_invasion_count": len(safety.lane_invasions),
            "traffic_vehicles_spawned": len(traffic.vehicles),
            "ambient_walkers_spawned": len(traffic.walkers),
            "camera_sync_skips": camera_sync_skips,
            "stuck_recovery_count": stuck_recovery_count,
            "metadata_semantic_proxy": bool(
                args.metadata_semantic_proxy
            ),
            "allow_stuck_recovery": bool(args.allow_stuck_recovery),
            "event_states": dict(events.states),
            "video_output": str(video_output),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        frame_log.close()
        command_log.close()
        event_log.close()
        if perception_camera is not None:
            try:
                perception_camera.destroy()
            except RuntimeError:
                pass
        if video_camera is not None:
            try:
                video_camera.destroy()
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
