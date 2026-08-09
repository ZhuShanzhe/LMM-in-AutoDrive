"""Run one CARLA scenario with control, logging, sensors, and metric export.

Run from this directory after CARLA is started:
    python run_control_experiment.py emergency_brake --duration-s 25
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from carla_bootstrap import setup_carla_api

setup_carla_api()

import carla

from control.agents import CarlaAgentController
from control.async_qwen_bridge import AsyncQwenBridge
from control.decision_provider import JsonFileDecisionPolicy
from control.live_perception_bridge import LivePerceptionBridge
from control.scene_bridge_policy import SceneBridgeDecisionPolicy
from control.scene_understanding_json_policy import SceneUnderstandingJsonPolicy
from control.scheduled_scene_bridge_policy import ScheduledSceneBridgePolicy
from control.structured_vla_scene_bridge_policy import StructuredVlaSceneBridgePolicy
from control.pid_controller import EgoPIDController
from evaluation.events import EventMonitor
from evaluation.camera import ExperimentCamera
from evaluation.logger import ExperimentLogger
from evaluation.metrics import summarize
from map_utils import resolve_carla_map_name
from perception.world_state import WorldState
from scene_event_adapter import scene_sensor_events
from scene_understanding_capture import SceneUnderstandingCapture
from scene_understanding.core.carla_world_state import CarlaWorldStateCollector
from scenarios.basic.straight_driving import StraightDrivingScenario
from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario
from scenarios.basic.urban_voice_5km import UrbanVoice5KmScenario
from scenarios.continuous.basic_track_5km import BasicTrack5KmScenario
from scenarios.complex.urban_complex_8km import UrbanComplex8KmScenario
from scenarios.emergency.emergency_brake import EmergencyBrakeScenario
from scenarios.pedestrian.pedestrian_crossing import PedestrianCrossingScenario
from scenarios.validation.braking_with_traffic import BrakingWithTrafficValidationScenario
from scenarios.validation.lane_change_with_traffic import LaneChangeWithTrafficValidationScenario


SCENARIOS = {
    "straight_driving": StraightDrivingScenario,
    "basic_voice_control_5km": BasicVoiceControl5KmScenario,
    "basic_voice_urban_5km": UrbanVoice5KmScenario,
    "basic_track_5km": BasicTrack5KmScenario,
    "complex_avoidance_8km": UrbanComplex8KmScenario,
    "emergency_brake": EmergencyBrakeScenario,
    "braking_with_traffic_validation": BrakingWithTrafficValidationScenario,
    "lane_change_with_traffic_validation": LaneChangeWithTrafficValidationScenario,
    "pedestrian_crossing": PedestrianCrossingScenario,
}


class RuleDecisionPolicy:
    """Temporary rule policy used to validate the control/evaluation pipeline.

    Replace ``decide`` with the LLM/decision-module call during integration.
    """
    def __init__(self, scenario_name, cruise_speed_kmh):
        self.scenario_name = scenario_name
        self.cruise_speed_kmh = cruise_speed_kmh
        self._emergency_latched = False
        self._emergency_reason = None

    def decide(self, world_state):
        nearby_vehicles = world_state.get("vehicles", [])
        nearby_pedestrians = world_state.get("pedestrians", [])
        ego_speed_kmh = float(world_state.get("ego", {}).get("speed(km/h)", 0.0))
        front_vehicles = [
            item for item in nearby_vehicles
            if item["relative_position"]["x"] > 0.0
            and abs(item["relative_position"]["y"]) < 2.0
        ]
        front_vehicle = min(front_vehicles, key=lambda item: item["distance"], default=None)
        front_distance = front_vehicle["distance"] if front_vehicle is not None else float("inf")
        pedestrian_distance = min([item["distance"] for item in nearby_pedestrians] or [float("inf")])
        front_speed_kmh = (
            float(front_vehicle.get("speed_kmh", ego_speed_kmh))
            if front_vehicle is not None
            else ego_speed_kmh
        )
        closing_speed_kmh = max(0.0, ego_speed_kmh - front_speed_kmh)
        closing_speed_mps = closing_speed_kmh / 3.6
        front_ttc_s = (
            front_distance / closing_speed_mps
            if closing_speed_mps > 0.1
            else float("inf")
        )
        imminent_front_risk = (
            front_vehicle is not None
            and (
                front_distance < 10.0
                or (front_distance < 25.0 and front_ttc_s < 3.0)
            )
        )
        if pedestrian_distance < 10.0 or imminent_front_risk:
            self._emergency_latched = True
            self._emergency_reason = (
                "front_vehicle_braking"
                if imminent_front_risk
                else "rule_safety_distance"
            )
        if self._emergency_latched:
            return {
                "action": "emergency_brake",
                "target_speed_kmh": 0.0,
                "emergency": True,
                "reason": self._emergency_reason,
            }
        if (
            front_vehicle is not None
            and front_distance < 35.0
            and closing_speed_kmh > 3.0
        ):
            return {
                "action": "decelerate",
                "target_speed_kmh": max(0.0, front_speed_kmh),
                "emergency": False,
                "reason": "closing_front_vehicle",
            }
        return {
            "action": "keep_lane",
            "target_speed_kmh": self.cruise_speed_kmh,
            "reason": "rule_cruise",
        }


def build_controller(name, vehicle, world_map, target_speed_kmh, scenario=None, force_low_level=False):
    custom_factory = getattr(scenario, "create_controller", None)
    if custom_factory is not None and not force_low_level:
        return custom_factory()
    if name == "pid":
        return EgoPIDController(vehicle, world_map, target_speed_kmh)
    return CarlaAgentController(vehicle, mode=name)


def get_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def visible_traffic_probe(world, ego_vehicle, max_distance_m=120.0, half_fov_deg=55.0):
    """Approximate forward-camera traffic visibility from actor geometry."""
    transform = ego_vehicle.get_transform()
    origin = transform.location
    forward = transform.get_forward_vector()
    minimum_cosine = math.cos(math.radians(float(half_fov_deg)))
    visible_ids = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_vehicle.id or not actor.is_alive:
            continue
        location = actor.get_location()
        dx = location.x - origin.x
        dy = location.y - origin.y
        distance = math.hypot(dx, dy)
        if distance <= 0.1 or distance > float(max_distance_m):
            continue
        forward_cosine = (dx * forward.x + dy * forward.y) / distance
        if forward_cosine >= minimum_cosine:
            visible_ids.append(int(actor.id))
    return {
        "front_cone_vehicle_count": len(visible_ids),
        "vehicle_ids": visible_ids,
        "max_distance_m": float(max_distance_m),
        "horizontal_fov_deg": float(half_fov_deg) * 2.0,
    }


def cleanup_project_vehicles(world):
    """Remove only stale actors previously created by this experiment runner."""
    role_prefixes = ("background_", "route_traffic")
    role_names = {"ego", "npc"}
    stale = []
    for actor in world.get_actors().filter("vehicle.*"):
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name in role_names or role_name.startswith(role_prefixes):
            stale.append(actor)
    for actor in stale:
        if actor.is_alive:
            actor.destroy()
    if stale:
        world.tick()
    return len(stale)


def draw_main_road_visual_markings(world, road_length_m=5000.0):
    """Draw persistent camera-visible markings for the generated six-lane road."""
    # Debug lines bypass the road material and otherwise clip to white in the
    # camera post-process pass. Keep them deliberately subdued.
    white = carla.Color(128, 128, 120)
    yellow = carla.Color(115, 82, 10)

    def line(start_x, end_x, y, color, width):
        world.debug.draw_line(
            carla.Location(x=start_x, y=y, z=0.08),
            carla.Location(x=end_x, y=y, z=0.08),
            thickness=width,
            color=color,
            life_time=3600.0,
        )

    line(0.0, road_length_m, -0.16, yellow, 0.06)
    line(0.0, road_length_m, 0.16, yellow, 0.06)
    for y in (-10.5, 10.5):
        line(0.0, road_length_m, y, white, 0.06)
    for y in (-3.5, -7.0, 3.5, 7.0):
        for start_x in range(0, int(road_length_m), 14):
            line(float(start_x), float(min(start_x + 6, road_length_m)), y, white, 0.05)


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        return {
            "x": round(float(value.x), 4),
            "y": round(float(value.y), 4),
            "z": round(float(value.z), 4),
        }
    if all(hasattr(value, attr) for attr in ("pitch", "yaw", "roll")):
        return {
            "pitch": round(float(value.pitch), 4),
            "yaw": round(float(value.yaw), 4),
            "roll": round(float(value.roll), 4),
        }
    return str(value)


def call_scenario_method(scenario, method_name, default=None, *args):
    method = getattr(scenario, method_name, None)
    if method is None:
        return default
    try:
        return json_safe(method(*args))
    except Exception as exc:
        return {"error": "{0}: {1}".format(type(exc).__name__, exc)}


def write_json_atomically(path, document):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".world-state-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False)
        for attempt in range(20):
            try:
                os.replace(temporary_path, path)
                temporary_path = None
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.005)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def decide_with_optional_wait(policy, state, wait_ms):
    """Wait for an exact-frame external decision without weakening fail-safe behavior."""

    intent = policy.decide(state)
    if wait_ms <= 0:
        return intent
    deadline = time.perf_counter() + wait_ms / 1000.0
    while policy.telemetry().get("status") != "accepted":
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.001)
        intent = policy.decide(state)
    return intent


def make_video_overlay(record):
    status = record.get("scenario_status", {})
    events = record.get("events", {})
    intent = record.get("intent", {})
    control = record.get("control", {})
    ego = record.get("ego", {})
    policy_telemetry = record.get("policy", {})
    presentation = policy_telemetry.get("command_presentation") or {}
    scene_trace = record.get("scene_decision", {})
    latest_perception = record.get("scene_understanding_latest") or {}
    plan_state = scene_trace.get("plan_state") or {}
    control_decision = scene_trace.get("control_decision") or {}
    risk_assessment = scene_trace.get("risk_assessment") or {}
    action = intent.get("action", "")
    decision_action = control_decision.get("action") or action
    emergency = bool(intent.get("emergency", False)) or decision_action == "emergency_brake"
    command_phase = str(
        intent.get("command_phase") or presentation.get("phase") or "WAITING"
    ).upper()
    if command_phase == "WAITING":
        asr_text = "WAITING"
    elif intent.get("voice_text"):
        asr_text = intent["voice_text"]
    else:
        asr_text = presentation.get("voice_text") or "VOICE COMMAND UNAVAILABLE"
    risk_level = str(risk_assessment.get("risk_level") or "unknown").upper()
    if command_phase == "WAITING":
        # The persisted scene plan intentionally refers to the command that
        # just completed. The current route-following intent is authoritative
        # for the waiting card and its held cruise speed.
        decision_action = action
        control_decision = {}
    policy_state = str(
        control_decision.get("decision_status")
        or plan_state.get("plan_status")
        or policy_telemetry.get("status")
        or "RUNNING"
    ).upper()
    if status.get("status") == "SUCCESS":
        policy_state = "COMPLETED"
    elif status.get("status") == "FAILURE":
        policy_state = "FAILED"
    async_qwen = record.get("async_qwen") or {}
    qwen_result = async_qwen.get("result") or {}
    qwen_submission = async_qwen.get("submission") or {}
    qwen_worker = async_qwen.get("worker") or {}
    return {
        "frame": record.get("frame", 0),
        "sim_time_s": record.get("sim_time_s", 0.0),
        "asr_text": asr_text,
        "command_phase": command_phase,
        "command_id": intent.get("command_id") or presentation.get("command_id"),
        "audio_file": intent.get("audio_file") or presentation.get("audio_file"),
        "parse_latency_ms": (policy_telemetry.get("parser") or {}).get("latency_ms"),
        "action": decision_action,
        "active_step_id": plan_state.get("active_step_id"),
        "reason": control_decision.get("reason") or intent.get("reason") or status.get("reason", ""),
        "target_speed_kmh": control_decision.get(
            "target_speed_kmh", intent.get("target_speed_kmh", 0.0)
        ),
        "emergency": emergency,
        "risk_level": risk_level,
        "policy_state": policy_state,
        "parse_status": control_decision.get("parse_status") or intent.get("parse_status"),
        "parse_confidence": control_decision.get("parse_confidence", intent.get("parse_confidence")),
        "perception_latency_ms": (
            record.get("latency_ms", {}).get("perception")
            or latest_perception.get("latency_ms")
        ),
        "perception_age_s": latest_perception.get("age_s"),
        "perception_status": latest_perception.get("status"),
        "detected_vehicle_count": latest_perception.get("vehicle_count", 0),
        "detected_pedestrian_count": latest_perception.get("pedestrian_count", 0),
        "scene_decision_latency_ms": record.get("latency_ms", {}).get("scene_decision"),
        "end_to_end_ms": record.get("latency_ms", {}).get("end_to_end", 0.0),
        "source_step_action": (
            control_decision.get("source_step_action")
            or intent.get("source_step_action")
            or decision_action
        ),
        "speed_kmh": ego.get("speed_kmh", 0.0),
        "throttle": control.get("throttle", 0.0),
        "brake": control.get("brake", 0.0),
        "steer": control.get("steer", 0.0),
        "status": status.get("status", "RUNNING"),
        "collisions": events.get("collision_count", 0),
        "lane_events": events.get("lane_invasion_count", 0),
        "route_progress_m": status.get("route_progress_m"),
        "route_length_m": status.get("route_length_m"),
        "active_events": status.get("scenario_events", {}).get("active", []),
        "qwen_status": qwen_result.get("status") or qwen_submission.get("status"),
        "qwen_latency_s": qwen_result.get("service_elapsed_seconds"),
        "qwen_worker": qwen_worker,
    }


def prepare_output_directory(output_dir):
    """Create a runner output directory before any controller opens logs."""

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolve_scenario_config(config_path, output_dir, resume_progress_m):
    if resume_progress_m is None:
        return config_path
    if config_path is None:
        raise ValueError("--resume-route-progress-m requires --scenario-config")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("route", {})["resume_progress_m"] = float(resume_progress_m)
    os.makedirs(output_dir, exist_ok=True)
    resolved_path = os.path.join(output_dir, "scenario_config.resolved.json")
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return resolved_path


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA control and evaluation runner")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--files-base-folder",
        default=None,
        help="Optional local directory for CARLA map and navigation cache files",
    )
    parser.add_argument("--map", default=None, help="Optional CARLA map name to load before setup")
    parser.add_argument(
        "--opendrive-map",
        default=None,
        help="Optional OpenDRIVE .xodr map to generate before scenario setup",
    )
    parser.add_argument(
        "--opendrive-visual-markings",
        action="store_true",
        help="Draw persistent six-lane road markings after OpenDRIVE generation",
    )
    parser.add_argument("--scenario-config", default=None, help="Optional JSON configuration for a configurable scenario")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--fixed-delta-s", type=float, default=0.05)
    parser.add_argument("--target-speed-kmh", type=float, default=25.0)
    parser.add_argument("--controller", choices=["pid", "basic", "behavior"], default="pid")
    parser.add_argument(
        "--decision-source",
        choices=["rule", "json_file", "scene_bridge", "voice_schedule", "voice_scene_bridge", "vla_scene_bridge"],
        default="rule",
        help="Use built-in rules or a per-tick external decision JSON file",
    )
    parser.add_argument(
        "--decision-json",
        default=None,
        help="DrivingIntent or ControlDecision JSON path for --decision-source json_file",
    )
    parser.add_argument(
        "--command-parser-model",
        default=None,
        help="ModernBERT model directory for text-to-DrivingIntent voice schedule parsing",
    )
    parser.add_argument("--command-parser-device", default="cuda")
    parser.add_argument(
        "--vla-checkpoint",
        default=None,
        help="Scene-2 structured-BEV VLA checkpoint for --decision-source vla_scene_bridge",
    )
    parser.add_argument(
        "--vla-config",
        default=None,
        help="VLA adapter JSON config for --decision-source vla_scene_bridge",
    )
    parser.add_argument("--vla-device", default="cuda")
    parser.add_argument(
        "--vla-precision", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument(
        "--driving-intent-json",
        default=None,
        help="DrivingIntent JSON path for --decision-source scene_bridge",
    )
    parser.add_argument(
        "--bridge-output-dir",
        default=None,
        help="Directory for per-tick scene-bridge artifacts",
    )
    parser.add_argument(
        "--decision-max-age-frames",
        type=int,
        default=None,
        help="Optional maximum accepted age for a JSON ControlDecision; stale decisions stop safely",
    )
    parser.add_argument(
        "--decision-wait-ms",
        type=float,
        default=0.0,
        help="Optional wait for an exact-frame external JSON decision before safe fallback",
    )
    parser.add_argument(
        "--world-state-output",
        default=None,
        help="Optional per-tick world-state JSON path for an external decision process",
    )
    parser.add_argument(
        "--scene-world-state-output",
        default=None,
        help="Optional per-tick schema-valid WorldState JSON path for the decision bridge",
    )
    parser.add_argument("--goal-distance-m", type=float, default=None)
    parser.add_argument("--stop-when-goal-reached", action="store_true")
    parser.add_argument(
        "--stop-at-route-progress-m",
        type=float,
        default=None,
        help="Finish a checkpoint segment after reaching this scenario route progress",
    )
    parser.add_argument(
        "--resume-route-progress-m",
        type=float,
        default=None,
        help="Start a configurable route scenario at this saved route progress",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--record-every-n", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-view", choices=["hood", "chase"], default="hood")
    parser.add_argument("--video-output", default=None, help="Optional direct H.264 output path")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--ffmpeg", default=None, help="Path to ffmpeg for --video-output")
    parser.add_argument(
        "--video-profile",
        choices=["realtime", "quality"],
        default="quality",
        help="H.264 profile; quality uses slower, higher-fidelity encoding",
    )
    parser.add_argument("--video-overlay", action="store_true", help="Overlay per-frame run telemetry on direct video")
    parser.add_argument("--terminal-hold-s", type=float, default=2.0, help="Seconds to hold SUCCESS/FAILURE video frame")
    parser.add_argument(
        "--scene-capture",
        action="store_true",
        help="Write frame-aligned scene-understanding capture bundles",
    )
    parser.add_argument(
        "--scene-capture-every-n",
        type=int,
        default=10,
        help="Capture one scene-understanding bundle every N simulation frames",
    )
    parser.add_argument("--scene-camera-width", type=int, default=800)
    parser.add_argument("--scene-camera-height", type=int, default=600)
    parser.add_argument("--scene-camera-timeout-s", type=float, default=1.0)
    parser.add_argument("--live-perception", action="store_true")
    parser.add_argument("--perception-yolop-root", default=None)
    parser.add_argument("--perception-yolo11-weights", default=None)
    parser.add_argument("--perception-device", default="cuda")
    parser.add_argument("--perception-image-size", type=int, choices=(320, 640), default=640)
    parser.add_argument("--perception-object-image-size", type=int, choices=(320, 640, 768, 960))
    parser.add_argument("--perception-score-threshold", type=float, default=0.10)
    parser.add_argument("--perception-min-iou", type=float, default=0.05)
    parser.add_argument("--async-qwen", action="store_true")
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument(
        "--qwen-prompt",
        default=os.path.join(PROJECT_ROOT, "scene_understanding", "prompts", "scene_understanding.txt"),
    )
    parser.add_argument("--qwen-max-age-s", type=float, default=15.0)
    parser.add_argument(
        "--qwen-finalize-timeout-s", type=float, default=0.0,
        help="Optional post-run wait to persist one asynchronous Qwen audit; never blocks control ticks",
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=768)
    parser.add_argument("--qwen-min-visual-tokens", type=int, default=256)
    parser.add_argument("--qwen-max-visual-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.decision_source == "json_file" and not args.decision_json:
        parser.error("--decision-json is required when --decision-source json_file")
    if args.decision_source == "scene_bridge" and not args.driving_intent_json:
        parser.error("--driving-intent-json is required when --decision-source scene_bridge")
    if args.decision_max_age_frames is not None and args.decision_max_age_frames < 0:
        parser.error("--decision-max-age-frames must be non-negative")
    if args.decision_wait_ms < 0:
        parser.error("--decision-wait-ms must be non-negative")
    if args.qwen_finalize_timeout_s < 0:
        parser.error("--qwen-finalize-timeout-s must be non-negative")
    if args.decision_wait_ms > 0:
        if args.decision_source != "json_file":
            parser.error("--decision-wait-ms requires --decision-source json_file")
        if args.decision_max_age_frames != 0:
            parser.error(
                "--decision-wait-ms requires --decision-max-age-frames 0 "
                "to enforce current-frame decisions"
            )
    if args.resume_route_progress_m is not None and args.scenario_config is None:
        parser.error("--resume-route-progress-m requires --scenario-config")
    if args.resume_route_progress_m is not None and args.resume_route_progress_m < 0.0:
        parser.error("--resume-route-progress-m must be non-negative")
    if args.live_perception and (
        not args.perception_yolop_root or not args.perception_yolo11_weights
    ):
        parser.error(
            "--live-perception requires --perception-yolop-root and "
            "--perception-yolo11-weights"
        )
    if args.async_qwen and not args.qwen_model_path:
        parser.error("--async-qwen requires --qwen-model-path")
    if args.decision_source == "vla_scene_bridge":
        if not args.command_parser_model:
            parser.error("--decision-source vla_scene_bridge requires --command-parser-model")
        if not args.vla_checkpoint or not args.vla_config:
            parser.error(
                "--decision-source vla_scene_bridge requires --vla-checkpoint and --vla-config"
            )
    return args


def main():
    args = parse_args()
    output_dir = prepare_output_directory(
        args.output_dir
        or os.path.join(
            "outputs",
            "runs",
            "{0}_{1}".format(
                args.scenario, time.strftime("%Y%m%d_%H%M%S")
            ),
        )
    )
    scenario_config_path = resolve_scenario_config(
        args.scenario_config,
        output_dir,
        args.resume_route_progress_m,
    )
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    if args.files_base_folder:
        cache_dir = os.path.abspath(args.files_base_folder)
        os.makedirs(cache_dir, exist_ok=True)
        client.set_files_base_folder(cache_dir)
    world = client.get_world()
    if args.opendrive_map:
        with open(args.opendrive_map, "r", encoding="utf-8") as handle:
            # Standalone OpenDRIVE defaults to collision walls at every road
            # boundary.  They are counterproductive on a continuous arterial
            # with gentle curves: a lane-centre spawn can touch a generated
            # wall and report ``static.unknown``.  The mesh remains visible
            # and receives a small tolerance width for stable vehicle motion.
            generation = carla.OpendriveGenerationParameters(
                vertex_distance=2.0,
                max_road_length=50.0,
                wall_height=0.0,
                additional_width=0.6,
                smooth_junctions=True,
                enable_mesh_visibility=True,
            )
            world = client.generate_opendrive_world(handle.read(), generation)
        if args.opendrive_visual_markings:
            draw_main_road_visual_markings(world)
    scenario_class = SCENARIOS[args.scenario]
    config_map = None
    if scenario_config_path is not None:
        with open(scenario_config_path, "r", encoding="utf-8") as handle:
            config_map = json.load(handle).get("map")
    target_map = args.map or config_map or getattr(scenario_class, "default_map", None)
    if target_map:
        target_map = resolve_carla_map_name(target_map, client.get_available_maps())
    if not args.opendrive_map and target_map and not world.get_map().name.endswith(target_map):
        world = client.load_world(target_map)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.fixed_delta_s
    if settings.substepping:
        max_substeps = max(1, int(settings.max_substeps))
        required_substep_delta = float(args.fixed_delta_s) / max_substeps
        settings.max_substep_delta_time = max(
            float(settings.max_substep_delta_time), required_substep_delta
        )
    world.apply_settings(settings)
    stale_actor_count = cleanup_project_vehicles(world)

    scenario_kwargs = {"external_control": True}
    if scenario_config_path is not None:
        scenario_kwargs["config_path"] = scenario_config_path
    scenario = scenario_class(world, **scenario_kwargs)
    scenario.client = client
    scenario.fixed_delta_s = args.fixed_delta_s
    monitor = None
    camera = None
    scene_capture = None
    live_perception = None
    last_live_perception = None
    async_qwen = None
    scene_world_state_collector = None
    logger = None
    policy = None
    unified_vla = None
    unified_route_pid = None
    records = []
    try:
        scenario.setup()
        world.tick()
        ego = scenario.get_ego_vehicle()
        if (
            args.scene_world_state_output
            or args.live_perception
            or args.async_qwen
            or args.decision_source in {"scene_bridge", "voice_scene_bridge", "vla_scene_bridge"}
        ):
            scene_world_state_collector = CarlaWorldStateCollector(world, ego)
        monitor = EventMonitor(world, ego)
        monitor.start()
        if args.record_images or args.video_output:
            camera_pose = (
                (1.5, 0.0, 2.4, 0.0, 0.0)
                if args.camera_view == "hood"
                else (-8.0, 0.0, 3.0, -10.0, 0.0)
            )
            camera = ExperimentCamera(
                world=world,
                ego_vehicle=ego,
                output_dir=os.path.join(output_dir, "camera_frames"),
                every_n_frames=args.record_every_n,
                width=args.camera_width,
                height=args.camera_height,
                save_images=args.record_images,
                video_output=args.video_output,
                video_fps=args.video_fps,
                ffmpeg_path=args.ffmpeg,
                video_overlay=args.video_overlay,
                camera_pose=camera_pose,
            )
            camera.start()
        if args.scene_capture or args.live_perception or args.async_qwen:
            scene_capture = SceneUnderstandingCapture(
                world,
                ego,
                output_dir=os.path.join(output_dir, "scene_understanding"),
                every_n_frames=args.scene_capture_every_n,
                image_width=args.scene_camera_width,
                image_height=args.scene_camera_height,
                camera_timeout_s=args.scene_camera_timeout_s,
            )
            scene_capture.setup()
        if args.live_perception:
            live_perception = LivePerceptionBridge(
                os.path.join(output_dir, "scene_understanding"),
                yolop_root=args.perception_yolop_root,
                yolo11_weights=args.perception_yolo11_weights,
                device=args.perception_device,
                image_size=args.perception_image_size,
                object_image_size=args.perception_object_image_size,
                score_threshold=args.perception_score_threshold,
                frame_rate=max(1, round(1.0 / (args.fixed_delta_s * args.scene_capture_every_n))),
                min_iou=args.perception_min_iou,
            )
        if args.async_qwen:
            async_qwen = AsyncQwenBridge(
                os.path.join(output_dir, "scene_understanding"),
                model_path=args.qwen_model_path,
                prompt_path=args.qwen_prompt,
                max_age_s=args.qwen_max_age_s,
                max_new_tokens=args.qwen_max_new_tokens,
                min_visual_tokens=args.qwen_min_visual_tokens,
                max_visual_tokens=args.qwen_max_visual_tokens,
            )
        controller = build_controller(
            args.controller,
            ego,
            world.get_map(),
            args.target_speed_kmh,
            scenario,
            force_low_level=args.decision_source in {"scene_bridge", "voice_scene_bridge", "vla_scene_bridge"},
        )
        controller_name = type(controller).__name__
        effective_decision_source = args.decision_source
        if args.decision_source == "json_file":
            policy = JsonFileDecisionPolicy(
                args.decision_json,
                args.target_speed_kmh,
                args.decision_max_age_frames,
            )
        elif args.decision_source == "scene_bridge":
            policy = SceneUnderstandingJsonPolicy(
                driving_intent_path=args.driving_intent_json,
                output_dir=args.bridge_output_dir or os.path.join(output_dir, "scene_bridge"),
                default_speed_kmh=args.target_speed_kmh,
                max_age_frames=0,
            )
        elif args.decision_source == "vla_scene_bridge":
            from control.generic_route_pid import GenericRoutePID
            from universal_vla_controller import UniversalVLAController

            commands = list(
                getattr(scenario, "config", {}).get("commands", [])
            )
            route_manager = getattr(scenario, "route_manager", None)
            unified_route_pid = GenericRoutePID(
                world,
                ego,
                target_speed_kmh=args.target_speed_kmh,
                fixed_delta_seconds=args.fixed_delta_s,
                route_manager=route_manager,
            )
            unified_vla = UniversalVLAController(
                world=world,
                ego=ego,
                route_controller=unified_route_pid,
                commands=commands,
                checkpoint_path=Path(
                    args.vla_checkpoint
                ).expanduser().resolve(),
                config_path=Path(
                    args.vla_config
                ).expanduser().resolve(),
                parser_model_path=Path(
                    args.command_parser_model
                ).expanduser().resolve(),
                output_path=Path(output_dir)
                / "vla_control_decisions.jsonl",
                device=args.vla_device,
                precision=args.vla_precision,
                decision_interval_frames=3,
                fixed_delta_seconds=args.fixed_delta_s,
                available_cameras=("front",),
                enable_lidar=False,
                default_speed_kmh=args.target_speed_kmh,
            )
            policy = None
            effective_decision_source = "universal_vla_controller"
            print("Scene 1 unified VLA controller assigned")
            # Give the sensor rig one complete cadence before the timed loop.
            for _ in range(6):
                world.tick()
        elif args.decision_source in {"voice_schedule", "voice_scene_bridge"} or args.scenario == "basic_voice_control_5km":
            create_policy = getattr(scenario, "create_temporary_policy", None)
            if create_policy is None:
                raise ValueError("voice_schedule requires a scenario with a temporary policy")
            schedule_policy = create_policy(
                args.target_speed_kmh,
                args.command_parser_model,
                args.command_parser_device,
            )
            if args.decision_source == "voice_scene_bridge":
                rule_policy = ScheduledSceneBridgePolicy(
                    schedule_policy,
                    args.bridge_output_dir or os.path.join(output_dir, "scene_bridge"),
                )
                policy = rule_policy
            else:
                policy = schedule_policy
            warmup = getattr(policy, "warmup", None)
            if callable(warmup):
                warmup()
            effective_decision_source = (
                "modernbert_voice_scene_bridge"
                if args.decision_source == "voice_scene_bridge"
                and args.command_parser_model
                else "voice_scene_bridge"
                if args.decision_source == "voice_scene_bridge"
                else "modernbert_voice_closed_loop"
                if args.command_parser_model
                else "configured_voice_schedule"
            )
        elif getattr(scenario, "create_decision_policy", None) is not None:
            policy = scenario.create_decision_policy()
            effective_decision_source = "scenario_route_policy"
        else:
            policy = RuleDecisionPolicy(args.scenario, args.target_speed_kmh)
        duration_s = args.duration_s
        if duration_s is None:
            duration_s = float(getattr(scenario, "default_duration_s", 25.0))
        scenario_info = call_scenario_method(scenario, "get_scenario_info", {})
        logger = ExperimentLogger(output_dir, {
            "scenario": args.scenario,
            "scenario_info": scenario_info,
            "controller": controller_name,
            "decision_source": effective_decision_source,
            "command_parser_model": args.command_parser_model,
            "decision_json": args.decision_json,
            "driving_intent_json": args.driving_intent_json,
            "bridge_output_dir": args.bridge_output_dir,
            "decision_max_age_frames": args.decision_max_age_frames,
            "decision_wait_ms": args.decision_wait_ms,
            "world_state_output": args.world_state_output,
            "scene_world_state_output": args.scene_world_state_output,
            "scenario_config": scenario_config_path,
            "resume_route_progress_m": args.resume_route_progress_m,
            "target_speed_kmh": args.target_speed_kmh,
            "fixed_delta_s": args.fixed_delta_s,
            "duration_s": duration_s,
            "carla_server": "{0}:{1}".format(args.host, args.port),
            "map": world.get_map().name,
            "camera": {
                "enabled": bool(args.record_images or args.video_output),
                "width": args.camera_width if (args.record_images or args.video_output) else None,
                "height": args.camera_height if (args.record_images or args.video_output) else None,
                "view": args.camera_view if (args.record_images or args.video_output) else None,
                "every_n_frames": args.record_every_n if args.record_images else None,
                "direct_video": args.video_output,
                "video_fps": args.video_fps if args.video_output else None,
                "video_profile": args.video_profile if args.video_output else None,
                "video_overlay": bool(args.video_overlay and args.video_output),
            },
            "scene_understanding_capture": {
                "enabled": bool(args.scene_capture or args.live_perception or args.async_qwen),
                "every_n_frames": args.scene_capture_every_n if (args.scene_capture or args.live_perception or args.async_qwen) else None,
                "width": args.scene_camera_width if (args.scene_capture or args.live_perception or args.async_qwen) else None,
                "height": args.scene_camera_height if (args.scene_capture or args.live_perception or args.async_qwen) else None,
            },
            "live_perception": {
                "enabled": bool(args.live_perception),
                "device": args.perception_device if args.live_perception else None,
                "image_size": args.perception_image_size if args.live_perception else None,
            },
            "async_qwen": {
                "enabled": bool(args.async_qwen),
                "model_path": args.qwen_model_path if args.async_qwen else None,
                "max_age_s": args.qwen_max_age_s if args.async_qwen else None,
                "finalize_timeout_s": args.qwen_finalize_timeout_s if args.async_qwen else None,
            },
        })
        logger.log_event({
            "type": "scenario_initialized",
            "scenario": args.scenario,
            "stale_project_vehicle_count_removed": stale_actor_count,
            "scenario_status": call_scenario_method(scenario, "get_status", {}),
        })
        for event in call_scenario_method(scenario, "drain_event_log", []):
            logger.log_event(event)
        start_location = ego.get_location()
        previous_location = start_location
        travelled_distance_m = 0.0
        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        sim_time = 0.0
        runner_stop_reason = "duration_limit"
        latest_scene_sensor_events = scene_sensor_events(None)
        previous_snapshot_time = start_sim_time
        control_delta_s = float(args.fixed_delta_s)
        previous_steer = None
        previous_steer_rate = None

        while previous_snapshot_time - start_sim_time < duration_s:
            scenario.fixed_delta_s = control_delta_s
            scenario.tick()
            state = WorldState(world, ego).get_state()
            decision_snapshot = world.get_snapshot()
            state["simulation_frame"] = int(decision_snapshot.frame)
            state["frame_id"] = "carla_{0}".format(int(decision_snapshot.frame))
            if args.world_state_output:
                write_json_atomically(args.world_state_output, {
                    "frame_id": state["frame_id"],
                    "simulation_frame": state["simulation_frame"],
                    "world_state": json_safe(state),
                })
            scene_world_state = None
            scene_capture_result = None
            live_perception_result = None
            async_qwen_submission = None
            async_qwen_result = None
            if scene_world_state_collector is not None:
                scene_world_state = scene_world_state_collector.collect(
                    sensor_events=latest_scene_sensor_events
                )
                if scene_capture is not None:
                    # This runs before policy.decide(), so a successful visual
                    # match is available to the same decision tick.
                    scene_capture_result = scene_capture.capture_current_frame()
                if live_perception is not None and scene_capture_result is not None:
                    live_perception_result = live_perception.process_capture(scene_capture_result)
                    enriched = live_perception_result.get("world_state")
                    if isinstance(enriched, dict):
                        scene_world_state = enriched
                    tracks = (live_perception_result.get("perception") or {}).get("tracks", [])
                    last_live_perception = {
                        "status": live_perception_result.get("status", "unknown"),
                        "latency_ms": live_perception_result.get("latency_ms", {}).get("total"),
                        "sim_time_s": sim_time,
                        "vehicle_count": sum(
                            1 for track in tracks if track.get("category") == "vehicle"
                        ),
                        "pedestrian_count": sum(
                            1 for track in tracks if track.get("category") == "pedestrian"
                        ),
                    }
                if async_qwen is not None:
                    if scene_capture_result is not None:
                        async_qwen_submission = async_qwen.submit_capture(scene_capture_result)
                    async_qwen_result = async_qwen.poll()
                    # A Qwen result may enrich only its own capture frame. In
                    # normal operation it completes later and is logged for
                    # asynchronous semantics; it is never applied to a newer
                    # control frame.
                    if (
                        isinstance(async_qwen_result, dict)
                        and async_qwen_result.get("frame_id") == scene_world_state.get("frame_id")
                        and isinstance(async_qwen_result.get("world_state"), dict)
                    ):
                        scene_world_state = async_qwen_result["world_state"]
                if args.scene_world_state_output:
                    write_json_atomically(args.scene_world_state_output, scene_world_state)
                set_scene_world_state = getattr(policy, "set_scene_world_state", None)
                if set_scene_world_state is not None:
                    set_scene_world_state(scene_world_state)
            set_context = getattr(policy, "set_context", None)
            if set_context is not None:
                set_context(call_scenario_method(scenario, "get_policy_context", {}))
            if unified_vla is not None:
                control = unified_vla.run_step()
                overlay = unified_vla.overlay()
                normalized_intent = {
                    "action": overlay.get("action", "keep_lane"),
                    "target_speed_kmh": overlay.get(
                        "target_speed_kmh", 0.0
                    ),
                    "emergency": bool(overlay.get("emergency", False)),
                    "command_id": (
                        unified_vla.active_command().get("id")
                        if unified_vla is not None
                        else None
                    ),
                    "request_id": f"unified-{int(decision_snapshot.frame)}",
                    "frame_id": f"carla_{int(decision_snapshot.frame)}",
                }
                decision_latency_ms = 0.0
                control_latency_ms = 0.0
            else:
                decision_start = time.perf_counter()
                intent = decide_with_optional_wait(
                    policy, state, args.decision_wait_ms
                )
                decision_latency_ms = (
                    time.perf_counter() - decision_start
                ) * 1000.0
                control_start = time.perf_counter()
                control, normalized_intent = controller.run_step(
                    intent, control_delta_s
                )
                control_latency_ms = (
                    time.perf_counter() - control_start
                ) * 1000.0
            call_scenario_method(scenario, "report_intent", None, normalized_intent)
            if control is not None:
                ego.apply_control(control)
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds - start_sim_time
            observed_delta_s = max(
                1e-6, snapshot.timestamp.elapsed_seconds - previous_snapshot_time
            )
            previous_snapshot_time = snapshot.timestamp.elapsed_seconds
            control_delta_s = observed_delta_s
            location = ego.get_location()
            travelled_distance_m += previous_location.distance(location)
            previous_location = location
            events = monitor.snapshot(int(snapshot.frame))
            latest_scene_sensor_events = scene_sensor_events(events)
            call_scenario_method(scenario, "report_events", None, events)
            step_feedback = None
            report_execution = getattr(policy, "report_execution", None)
            if report_execution is not None and scene_world_state_collector is not None:
                step_feedback = report_execution(
                    scene_world_state_collector.collect(sensor_events=latest_scene_sensor_events),
                    normalized_intent,
                    controller,
                )
            scenario_status = call_scenario_method(scenario, "get_status", {})
            scenario_metrics = scenario_status.get("metrics", {})
            events["illegal_lane_invasion_count"] = int(
                scenario_metrics.get(
                    "illegal_lane_invasion_count", events["lane_invasion_count"]
                )
            )
            applied_control = ego.get_control()
            applied_steer = float(applied_control.steer)
            steer_rate = (
                0.0
                if previous_steer is None
                else (applied_steer - previous_steer) / observed_delta_s
            )
            steer_accel = (
                0.0
                if previous_steer_rate is None
                else (steer_rate - previous_steer_rate) / observed_delta_s
            )
            previous_steer = applied_steer
            previous_steer_rate = steer_rate
            ego_waypoint = world.get_map().get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            policy_telemetry = call_scenario_method(policy, "telemetry", {})
            policy_trace = call_scenario_method(policy, "trace", {})
            bridge_telemetry = policy_telemetry.get("scene_bridge", policy_telemetry)
            scene_decision_latency_ms = bridge_telemetry.get("scene_decision_latency_ms")
            if scene_decision_latency_ms is None:
                for telemetry_key in ("producer", "scene_bridge"):
                    nested_telemetry = bridge_telemetry.get(telemetry_key, {})
                    if not isinstance(nested_telemetry, dict):
                        continue
                    scene_decision_latency_ms = nested_telemetry.get(
                        "scene_decision_latency_ms"
                    )
                    if scene_decision_latency_ms is not None:
                        break
            perception_latency_ms = None
            if live_perception_result is not None:
                perception_latency_ms = live_perception_result.get("latency_ms", {}).get("total")
            record = {
                "frame": int(snapshot.frame),
                "sim_time_s": round(sim_time, 4),
                "scenario": args.scenario,
                "decision_source": effective_decision_source,
                "scenario_status": scenario_status,
                "intent": normalized_intent,
                "control": {
                    "throttle": round(float(applied_control.throttle), 4),
                    "brake": round(float(applied_control.brake), 4),
                    "steer": round(applied_steer, 4),
                },
                "steering_dynamics": {
                    "normalized_steer": round(applied_steer, 6),
                    "steer_rate_per_s": round(steer_rate, 6),
                    "steer_accel_per_s2": round(steer_accel, 6),
                    "action": normalized_intent.get("action", "unknown"),
                    "controller_terms": dict(
                        getattr(
                            getattr(
                                getattr(
                                    unified_vla, "route_controller", None
                                )
                                or controller,
                                "_pid",
                                None,
                            )
                            or controller,
                            "_last_lateral_debug",
                            {},
                        )
                    ),
                },
                "ego": {
                    "speed_kmh": round(get_speed_kmh(ego), 4),
                    "location": {"x": round(location.x, 3), "y": round(location.y, 3), "z": round(location.z, 3)},
                    "yaw_deg": round(float(ego.get_transform().rotation.yaw), 4),
                    "road": {
                        "road_id": int(ego_waypoint.road_id) if ego_waypoint is not None else None,
                        "section_id": int(ego_waypoint.section_id) if ego_waypoint is not None else None,
                        "lane_id": int(ego_waypoint.lane_id) if ego_waypoint is not None else None,
                        "is_junction": bool(ego_waypoint.is_junction) if ego_waypoint is not None else None,
                    },
                },
                "distance_m": round(travelled_distance_m, 4),
                "traffic_visibility": visible_traffic_probe(world, ego),
                "events": events,
                "policy": policy_telemetry,
                "scene_decision": policy_trace,
                "latency_ms": {
                    "decision": round(decision_latency_ms, 4),
                    "control": round(control_latency_ms, 4),
                    "end_to_end": round(decision_latency_ms + control_latency_ms, 4),
                    "scene_decision": scene_decision_latency_ms,
                    "perception": perception_latency_ms,
                },
            }
            if step_feedback is not None:
                record["step_feedback"] = step_feedback
            if scene_capture_result is not None:
                record["scene_capture"] = scene_capture_result
            if live_perception_result is not None:
                record["scene_understanding"] = live_perception_result
            if last_live_perception is not None:
                latest_perception = dict(last_live_perception)
                latest_perception["age_s"] = round(
                    max(0.0, sim_time - latest_perception["sim_time_s"]), 3
                )
                record["scene_understanding_latest"] = latest_perception
            if async_qwen is not None:
                record["async_qwen"] = {
                    "submission": async_qwen_submission,
                    "result": async_qwen_result,
                    "worker": async_qwen.stats(),
                }
            if camera is not None:
                camera.save_frame(
                    snapshot.frame,
                    overlay=make_video_overlay(record) if args.video_overlay else None,
                )
            logger.log_frame(record)
            for event in call_scenario_method(scenario, "drain_event_log", []):
                logger.log_event(event)
            records.append(record)
            if scenario.finished():
                runner_stop_reason = "scenario_{0}".format(
                    call_scenario_method(scenario, "get_status", {}).get("status", "finished").lower()
                )
                break
            if args.stop_when_goal_reached and args.goal_distance_m is not None:
                if travelled_distance_m >= args.goal_distance_m:
                    runner_stop_reason = "external_goal_distance_reached"
                    break
            if args.stop_at_route_progress_m is not None:
                route_progress_m = record["scenario_status"].get("route_progress_m")
                if route_progress_m is not None and route_progress_m >= args.stop_at_route_progress_m:
                    runner_stop_reason = "route_checkpoint_reached"
                    break

        scenario_goal_distance_m = call_scenario_method(
            scenario, "get_goal_distance_m", args.goal_distance_m
        )
        if async_qwen is not None and args.qwen_finalize_timeout_s > 0:
            final_qwen_result = async_qwen.wait_for_result(
                args.qwen_finalize_timeout_s
            )
            if final_qwen_result is not None:
                logger.log_event({
                    "type": "async_qwen_finalized",
                    "result": final_qwen_result,
                })
        metrics = summarize(records, args.scenario, scenario_goal_distance_m)
        final_status = call_scenario_method(scenario, "get_status", {})
        if (
            camera is not None
            and hasattr(camera, "append_terminal_overlay")
            and final_status.get("status") in ("SUCCESS", "FAILURE")
        ):
            terminal_record = dict(records[-1]) if records else {"scenario": args.scenario}
            terminal_record["scenario_status"] = final_status
            camera.append_terminal_overlay(
                make_video_overlay(terminal_record), args.terminal_hold_s
            )
        metrics["scenario_status"] = final_status
        metrics["runner_stop_reason"] = runner_stop_reason
        if unified_vla is not None:
            metrics["universal_vla_controller"] = unified_vla.summary()
        if scene_capture is not None:
            metrics["scene_understanding_capture"] = scene_capture.stats()
        if async_qwen is not None:
            metrics["async_qwen"] = async_qwen.stats()
        if final_status.get("status") in ("SUCCESS", "FAILURE"):
            metrics["task_completed"] = (
                final_status["status"] == "SUCCESS"
                and metrics.get("violation_free", False)
            )
            metrics["scenario_reason"] = final_status.get("reason", "")
        else:
            metrics["task_completed"] = False
            metrics["scenario_reason"] = runner_stop_reason
        logger.write_summary(metrics)
        print("[Done] Metrics written to {0}".format(output_dir))
        print(metrics)
    finally:
        if unified_vla is not None:
            try:
                unified_vla.close()
            except RuntimeError:
                pass
        if policy is not None:
            close_policy = getattr(policy, "close", None)
            if callable(close_policy):
                close_policy()
        if logger is not None:
            logger.close()
        if monitor is not None:
            monitor.destroy()
        if scene_capture is not None:
            scene_capture.destroy()
        if async_qwen is not None:
            async_qwen.close()
        if camera is not None:
            camera.destroy()
        call_scenario_method(scenario, "restore_runtime", None)
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            # Scenario cleanup can invalidate the old World proxy on CARLA map
            # transitions. Reacquire it before restoring synchronous settings.
            client.get_world().apply_settings(original_settings)
        scenario.destroy()


if __name__ == "__main__":
    main()
