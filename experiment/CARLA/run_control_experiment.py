"""Run one CARLA scenario with control, logging, sensors, and metric export.

Run from this directory after CARLA is started:
    python run_control_experiment.py emergency_brake --duration-s 25
"""

import argparse
import json
import math
import os
import tempfile
import time

from carla_bootstrap import setup_carla_api

setup_carla_api()

import carla

from control.agents import CarlaAgentController
from control.decision_provider import JsonFileDecisionPolicy
from control.pid_controller import EgoPIDController
from evaluation.events import EventMonitor
from evaluation.camera import ExperimentCamera
from evaluation.logger import ExperimentLogger
from evaluation.metrics import summarize
from perception.world_state import WorldState
from scene_understanding_capture import SceneUnderstandingCapture
from scenarios.basic.straight_driving import StraightDrivingScenario
from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario
from scenarios.continuous.basic_track_5km import BasicTrack5KmScenario
from scenarios.emergency.emergency_brake import EmergencyBrakeScenario
from scenarios.pedestrian.pedestrian_crossing import PedestrianCrossingScenario


SCENARIOS = {
    "straight_driving": StraightDrivingScenario,
    "basic_voice_control_5km": BasicVoiceControl5KmScenario,
    "basic_track_5km": BasicTrack5KmScenario,
    "emergency_brake": EmergencyBrakeScenario,
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


def build_controller(name, vehicle, world_map, target_speed_kmh, scenario=None):
    custom_factory = getattr(scenario, "create_controller", None)
    if custom_factory is not None:
        return custom_factory()
    if name == "pid":
        return EgoPIDController(vehicle, world_map, target_speed_kmh)
    return CarlaAgentController(vehicle, mode=name)


def get_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


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


def make_video_overlay(record):
    status = record.get("scenario_status", {})
    events = record.get("events", {})
    intent = record.get("intent", {})
    control = record.get("control", {})
    ego = record.get("ego", {})
    scenario_name = record.get("scenario", "")
    decision_source = record.get("decision_source", "rule")
    action = intent.get("action", "")
    emergency = bool(intent.get("emergency", False))
    if intent.get("voice_text"):
        asr_text = intent["voice_text"]
    elif decision_source == "json_file":
        asr_text = "保持安全车距行驶"
    elif scenario_name == "pedestrian_crossing":
        asr_text = "前方行人横穿，减速避让"
    elif scenario_name == "emergency_brake":
        asr_text = "前车紧急制动，立即刹车" if emergency else "保持车距，正常行驶"
    else:
        asr_text = "保持当前车道，匀速前进至终点"
    if emergency:
        risk_level = "HIGH"
        policy_state = "EMERGENCY_BRAKING"
    elif action in ("decelerate", "stop"):
        risk_level = "MEDIUM"
        policy_state = "DECELERATING"
    else:
        risk_level = "LOW"
        policy_state = "NORMAL_DRIVING"
    if status.get("status") == "SUCCESS":
        policy_state = "COMPLETED"
    elif status.get("status") == "FAILURE":
        policy_state = "FAILED"
    return {
        "scenario": scenario_name,
        "frame": record.get("frame", 0),
        "sim_time_s": record.get("sim_time_s", 0.0),
        "asr_text": asr_text,
        "decision_source": decision_source,
        "action": action,
        "reason": intent.get("reason", "") or status.get("reason", ""),
        "target_speed_kmh": intent.get("target_speed_kmh", 0.0),
        "emergency": emergency,
        "risk_level": risk_level,
        "policy_state": policy_state,
        "speed_kmh": ego.get("speed_kmh", 0.0),
        "throttle": control.get("throttle", 0.0),
        "brake": control.get("brake", 0.0),
        "steer": control.get("steer", 0.0),
        "status": status.get("status", "RUNNING"),
        "collisions": events.get("collision_count", 0),
        "lane_events": events.get("lane_invasion_count", 0),
        "route_progress_m": status.get("route_progress_m"),
        "route_length_m": status.get("route_length_m"),
        "traffic_count": status.get("traffic", {}).get("background_actor_count", 0),
        "pedestrian_count": status.get("pedestrians", {}).get("walker_count", 0),
        "active_events": status.get("scenario_events", {}).get("active", []),
    }


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
    parser.add_argument("--map", default=None, help="Optional CARLA map name to load before setup")
    parser.add_argument("--scenario-config", default=None, help="Optional JSON configuration for a configurable scenario")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--fixed-delta-s", type=float, default=0.05)
    parser.add_argument("--target-speed-kmh", type=float, default=25.0)
    parser.add_argument("--controller", choices=["pid", "basic", "behavior"], default="pid")
    parser.add_argument(
        "--decision-source",
        choices=["rule", "json_file", "voice_schedule"],
        default="rule",
        help="Use built-in rules or a per-tick external decision JSON file",
    )
    parser.add_argument(
        "--decision-json",
        default=None,
        help="DrivingIntent or ControlDecision JSON path for --decision-source json_file",
    )
    parser.add_argument(
        "--world-state-output",
        default=None,
        help="Optional per-tick world-state JSON path for an external decision process",
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
    parser.add_argument("--video-overlay", action="store_true", help="Overlay per-frame run telemetry on direct video")
    parser.add_argument("--terminal-hold-s", type=float, default=2.0, help="Seconds to hold SUCCESS/FAILURE video frame")
    parser.add_argument(
        "--scene-capture",
        action="store_true",
        help="Write frame-aligned scene_understanding capture bundles",
    )
    parser.add_argument("--scene-capture-every-n", type=int, default=10)
    parser.add_argument("--scene-camera-width", type=int, default=800)
    parser.add_argument("--scene-camera-height", type=int, default=600)
    parser.add_argument("--scene-camera-timeout-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.decision_source == "json_file" and not args.decision_json:
        parser.error("--decision-json is required when --decision-source json_file")
    if args.resume_route_progress_m is not None and args.scenario_config is None:
        parser.error("--resume-route-progress-m requires --scenario-config")
    if args.resume_route_progress_m is not None and args.resume_route_progress_m < 0.0:
        parser.error("--resume-route-progress-m must be non-negative")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join("outputs", "runs", "{0}_{1}".format(args.scenario, time.strftime("%Y%m%d_%H%M%S")))
    scenario_config_path = resolve_scenario_config(
        args.scenario_config,
        output_dir,
        args.resume_route_progress_m,
    )
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    scenario_class = SCENARIOS[args.scenario]
    config_map = None
    if scenario_config_path is not None:
        with open(scenario_config_path, "r", encoding="utf-8") as handle:
            config_map = json.load(handle).get("map")
    target_map = args.map or config_map or getattr(scenario_class, "default_map", None)
    if target_map and not world.get_map().name.endswith(target_map):
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

    scenario_kwargs = {"external_control": True}
    if scenario_config_path is not None:
        scenario_kwargs["config_path"] = scenario_config_path
    scenario = scenario_class(world, **scenario_kwargs)
    scenario.client = client
    scenario.fixed_delta_s = args.fixed_delta_s
    monitor = None
    camera = None
    scene_capture = None
    logger = None
    records = []
    try:
        scenario.setup()
        world.tick()
        ego = scenario.get_ego_vehicle()
        monitor = EventMonitor(world, ego)
        monitor.start()
        if args.record_images or args.video_output:
            camera = ExperimentCamera(
                world,
                ego,
                os.path.join(output_dir, "camera_frames"),
                args.record_every_n,
                args.camera_width,
                args.camera_height,
                args.record_images,
                args.video_output,
                args.video_fps,
                args.ffmpeg,
                args.video_overlay,
                args.fixed_delta_s * args.record_every_n,
                args.camera_view,
            )
            camera.start()
        if args.scene_capture:
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
        controller = build_controller(
            args.controller,
            ego,
            world.get_map(),
            args.target_speed_kmh,
            scenario,
        )
        controller_name = type(controller).__name__
        effective_decision_source = args.decision_source
        if args.decision_source == "json_file":
            policy = JsonFileDecisionPolicy(args.decision_json, args.target_speed_kmh)
        elif args.decision_source == "voice_schedule" or args.scenario == "basic_voice_control_5km":
            create_policy = getattr(scenario, "create_temporary_policy", None)
            if create_policy is None:
                raise ValueError("voice_schedule requires a scenario with a temporary policy")
            policy = create_policy(args.target_speed_kmh)
            effective_decision_source = "temporary_voice_schedule"
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
            "decision_json": args.decision_json,
            "world_state_output": args.world_state_output,
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
                "video_overlay": bool(args.video_overlay and args.video_output),
            },
            "scene_understanding_capture": {
                "enabled": bool(args.scene_capture),
                "every_n_frames": args.scene_capture_every_n if args.scene_capture else None,
                "width": args.scene_camera_width if args.scene_capture else None,
                "height": args.scene_camera_height if args.scene_capture else None,
            },
        })
        logger.log_event({
            "type": "scenario_initialized",
            "scenario": args.scenario,
            "scenario_status": call_scenario_method(scenario, "get_status", {}),
        })
        for event in call_scenario_method(scenario, "drain_event_log", []):
            logger.log_event(event)
        start_location = ego.get_location()
        previous_location = start_location
        travelled_distance_m = 0.0
        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        max_ticks = int(duration_s / args.fixed_delta_s)
        runner_stop_reason = "duration_limit"

        for _ in range(max_ticks):
            scenario.tick()
            state = WorldState(world, ego).get_state()
            if args.world_state_output:
                snapshot = world.get_snapshot()
                write_json_atomically(args.world_state_output, {
                    "frame_id": "carla_{0}".format(int(snapshot.frame)),
                    "world_state": json_safe(state),
                })
            set_context = getattr(policy, "set_context", None)
            if set_context is not None:
                set_context(call_scenario_method(scenario, "get_policy_context", {}))
            decision_start = time.perf_counter()
            intent = policy.decide(state)
            decision_latency_ms = (time.perf_counter() - decision_start) * 1000.0
            control_start = time.perf_counter()
            control, normalized_intent = controller.run_step(intent, args.fixed_delta_s)
            control_latency_ms = (time.perf_counter() - control_start) * 1000.0
            call_scenario_method(scenario, "report_intent", None, normalized_intent)
            if control is not None:
                ego.apply_control(control)
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds - start_sim_time
            location = ego.get_location()
            travelled_distance_m += previous_location.distance(location)
            previous_location = location
            events = monitor.snapshot(int(snapshot.frame))
            call_scenario_method(scenario, "report_events", None, events)
            scenario_status = call_scenario_method(scenario, "get_status", {})
            scenario_metrics = scenario_status.get("metrics", {})
            events["illegal_lane_invasion_count"] = int(
                scenario_metrics.get(
                    "illegal_lane_invasion_count", events["lane_invasion_count"]
                )
            )
            applied_control = ego.get_control()
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
                    "steer": round(float(applied_control.steer), 4),
                },
                "ego": {
                    "speed_kmh": round(get_speed_kmh(ego), 4),
                    "location": {"x": round(location.x, 3), "y": round(location.y, 3), "z": round(location.z, 3)},
                },
                "distance_m": round(travelled_distance_m, 4),
                "events": events,
                "policy": call_scenario_method(policy, "telemetry", {}),
                "latency_ms": {
                    "decision": round(decision_latency_ms, 4),
                    "control": round(control_latency_ms, 4),
                    "end_to_end": round(decision_latency_ms + control_latency_ms, 4),
                },
            }
            if scene_capture is not None:
                capture_result = scene_capture.capture_current_frame()
                if capture_result is not None:
                    record["scene_capture"] = capture_result
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
        metrics = summarize(records, args.scenario, scenario_goal_distance_m)
        final_status = call_scenario_method(scenario, "get_status", {})
        if camera is not None and final_status.get("status") in ("SUCCESS", "FAILURE"):
            terminal_record = dict(records[-1]) if records else {"scenario": args.scenario}
            terminal_record["scenario_status"] = final_status
            camera.append_terminal_overlay(
                make_video_overlay(terminal_record), args.terminal_hold_s
            )
        metrics["scenario_status"] = final_status
        metrics["runner_stop_reason"] = runner_stop_reason
        if scene_capture is not None:
            metrics["scene_understanding_capture"] = scene_capture.stats()
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
        if logger is not None:
            logger.close()
        if monitor is not None:
            monitor.destroy()
        if camera is not None:
            camera.destroy()
        if scene_capture is not None:
            scene_capture.destroy()
        scenario.destroy()
        world.apply_settings(original_settings)
        call_scenario_method(scenario, "restore_runtime", None)


if __name__ == "__main__":
    main()
