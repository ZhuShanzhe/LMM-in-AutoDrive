"""Run one CARLA scenario with control, logging, sensors, and metric export.

Run from this directory after CARLA is started:
    python run_control_experiment.py emergency_brake --duration-s 25
"""

import argparse
import math
import os
import time

from carla_bootstrap import setup_carla_api

setup_carla_api()

import carla

from control.agents import CarlaAgentController
from control.pid_controller import EgoPIDController
from evaluation.events import EventMonitor
from evaluation.camera import ExperimentCamera
from evaluation.logger import ExperimentLogger
from evaluation.metrics import summarize
from perception.world_state import WorldState
from scenarios.basic.straight_driving import StraightDrivingScenario
from scenarios.emergency.emergency_brake import EmergencyBrakeScenario
from scenarios.pedestrian.pedestrian_crossing import PedestrianCrossingScenario


SCENARIOS = {
    "straight_driving": StraightDrivingScenario,
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
        front_braking = (
            front_vehicle is not None
            and front_distance < 30.0
            and ego_speed_kmh - front_vehicle.get("speed_kmh", ego_speed_kmh) > 5.0
        )
        if front_distance < 12.0 or pedestrian_distance < 10.0 or front_braking:
            return {
                "action": "emergency_brake",
                "target_speed_kmh": 0.0,
                "emergency": True,
                "reason": "front_vehicle_braking" if front_braking else "rule_safety_distance",
            }
        return {
            "action": "keep_lane",
            "target_speed_kmh": self.cruise_speed_kmh,
            "reason": "rule_cruise",
        }


def build_controller(name, vehicle, world_map, target_speed_kmh):
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


def call_scenario_method(scenario, method_name, default=None):
    method = getattr(scenario, method_name, None)
    if method is None:
        return default
    try:
        return json_safe(method())
    except Exception as exc:
        return {"error": "{0}: {1}".format(type(exc).__name__, exc)}


def make_video_overlay(record):
    status = record.get("scenario_status", {})
    events = record.get("events", {})
    intent = record.get("intent", {})
    control = record.get("control", {})
    ego = record.get("ego", {})
    return [
        "scenario={0} frame={1} sim={2:.2f}s".format(
            record.get("scenario", ""), record.get("frame", 0), record.get("sim_time_s", 0.0)
        ),
        "intent={0} target={1:.1f}km/h emergency={2}".format(
            intent.get("action", ""), intent.get("target_speed_kmh", 0.0), intent.get("emergency", False)
        ),
        "ego={0:.1f}km/h throttle={1:.2f} brake={2:.2f} steer={3:.2f}".format(
            ego.get("speed_kmh", 0.0), control.get("throttle", 0.0), control.get("brake", 0.0), control.get("steer", 0.0)
        ),
        "status={0} reason={1} collisions={2} lane_events={3}".format(
            status.get("status", ""), status.get("reason", ""),
            events.get("collision_count", 0), events.get("lane_invasion_count", 0),
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA control and evaluation runner")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--fixed-delta-s", type=float, default=0.05)
    parser.add_argument("--target-speed-kmh", type=float, default=25.0)
    parser.add_argument("--controller", choices=["pid", "basic", "behavior"], default="pid")
    parser.add_argument("--goal-distance-m", type=float, default=None)
    parser.add_argument("--stop-when-goal-reached", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--record-every-n", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--video-output", default=None, help="Optional direct H.264 output path")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--ffmpeg", default=None, help="Path to ffmpeg.exe for --video-output")
    parser.add_argument("--video-overlay", action="store_true", help="Overlay per-frame run telemetry on direct video")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join("outputs", "runs", "{0}_{1}".format(args.scenario, time.strftime("%Y%m%d_%H%M%S")))
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.fixed_delta_s
    world.apply_settings(settings)

    scenario = SCENARIOS[args.scenario](world, external_control=True)
    monitor = None
    camera = None
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
            )
            camera.start()
        controller = build_controller(args.controller, ego, world.get_map(), args.target_speed_kmh)
        policy = RuleDecisionPolicy(args.scenario, args.target_speed_kmh)
        scenario_info = call_scenario_method(scenario, "get_scenario_info", {})
        logger = ExperimentLogger(output_dir, {
            "scenario": args.scenario,
            "scenario_info": scenario_info,
            "controller": args.controller,
            "target_speed_kmh": args.target_speed_kmh,
            "fixed_delta_s": args.fixed_delta_s,
            "carla_server": "{0}:{1}".format(args.host, args.port),
            "camera": {
                "enabled": bool(args.record_images or args.video_output),
                "width": args.camera_width if (args.record_images or args.video_output) else None,
                "height": args.camera_height if (args.record_images or args.video_output) else None,
                "every_n_frames": args.record_every_n if args.record_images else None,
                "direct_video": args.video_output,
                "video_fps": args.video_fps if args.video_output else None,
                "video_overlay": bool(args.video_overlay and args.video_output),
            },
        })
        start_location = ego.get_location()
        previous_location = start_location
        travelled_distance_m = 0.0
        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        max_ticks = int(args.duration_s / args.fixed_delta_s)

        for _ in range(max_ticks):
            scenario.tick()
            state = WorldState(world, ego).get_state()
            decision_start = time.perf_counter()
            intent = policy.decide(state)
            decision_latency_ms = (time.perf_counter() - decision_start) * 1000.0
            control_start = time.perf_counter()
            control, normalized_intent = controller.run_step(intent, args.fixed_delta_s)
            control_latency_ms = (time.perf_counter() - control_start) * 1000.0
            ego.apply_control(control)
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds - start_sim_time
            location = ego.get_location()
            travelled_distance_m += previous_location.distance(location)
            previous_location = location
            record = {
                "frame": int(snapshot.frame),
                "sim_time_s": round(sim_time, 4),
                "scenario": args.scenario,
                "scenario_status": call_scenario_method(scenario, "get_status", {}),
                "intent": normalized_intent,
                "control": {
                    "throttle": round(float(control.throttle), 4),
                    "brake": round(float(control.brake), 4),
                    "steer": round(float(control.steer), 4),
                },
                "ego": {
                    "speed_kmh": round(get_speed_kmh(ego), 4),
                    "location": {"x": round(location.x, 3), "y": round(location.y, 3), "z": round(location.z, 3)},
                },
                "distance_m": round(travelled_distance_m, 4),
                "events": monitor.snapshot(int(snapshot.frame)),
                "latency_ms": {
                    "decision": round(decision_latency_ms, 4),
                    "control": round(control_latency_ms, 4),
                    "end_to_end": round(decision_latency_ms + control_latency_ms, 4),
                },
            }
            if camera is not None:
                camera.save_frame(
                    snapshot.frame,
                    overlay_lines=make_video_overlay(record) if args.video_overlay else None,
                )
            logger.log_frame(record)
            records.append(record)
            if scenario.finished():
                break
            if args.stop_when_goal_reached and args.goal_distance_m is not None:
                if travelled_distance_m >= args.goal_distance_m:
                    break

        metrics = summarize(records, args.scenario, args.goal_distance_m)
        final_status = call_scenario_method(scenario, "get_status", {})
        metrics["scenario_status"] = final_status
        if final_status.get("status") in ("SUCCESS", "FAILURE"):
            metrics["task_completed"] = (
                final_status["status"] == "SUCCESS"
                and metrics.get("violation_free", False)
            )
            metrics["scenario_reason"] = final_status.get("reason", "")
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
        scenario.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
