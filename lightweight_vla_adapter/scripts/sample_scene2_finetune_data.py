"""Sample Town05 Scene-2 traffic/event data for V6 risk-head fine-tuning.

The sampler drives the ego through scripted dense-traffic episodes (slow
vehicle, crossing pedestrian, cyclist, bus stop) and records synchronized
four-view RGB + real LiDAR BEV samples.  Risk labels are derived from actor
truth ONLY for offline training labels (never used online): high when a
vehicle/VRU is within 12 m ahead, medium within 25 m, low otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import carla
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENT_CARLA = REPO_ROOT / "experiment" / "CARLA"
if str(EXPERIMENT_CARLA) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_CARLA))

from carla_bootstrap import setup_carla_api  # noqa: E402
from carla_multiview_sensor import (  # noqa: E402
    CAMERA_ORDER,
    SynchronizedMultiviewCameraRig,
)

setup_carla_api()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/root/autodl-tmp/datasets/training")
            / "universal_three_scene_v6_finetune_scene2"
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/modernbert-drive-command-compositional"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--town", default="Town05_Opt")
    return parser.parse_args()


def environment_features(world: Any) -> list[float]:
    weather = world.get_weather()
    return [
        float(weather.cloudiness) / 100.0,
        float(weather.precipitation) / 100.0,
        float(weather.precipitation_deposits) / 100.0,
        float(weather.wind_intensity) / 100.0,
        float(weather.sun_azimuth_angle) / 360.0,
        max(-1.0, min(1.0, float(weather.sun_altitude_angle) / 90.0)),
        float(weather.fog_density) / 100.0,
        float(weather.fog_distance) / 1000.0,
        0.02,
        float(weather.wetness) / 100.0,
        0.10,
        0.80,
        0.32,
        0.32,
    ]


def ego_features(ego: Any, waypoint: Any, speed_cap_kmh: float) -> list[float]:
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    angular = ego.get_angular_velocity()
    control = ego.get_control()
    return [
        math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2),
        math.sqrt(
            float(acceleration.x) ** 2 + float(acceleration.y) ** 2
        ),
        math.radians(float(angular.z)),
        float(control.steer),
        float(control.throttle),
        float(control.brake),
        float(speed_cap_kmh) / 3.6,
        float(bool(waypoint is not None and waypoint.is_junction)),
    ]


def speed_kmh(actor: Any) -> float:
    try:
        velocity = actor.get_velocity()
        return 3.6 * math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
    except (RuntimeError, AttributeError):
        return 0.0


def alive(actor: Any) -> bool:
    try:
        return actor is not None and bool(actor.is_alive)
    except (RuntimeError, AttributeError):
        return False


def follow_road(ego: Any, world: Any, target_speed_kmh: float) -> None:
    """Simple road follower: steer toward the lane 8 m ahead and hold speed."""

    try:
        waypoint = world.get_map().get_waypoint(
            ego.get_location(), project_to_road=True
        )
    except (RuntimeError, AttributeError):
        return
    if waypoint is None:
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return
    try:
        candidates = waypoint.next(8.0)
        target = candidates[0] if candidates else waypoint
        current_yaw = math.radians(ego.get_transform().rotation.yaw)
        target_yaw = math.radians(target.transform.rotation.yaw)
    except (RuntimeError, AttributeError):
        return
    error = math.atan2(
        math.sin(target_yaw - current_yaw),
        math.cos(target_yaw - current_yaw),
    )
    steer = max(-0.5, min(0.5, 0.8 * error))
    current = speed_kmh(ego)
    speed_error = float(target_speed_kmh) - current
    if speed_error > 1.0:
        throttle, brake = 0.45, 0.0
    elif speed_error < -2.0:
        throttle, brake = 0.0, 0.72 if speed_error < -6.0 else 0.55
    else:
        throttle, brake = 0.12, 0.0
    ego.apply_control(
        carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
    )


def forward_lateral(ego: Any, actor: Any) -> tuple[float, float]:
    try:
        ego_transform = ego.get_transform()
        actor_location = actor.get_location()
        dx = actor_location.x - ego_transform.location.x
        dy = actor_location.y - ego_transform.location.y
        yaw = math.radians(ego_transform.rotation.yaw)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return float(forward), float(lateral)
    except (RuntimeError, AttributeError):
        return 1e9, 0.0


def label_risk(ego: Any, world: Any) -> dict[str, Any]:
    """Offline-only physics-based risk labels from actor truth.

    Risk integrates longitudinal gap, relative speed, TTC, ego stopping
    distance, short-horizon trajectory intersection, actor type and lane
    topology.  Target-lane risk also checks fast-approaching rear vehicles.
    """

    reaction_time_s = 1.0
    comfort_decel_mps2 = 4.0
    ego_velocity = ego.get_velocity()
    ego_speed_mps = max(
        0.0,
        math.sqrt(
            float(ego_velocity.x) ** 2
            + float(ego_velocity.y) ** 2
            + float(ego_velocity.z) ** 2
        ),
    )
    ego_stopping_m = (
        ego_speed_mps * reaction_time_s
        + ego_speed_mps * ego_speed_mps / (2.0 * comfort_decel_mps2)
    )
    ego_yaw = math.radians(ego.get_transform().rotation.yaw)

    front = None  # (rank, ttc, gap, rel_speed, is_vru)
    lanes = {"left": None, "right": None}

    def rank_of(level: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}[level]

    for actor in world.get_actors():
        if actor.id == ego.id:
            continue
        if not alive(actor):
            continue
        try:
            type_id = actor.type_id
            actor_velocity = actor.get_velocity()
        except (RuntimeError, AttributeError):
            continue
        is_vru = (
            "walker" in type_id
            or "bicycle" in type_id
            or "bike" in type_id
        )
        forward, lateral = forward_lateral(ego, actor)
        if forward >= 1e9:
            continue
        distance = math.hypot(forward, lateral)
        target_speed_along = (
            float(actor_velocity.x) * math.cos(ego_yaw)
            + float(actor_velocity.y) * math.sin(ego_yaw)
        )
        rel_speed_mps = ego_speed_mps - target_speed_along
        ttc = forward / max(rel_speed_mps, 0.5)
        projected_gap_15 = forward - max(0.0, rel_speed_mps) * 1.5
        margin = forward - ego_stopping_m

        level = "low"
        if forward > 0.5:
            if (
                ttc < 1.8
                or margin < 0.0
                or projected_gap_15 < 1.0
                or (is_vru and forward < 12.0)
            ):
                level = "high"
            elif (
                ttc < 4.0
                or (forward < 25.0 and rel_speed_mps > 1.0)
                or (is_vru and forward < 25.0)
            ):
                level = "medium"

        rank = rank_of(level)
        if abs(lateral) < 2.6 and 0.5 < forward < 60.0:
            if front is None or rank > front[0]:
                front = (rank, ttc, forward, rel_speed_mps, is_vru)

        lane_level = None
        if forward > 0.5 and forward < 40.0:
            if ttc < 2.2 or (is_vru and forward < 15.0):
                lane_level = "high"
            elif ttc < 5.0 or forward < 25.0:
                lane_level = "medium"
        elif forward < 0.0:
            behind = -forward
            if behind < 40.0 and rel_speed_mps < -3.0:
                lane_level = "high" if behind < 20.0 else "medium"
        if lane_level is not None:
            if -5.5 <= lateral <= -2.2 and (
                lanes["left"] is None
                or rank_of(lane_level) > rank_of(lanes["left"])
            ):
                lanes["left"] = lane_level
            if 2.2 <= lateral <= 5.5 and (
                lanes["right"] is None
                or rank_of(lane_level) > rank_of(lanes["right"])
            ):
                lanes["right"] = lane_level

    if front is None:
        risk_level = "low"
        ttc_s = None
        gap_m = None
        rel_kmh = None
        vru_front = False
    else:
        rank, ttc_s, gap_m, rel_speed, vru_front = front
        risk_level = {0: "low", 1: "medium", 2: "high"}[rank]
        rel_kmh = rel_speed * 3.6

    if risk_level == "high":
        if vru_front:
            action_hint = "stop"
        elif ttc_s is not None and ttc_s < 1.2:
            action_hint = "emergency_brake"
        elif rel_kmh is not None and rel_kmh < 1.0:
            action_hint = "stop"
        else:
            action_hint = "decelerate"
    elif risk_level == "medium":
        action_hint = "decelerate"
    else:
        action_hint = "keep_lane"

    if ttc_s is None:
        risk_score = 0.0
    else:
        risk_score = max(
            0.0,
            min(1.0, 1.0 - float(ttc_s) / 4.0),
        )
        if vru_front:
            risk_score = min(1.0, risk_score * 1.15 + 0.15)
    horizon_probabilities = []
    for horizon_s in (0.5, 1.0, 2.0, 3.0):
        if ttc_s is None or ttc_s <= 0.0:
            horizon_probabilities.append(0.0)
        else:
            horizon_probabilities.append(
                max(0.0, min(1.0, 1.0 - float(ttc_s) / horizon_s))
            )
    lane_risk_levels = [
        {"low": 0, "medium": 1, "high": 2}[lanes[direction] or "low"]
        for direction in ("left", "right")
    ]

    return {
        "risk_level": risk_level,
        "action_hint": action_hint,
        "risk_score": round(risk_score, 4),
        "horizon_probabilities": [round(v, 4) for v in horizon_probabilities],
        "lane_risk_levels": lane_risk_levels,
        "ttc_s": ttc_s,
        "gap_m": gap_m,
        "rel_speed_kmh": rel_kmh,
        "front_distance_m": gap_m,
        "lane_change": {
            direction: {
                "risk_level": lanes[direction] or "low",
                "is_safe": (lanes[direction] or "low") == "low",
            }
            for direction in ("left", "right")
        },
    }


def spawn_actor(world: Any, blueprint_name: str, transform: Any) -> Any:
    try:
        blueprint = world.get_blueprint_library().find(blueprint_name)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "sampler_npc")
        actor = world.spawn_actor(blueprint, transform)
        return actor
    except (RuntimeError, AttributeError):
        return None


def pick_road_ahead(world: Any, ego: Any, distance_m: float) -> carla.Transform:
    waypoint = world.get_map().get_waypoint(
        ego.get_location(), project_to_road=True
    )
    candidates = waypoint.next(max(1.0, float(distance_m)))
    if candidates:
        waypoint = candidates[0]
    transform = waypoint.transform
    transform.location.z += 0.6
    return transform


WEATHER_PROFILES = {
    "clear_noon": carla.WeatherParameters.ClearNoon,
    "cloudy_noon": carla.WeatherParameters.CloudyNoon,
    "foggy_morning": carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        fog_density=70.0,
        fog_distance=12.0,
        sun_altitude_angle=25.0,
    ),
    "wet_cloudy": carla.WeatherParameters(
        cloudiness=60.0,
        precipitation=20.0,
        precipitation_deposits=70.0,
        wind_intensity=20.0,
        sun_altitude_angle=50.0,
    ),
    "rainy_night": carla.WeatherParameters(
        cloudiness=90.0,
        precipitation=80.0,
        precipitation_deposits=90.0,
        wind_intensity=40.0,
        sun_altitude_angle=-12.0,
        fog_density=15.0,
    ),
    "sunset": carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_altitude_angle=8.0,
        sun_azimuth_angle=200.0,
    ),
}


CURRENT_WEATHER = "clear_noon"


def set_weather(world: Any, name: str) -> None:
    global CURRENT_WEATHER
    CURRENT_WEATHER = name
    world.set_weather(WEATHER_PROFILES[name])
    for _ in range(8):
        world.tick()


def reset_ego(
    ego: Any,
    world: Any,
    spawn_points: list[Any],
    offset: int,
) -> None:
    """Teleport the ego to a fresh straight-road spawn and zero its motion."""

    index = (244 + int(offset) * 23) % len(spawn_points)
    transform = spawn_points[index]
    waypoint = world.get_map().get_waypoint(
        transform.location, project_to_road=True
    )
    if waypoint is not None:
        candidates = waypoint.next(30.0)
        if candidates:
            delta = abs(
                math.degrees(
                    math.atan2(
                        math.sin(
                            math.radians(
                                candidates[0].transform.rotation.yaw
                                - waypoint.transform.rotation.yaw
                            )
                        ),
                        math.cos(
                            math.radians(
                                candidates[0].transform.rotation.yaw
                                - waypoint.transform.rotation.yaw
                            )
                        ),
                    )
                )
            )
            if delta < 12.0:
                transform = carla.Transform(
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z + 0.6,
                    ),
                    waypoint.transform.rotation,
                )
    ego.set_transform(transform)
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    for _ in range(20):
        world.tick()


def reset_to_straight(
    ego: Any,
    world: Any,
    spawn_points: list[Any],
    offset: int,
) -> bool:
    """Teleport the ego to a long straight segment (>=200 m, heading-stable)."""

    map_api = world.get_map()
    for attempt in range(12):
        index = (244 + (int(offset) + attempt * 37) * 23) % len(spawn_points)
        transform = spawn_points[index]
        waypoint = map_api.get_waypoint(
            transform.location, project_to_road=True
        )
        if waypoint is None:
            continue
        start_yaw = math.radians(waypoint.transform.rotation.yaw)
        cursor = waypoint
        straight = True
        for _ in range(20):
            candidates = cursor.next(10.0)
            if not candidates:
                straight = False
                break
            cursor = candidates[0]
            delta = abs(
                math.degrees(
                    math.atan2(
                        math.sin(
                            math.radians(cursor.transform.rotation.yaw)
                            - start_yaw
                        ),
                        math.cos(
                            math.radians(cursor.transform.rotation.yaw)
                            - start_yaw
                        ),
                    )
                )
            )
            if delta > 5.0:
                straight = False
                break
        if not straight:
            continue
        ego.set_transform(
            carla.Transform(
                carla.Location(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    z=waypoint.transform.location.z + 0.6,
                ),
                waypoint.transform.rotation,
            )
        )
        ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        for _ in range(20):
            world.tick()
        return True
    return False


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(180)
    world = None
    for attempt in range(3):
        try:
            world = client.load_world(args.town)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(10)
    if world is None:
        raise RuntimeError("failed to load Town05_Opt")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    world.tick()

    library = world.get_blueprint_library()
    ego_blueprint = library.find("vehicle.lincoln.mkz_2020")
    spawn_points = world.get_map().get_spawn_points()
    spawn = spawn_points[244 % len(spawn_points)]
    ego = world.spawn_actor(ego_blueprint, spawn)
    if ego is None:
        raise RuntimeError("failed to spawn ego")
    ego.set_autopilot(False)

    rig = SynchronizedMultiviewCameraRig(
        world,
        ego,
        width=224,
        height=224,
        fov=100.0,
        sensor_tick=0.05,
        enable_lidar=True,
        lidar_range_m=80.0,
        lidar_channels=32,
        available_cameras=CAMERA_ORDER,
    )

    from structured_command_parser.src.modernbert_service import (
        ModernBertCommandService,
    )

    service = ModernBertCommandService(str(args.parser_model), device=args.device)
    service.warmup()
    parser = service.parser
    parser.load()

    output_dir = args.output_dir.resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "tensors").mkdir(parents=True, exist_ok=True)
    (output_dir / "intents").mkdir(parents=True, exist_ok=True)

    intent_texts = {
        "keep_40": "Keep the current lane at 40.0 kilometers per hour.",
        "keep_30": "Keep the current lane at 30.0 kilometers per hour.",
        "decel_20": "Slow down smoothly to 20.0 kilometers per hour.",
        "yield_10": "Slow down and yield to the road user ahead.",
        "stop": "Stop the vehicle immediately.",
    }
    intent_paths = {}
    for name, text in intent_texts.items():
        encoded = parser.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=parser.max_length
        )
        encoded = {key: value.to(parser.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        path = output_dir / "intents" / f"{name}.pt"
        torch.save(
            {
                "intent_tokens": tokens[0].detach().float().cpu(),
                "intent_mask": encoded["attention_mask"][0].detach().bool().cpu(),
            },
            path,
        )
        intent_paths[name] = path

    manifest_path = output_dir / "manifest.jsonl"
    samples = 0
    last_frame = -1
    spawn_cleanup: list[Any] = []

    def snapshot(
        episode: str,
        risk: dict[str, Any],
        intent_name: str,
        speed_cap_kmh: float,
        quotas: dict[str, int],
        counts: dict[str, int],
        split_label: str,
    ) -> None:
        nonlocal samples, last_frame
        bucket = risk["risk_level"]
        if (
            samples >= args.max_samples
            or counts[bucket] >= quotas[bucket]
        ):
            return
        try:
            frame, images, mask, lidar_bev, _ = rig.latest_multisensor(
                minimum_frame=last_frame + 1, timeout_s=0.25
            )
        except RuntimeError:
            return
        last_frame = frame
        sample_id = (
            f"scene2_{episode}_{samples:05d}_{frame}"
        )
        torch.save(images[0], output_dir / "images" / f"{sample_id}.pt")
        ego_wp = world.get_map().get_waypoint(
            ego.get_location(), project_to_road=True
        )
        torch.save(
            {
                "camera_bev": torch.zeros(8, 64, 64),
                "lidar_bev": lidar_bev[0].float().cpu(),
                "ego_features": torch.tensor(
                    ego_features(ego, ego_wp, speed_cap_kmh)
                ),
                "candidate_features": torch.zeros(32, 12),
                "candidate_mask": torch.zeros(32, dtype=torch.bool),
                "environment_features": torch.tensor(
                    environment_features(world)
                ),
                "camera_view_mask": torch.tensor([True, True, True, True]),
            },
            output_dir / "tensors" / f"{sample_id}.pt",
        )
        risk_level = risk["risk_level"]
        action_hint = risk.get("action_hint", "keep_lane")
        if risk_level == "low":
            action = "keep_lane"
        elif action_hint in {"emergency_brake", "stop"}:
            action = action_hint
        elif action_hint == "yield":
            action = "decelerate"
        else:
            action = "decelerate"
        if action == "emergency_brake":
            target_speed = 0.0
        elif action == "stop":
            target_speed = 0.0
        elif risk_level != "low":
            target_speed = 15.0
        else:
            target_speed = speed_cap_kmh
        split = split_label
        manifest.write(
            json.dumps(
                {
                    "schema_version": "scene3_multimodal_training_sample/2.0",
                    "sample_id": sample_id,
                    "source_dataset": "CARLA_scene2_finetune_traffic",
                    "source_frame": int(frame),
                    "route_s_m": round(
                        float(ego_wp.s if ego_wp is not None else 0.0), 3
                    ),
                    "split_group": f"scene2_finetune:{episode}",
                    "split": split,
                    "counterfactual_set_id": sample_id,
                    "variant_type": f"scene2_{episode}",
                    "command_id": intent_name,
                    "source_text": intent_texts[intent_name],
                    "normalized_text": intent_texts[intent_name],
                    "camera_order": list(CAMERA_ORDER),
                    "camera_view_mask": [True, True, True, True],
                    "image_paths": [f"images/{sample_id}.pt"] * 4,
                    "image_tensor_path": f"images/{sample_id}.pt",
                    "tensor_path": f"tensors/{sample_id}.pt",
                    "intent_tensor_path": f"intents/{intent_name}.pt",
                    "label": {
                        "action": action,
                        "target_speed_kmh": round(target_speed, 3),
                        "target_lane": None,
                    },
                    "risk_level": risk_level,
                    "risk_reason_codes": [],
                    "risk_physics": {
                        "ttc_s": risk.get("ttc_s"),
                        "gap_m": risk.get("gap_m"),
                        "rel_speed_kmh": risk.get("rel_speed_kmh"),
                    },
                    "risk_score": risk.get("risk_score"),
                    "horizon_probabilities": risk.get(
                        "horizon_probabilities"
                    ),
                    "lane_risk_levels": risk.get("lane_risk_levels"),
                    "lane_change_risk": risk.get("lane_change", {}),
                    "front_distance_m": risk.get("front_distance_m"),
                    "weather_profile": CURRENT_WEATHER,
                    "control_speed_cap_kmh": speed_cap_kmh,
                    "capture_quality": "moving_scene2_traffic",
                    "sampling_weight": (
                        20.0 if risk_level == "high" else 6.0
                        if risk_level == "medium" else 1.0
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        counts[bucket] += 1
        samples += 1

    def cleanup(actors: list[Any]) -> None:
        for actor in actors:
            try:
                if alive(actor):
                    try:
                        actor.set_autopilot(False)
                    except (RuntimeError, AttributeError):
                        pass
                    actor.destroy()
            except RuntimeError:
                pass

    def run_episode(
        name: str,
        *,
        target_speed_kmh: float,
        ticks: int,
        sample_every: int = 3,
        quotas: dict[str, int] | None = None,
        counts: dict[str, int] | None = None,
        weather: str | None = None,
        split_label: str = "train",
    ) -> None:
        nonlocal last_frame, samples
        if weather is not None:
            set_weather(world, weather)
        quota = quotas or {"low": 10**9, "medium": 10**9, "high": 10**9}
        counter = counts if counts is not None else {
            "low": 0,
            "medium": 0,
            "high": 0,
        }
        for tick_index in range(ticks):
            follow_road(ego, world, target_speed_kmh)
            world.tick()
            if tick_index % sample_every == 0 and samples < args.max_samples:
                risk = label_risk(ego, world)
                if counter[risk["risk_level"]] >= quota[risk["risk_level"]]:
                    continue
                hint = risk.get("action_hint", "keep_lane")
                if risk["risk_level"] == "high" and hint in {
                    "emergency_brake",
                    "stop",
                }:
                    intent = "stop"
                elif risk["risk_level"] == "high":
                    intent = "yield_10"
                elif risk["risk_level"] == "medium":
                    intent = "decel_20"
                else:
                    intent = "keep_40"
                snapshot(
                    name,
                    risk,
                    intent,
                    target_speed_kmh,
                    quota,
                    counter,
                    split_label,
                )
        return counter

    try:
        manifest = manifest_path.open("w", encoding="utf-8")

        # Episode 1: baseline cruise on an empty road (clean low-risk
        # hard negatives; no parked actors to pollute the front cone).
        reset_to_straight(ego, world, spawn_points, 0)
        run_episode(
            "baseline",
            target_speed_kmh=40.0,
            ticks=1500,
            sample_every=3,
            quotas={"low": 240, "medium": 40, "high": 20},
            weather="clear_noon",
        )

        # Episode 1b: rainy-night empty-road low-risk hard negatives.
        reset_to_straight(ego, world, spawn_points, 11)
        run_episode(
            "baseline_rainy_night",
            target_speed_kmh=35.0,
            ticks=1200,
            sample_every=3,
            quotas={"low": 160, "medium": 40, "high": 20},
            weather="rainy_night",
        )

        # Episode 2: slow vehicle ahead.
        reset_to_straight(ego, world, spawn_points, 1)
        set_weather(world, "rainy_night")
        lead = spawn_actor(
            world,
            "vehicle.toyota.prius",
            pick_road_ahead(world, ego, 55.0),
        )
        if lead is not None:
            spawn_cleanup.append(lead)
            counts2 = {"low": 0, "medium": 0, "high": 0}
            quota2 = {"low": 120, "medium": 120, "high": 120}
            left_wp = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True
            )
            left_lane = left_wp.get_left_lane() if left_wp is not None else None
            left_vehicle = None
            if left_lane is not None:
                left_points = left_lane.next(30.0)
                if left_points:
                    left_transform = left_points[0].transform
                    left_transform.location.z += 0.6
                    left_vehicle = spawn_actor(
                        world, "vehicle.audi.tt", left_transform
                    )
                    if left_vehicle is not None:
                        spawn_cleanup.append(left_vehicle)
            for tick_index in range(900):
                if alive(lead):
                    try:
                        lead.apply_control(
                            carla.VehicleControl(throttle=0.28, brake=0.0)
                        )
                    except RuntimeError:
                        pass
                if alive(left_vehicle):
                    try:
                        left_vehicle.apply_control(
                            carla.VehicleControl(throttle=0.32, brake=0.0)
                        )
                    except RuntimeError:
                        pass
                forward, _ = forward_lateral(ego, lead)
                if forward >= 1e9:
                    break
                target = 30.0 if forward > 24.0 else 12.0 if forward > 14.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    risk = label_risk(ego, world)
                    if counts2[risk["risk_level"]] >= quota2[risk["risk_level"]]:
                        continue
                    hint = risk.get("action_hint", "keep_lane")
                    intent = (
                        "stop"
                        if risk["risk_level"] == "high"
                        and hint in {"emergency_brake", "stop"}
                        else "yield_10"
                        if risk["risk_level"] == "high"
                        else "decel_20"
                        if risk["risk_level"] == "medium"
                        else "keep_40"
                    )
                    snapshot(
                        "slow_vehicle",
                        risk,
                        intent,
                        32.0,
                        quota2,
                        counts2,
                        "validation",
                    )
            cleanup([lead] + ([left_vehicle] if left_vehicle is not None else []))
            spawn_cleanup.clear()

        # Episode 3: crossing pedestrian ahead.
        reset_to_straight(ego, world, spawn_points, 2)
        set_weather(world, "wet_cloudy")
        walker_bp = library.find("walker.pedestrian.0001")
        walker_ahead = pick_road_ahead(world, ego, 38.0)
        walker_ahead.location.x += 2.5
        walker = world.spawn_actor(walker_bp, walker_ahead)
        if walker is not None:
            spawn_cleanup.append(walker)
            counts3 = {"low": 0, "medium": 0, "high": 0}
            quota3 = {"low": 120, "medium": 120, "high": 120}
            walker_control = carla.WalkerControl()
            walker_control.speed = 1.4
            for tick_index in range(700):
                forward, _ = forward_lateral(ego, walker)
                target = 28.0 if forward > 24.0 else 10.0 if forward > 14.0 else 0.0
                follow_road(ego, world, target)
                if alive(walker):
                    try:
                        walker_control.direction = carla.Vector3D(0.0, -1.0, 0.0)
                        walker.apply_control(walker_control)
                    except RuntimeError:
                        pass
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    risk = label_risk(ego, world)
                    if counts3[risk["risk_level"]] >= quota3[risk["risk_level"]]:
                        continue
                    hint = risk.get("action_hint", "keep_lane")
                    intent = (
                        "stop"
                        if risk["risk_level"] == "high"
                        and hint in {"emergency_brake", "stop"}
                        else "yield_10"
                        if risk["risk_level"] == "high"
                        else "decel_20"
                        if risk["risk_level"] == "medium"
                        else "keep_40"
                    )
                    snapshot(
                        "pedestrian",
                        risk,
                        intent,
                        30.0,
                        quota3,
                        counts3,
                        "train",
                    )
            cleanup([walker])
            spawn_cleanup.clear()

        # Episode 4: slow cyclist in the right lane.
        reset_to_straight(ego, world, spawn_points, 3)
        set_weather(world, "sunset")
        cyclist = spawn_actor(
            world,
            "vehicle.diamondback.century",
            pick_road_ahead(world, ego, 50.0),
        )
        if cyclist is not None:
            spawn_cleanup.append(cyclist)
            counts4 = {"low": 0, "medium": 0, "high": 0}
            quota4 = {"low": 120, "medium": 120, "high": 120}
            for tick_index in range(800):
                if alive(cyclist):
                    try:
                        cyclist.apply_control(
                            carla.VehicleControl(throttle=0.18, brake=0.0)
                        )
                    except RuntimeError:
                        pass
                forward, _ = forward_lateral(ego, cyclist)
                target = 28.0 if forward > 24.0 else 10.0 if forward > 14.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    risk = label_risk(ego, world)
                    if counts4[risk["risk_level"]] >= quota4[risk["risk_level"]]:
                        continue
                    hint = risk.get("action_hint", "keep_lane")
                    intent = (
                        "stop"
                        if risk["risk_level"] == "high"
                        and hint in {"emergency_brake", "stop"}
                        else "yield_10"
                        if risk["risk_level"] == "high"
                        else "decel_20"
                        if risk["risk_level"] == "medium"
                        else "keep_40"
                    )
                    snapshot(
                        "cyclist",
                        risk,
                        intent,
                        30.0,
                        quota4,
                        counts4,
                        "validation",
                    )
            cleanup([cyclist])
            spawn_cleanup.clear()

        # Episode 5: stopped bus with waiting pedestrians (right lane).
        reset_to_straight(ego, world, spawn_points, 4)
        set_weather(world, "foggy_morning")
        bus = spawn_actor(
            world,
            "vehicle.volkswagen.t2",
            pick_road_ahead(world, ego, 45.0),
        )
        if bus is not None:
            spawn_cleanup.append(bus)
            counts5 = {"low": 0, "medium": 0, "high": 0}
            quota5 = {"low": 120, "medium": 120, "high": 120}
            walker_bp2 = library.find("walker.pedestrian.0002")
            walker1 = world.spawn_actor(
                walker_bp2,
                carla.Transform(
                    carla.Location(
                        x=bus.get_location().x + 3.0,
                        y=bus.get_location().y + 2.5,
                        z=0.8,
                    ),
                    carla.Rotation(yaw=90.0),
                ),
            )
            for tick_index in range(700):
                if alive(bus):
                    try:
                        bus.apply_control(
                            carla.VehicleControl(throttle=0.0, brake=1.0)
                        )
                    except RuntimeError:
                        pass
                forward, _ = forward_lateral(ego, bus)
                target = 28.0 if forward > 24.0 else 10.0 if forward > 14.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    risk = label_risk(ego, world)
                    if counts5[risk["risk_level"]] >= quota5[risk["risk_level"]]:
                        continue
                    hint = risk.get("action_hint", "keep_lane")
                    intent = (
                        "stop"
                        if risk["risk_level"] == "high"
                        and hint in {"emergency_brake", "stop"}
                        else "yield_10"
                        if risk["risk_level"] == "high"
                        else "decel_20"
                        if risk["risk_level"] == "medium"
                        else "keep_40"
                    )
                    snapshot(
                        "bus_stop",
                        risk,
                        intent,
                        30.0,
                        quota5,
                        counts5,
                        "train",
                    )
            cleanup([bus] + ([walker1] if walker1 is not None else []))
            spawn_cleanup.clear()

        manifest.close()
    finally:
        cleanup(spawn_cleanup)
        rig.close()
        try:
            ego.destroy()
        except RuntimeError:
            pass
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    print("samples:", samples)
    print("manifest:", manifest_path)
    return 0 if samples >= 500 else 2


if __name__ == "__main__":
    raise SystemExit(main())
