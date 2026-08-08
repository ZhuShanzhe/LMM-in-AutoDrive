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
        throttle, brake = 0.0, 0.55
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
    """Offline-only risk labels from actor truth."""

    front_best = None
    left_best = None
    right_best = None
    for actor in world.get_actors():
        if actor.id == ego.id:
            continue
        if not alive(actor):
            continue
        try:
            type_id = actor.type_id
        except (RuntimeError, AttributeError):
            continue
        if not ("vehicle" in type_id or "walker" in type_id):
            continue
        forward, lateral = forward_lateral(ego, actor)
        if forward >= 1e9:
            continue
        if forward <= 0.5 or forward > 60.0:
            continue
        distance = math.hypot(forward, lateral)
        if abs(lateral) < 2.6 and (front_best is None or distance < front_best):
            front_best = distance
        if -5.5 <= lateral <= -2.2 and (
            left_best is None or distance < left_best
        ):
            left_best = distance
        if 2.2 <= lateral <= 5.5 and (
            right_best is None or distance < right_best
        ):
            right_best = distance

    def level(distance: float | None) -> str:
        if distance is None:
            return "low"
        if distance < 12.0:
            return "high"
        if distance < 25.0:
            return "medium"
        return "low"

    return {
        "risk_level": level(front_best),
        "front_distance_m": front_best,
        "lane_change": {
            "left": {"risk_level": level(left_best), "is_safe": level(left_best) == "low"},
            "right": {"risk_level": level(right_best), "is_safe": level(right_best) == "low"},
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
    for _ in range(20):
        candidates = waypoint.next(distance_m / 4.0)
        if not candidates:
            break
        waypoint = candidates[0]
    transform = waypoint.transform
    transform.location.z += 0.6
    return transform


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.load_world("Town05_Opt")
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
    ) -> None:
        nonlocal samples, last_frame
        if samples >= args.max_samples:
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
        action = "decelerate" if risk_level != "low" else "keep_lane"
        target_speed = 20.0 if risk_level != "low" else speed_cap_kmh
        split = "validation" if samples % 10 == 0 else "train"
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
                    "lane_change_risk": risk.get("lane_change", {}),
                    "front_distance_m": risk.get("front_distance_m"),
                    "weather_profile": "ClearNoon",
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
        samples += 1

    def cleanup(actors: list[Any]) -> None:
        for actor in actors:
            try:
                if alive(actor):
                    actor.destroy()
            except RuntimeError:
                pass

    def run_episode(
        name: str,
        *,
        target_speed_kmh: float,
        ticks: int,
        sample_every: int = 3,
    ) -> None:
        nonlocal last_frame, samples
        for tick_index in range(ticks):
            follow_road(ego, world, target_speed_kmh)
            world.tick()
            if tick_index % sample_every == 0 and samples < args.max_samples:
                risk = label_risk(ego, world)
                intent = (
                    "yield_10"
                    if risk["risk_level"] == "high"
                    else "decel_20"
                    if risk["risk_level"] == "medium"
                    else "keep_40"
                )
                snapshot(name, risk, intent, target_speed_kmh)

    try:
        manifest = manifest_path.open("w", encoding="utf-8")

        # Episode 1: baseline cruise with sparse background traffic.
        traffic = []
        for index in range(8):
            point = spawn_points[(244 + 30 + index * 7) % len(spawn_points)]
            actor = spawn_actor(
                world,
                rng.choice(
                    [
                        "vehicle.audi.tt",
                        "vehicle.toyota.prius",
                        "vehicle.volkswagen.t2",
                        "vehicle.nissan.patrol",
                    ]
                ),
                point,
            )
            if actor is not None:
                actor.set_autopilot(True, 8000)
                traffic.append(actor)
        for _ in range(60):
            world.tick()
        run_episode("baseline", target_speed_kmh=45.0, ticks=800, sample_every=4)
        cleanup(traffic)

        # Episode 2: slow vehicle ahead.
        lead = spawn_actor(
            world,
            "vehicle.toyota.prius",
            pick_road_ahead(world, ego, 55.0),
        )
        if lead is not None:
            spawn_cleanup.append(lead)
            for tick_index in range(900):
                if alive(lead):
                    try:
                        lead.apply_control(
                            carla.VehicleControl(throttle=0.28, brake=0.0)
                        )
                    except RuntimeError:
                        pass
                forward, _ = forward_lateral(ego, lead)
                if forward >= 1e9:
                    break
                target = 32.0 if forward > 16.0 else 12.0 if forward > 9.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if (
                    tick_index % 3 == 0
                    and samples < args.max_samples
                ):
                    snapshot("slow_vehicle", label_risk(ego, world), "yield_10", 32.0)
            cleanup([lead])
            spawn_cleanup.clear()

        # Episode 3: crossing pedestrian ahead.
        walker_bp = library.find("walker.pedestrian.0001")
        walker_ahead = pick_road_ahead(world, ego, 38.0)
        walker_ahead.location.x += 2.5
        walker = world.spawn_actor(walker_bp, walker_ahead)
        if walker is not None:
            spawn_cleanup.append(walker)
            walker_control = carla.WalkerControl()
            walker_control.speed = 1.4
            for tick_index in range(700):
                forward, _ = forward_lateral(ego, walker)
                target = 30.0 if forward > 15.0 else 8.0 if forward > 8.0 else 0.0
                follow_road(ego, world, target)
                if alive(walker):
                    try:
                        walker_control.direction = carla.Vector3D(0.0, -1.0, 0.0)
                        walker.apply_control(walker_control)
                    except RuntimeError:
                        pass
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    snapshot("pedestrian", label_risk(ego, world), "yield_10", 30.0)
            cleanup([walker])
            spawn_cleanup.clear()

        # Episode 4: slow cyclist in the right lane.
        cyclist = spawn_actor(
            world,
            "vehicle.diamondback.century",
            pick_road_ahead(world, ego, 50.0),
        )
        if cyclist is not None:
            spawn_cleanup.append(cyclist)
            for tick_index in range(800):
                if alive(cyclist):
                    try:
                        cyclist.apply_control(
                            carla.VehicleControl(throttle=0.18, brake=0.0)
                        )
                    except RuntimeError:
                        pass
                forward, _ = forward_lateral(ego, cyclist)
                target = 30.0 if forward > 15.0 else 10.0 if forward > 8.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    snapshot("cyclist", label_risk(ego, world), "yield_10", 30.0)
            cleanup([cyclist])
            spawn_cleanup.clear()

        # Episode 5: stopped bus with waiting pedestrians (right lane).
        bus = spawn_actor(
            world,
            "vehicle.volkswagen.t2",
            pick_road_ahead(world, ego, 45.0),
        )
        if bus is not None:
            spawn_cleanup.append(bus)
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
                target = 30.0 if forward > 15.0 else 8.0 if forward > 8.0 else 0.0
                follow_road(ego, world, target)
                world.tick()
                if tick_index % 3 == 0 and samples < args.max_samples:
                    snapshot("bus_stop", label_risk(ego, world), "yield_10", 30.0)
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
