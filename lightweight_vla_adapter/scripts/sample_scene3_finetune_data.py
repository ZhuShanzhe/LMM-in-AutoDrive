"""Sample rainy-night Scene-3 cruise data for fine-tuning the V6 policy.

The sampler drives the ego along the official Town05_Opt route and records
four-view RGB, vehicle/environment state and text tokens with labels that are
derived from the scene (no actor within range -> low risk, keep lane).  It
densely samples the corridor segments where the V6 risk head is known to
false-positive on static rainy-night views.
"""

from __future__ import annotations

import argparse
import json
import math
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
from carla_multiview_sensor import CAMERA_TRANSFORMS  # noqa: E402
import run_emergency_response_6km as runner  # noqa: E402
from run_emergency_response_6km import (  # noqa: E402
    apply_emergency_weather,
    build_town05_route_context,
    load_runtime_config,
)

setup_carla_api()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=(
            EXPERIMENT_CARLA
            / "configs"
            / "scene_3_emergency_6km_runtime.json"
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=(
            Path("/root/autodl-tmp/models/modernbert-drive-command-compositional")
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/root/autodl-tmp/datasets/training")
            / "universal_three_scene_v6_finetune"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
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


def ego_features(
    ego: Any,
    waypoint: Any,
    speed_cap_kmh: float,
) -> list[float]:
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


def main() -> None:
    args = parse_args()
    random = np.random.RandomState(args.seed)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.load_world("Town05_Opt")
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    config = load_runtime_config(args.runtime_config, event_variant="cautious_sparse", seed=args.seed)
    context = build_town05_route_context(world.get_map(), config["map"]["route"])
    runner.carla = carla
    apply_emergency_weather(
        world, config["weather"], profile="official-rainy-night"
    )
    world.tick()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2020")
    start_wp = context.adapter.logical_waypoint(-2, 0.0) or context.route[0][0]
    ego = world.spawn_actor(
        vehicle_bp,
        carla.Transform(
            carla.Location(
                x=start_wp.transform.location.x,
                y=start_wp.transform.location.y,
                z=start_wp.transform.location.z + 1.0,
            ),
            start_wp.transform.rotation,
        ),
    )
    if ego is None:
        raise RuntimeError("failed to spawn ego")

    cameras = []
    results = {}
    for name, (x, y, z, pitch, yaw) in CAMERA_TRANSFORMS.items():
        bp = blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "224")
        bp.set_attribute("image_size_y", "224")
        bp.set_attribute("fov", "100")
        bp.set_attribute("sensor_tick", "0.05")
        camera = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
            ),
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        cameras.append(camera)

        def make_callback(view_name: str):
            def receive(image: Any) -> None:
                bgra = np.frombuffer(
                    image.raw_data, dtype=np.uint8
                ).reshape(image.height, image.width, 4)
                results[view_name] = np.ascontiguousarray(bgra[:, :, 2::-1])
                results["frame"] = int(image.frame)

            return receive

        camera.listen(make_callback(name))

    parser_path = args.parser_model
    from structured_command_parser.src.modernbert_service import (
        ModernBertCommandService,
    )

    service = ModernBertCommandService(str(parser_path), device=args.device)
    service.warmup()
    parser = service.parser
    parser.load()
    text = "Keep the current lane at 32.0 kilometers per hour."
    encoded = parser.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=parser.max_length
    )
    encoded = {
        key: value.to(parser.device) for key, value in encoded.items()
    }
    with torch.inference_mode():
        intent_tokens = parser.model.backbone(**encoded).last_hidden_state

    output_dir = args.output_dir.resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "tensors").mkdir(parents=True, exist_ok=True)
    (output_dir / "intents").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "intent_tokens": intent_tokens[0].detach().float().cpu(),
            "intent_mask": encoded["attention_mask"][0].detach().bool().cpu(),
        },
        output_dir / "intents" / "keep_lane_32.pt",
    )

    # The route is anchored at (250, 240); the known rainy-night false
    # positive corridor now starts near 60 m.  Cover it densely on both
    # lanes and sample the remaining route sparsely.
    dense = [float(value) for value in range(60, 501, 2)]
    sparse = [
        0.0, 12.5, 22.5, 40.6, 55.4, 67.9, 96.3, 127.5, 156.0, 196.0,
        226.0, 272.0, 302.0, 450.0, 500.0, 600.0, 700.0, 800.0, 900.0,
        1000.0, 1100.0, 1200.0, 1300.0, 1400.0,
    ]
    sparse += [float(value) for value in range(500, 6001, 50)]
    positions = sorted(set(dense) | set(sparse))
    lanes = (-2, -1)
    samples = 0
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for progress_m in positions:
            for lane in lanes:
                waypoint = context.adapter.logical_waypoint(lane, progress_m)
                if waypoint is None:
                    continue
                transform = carla.Transform(
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z + 0.5,
                    ),
                    waypoint.transform.rotation,
                )
                ego.set_transform(transform)
                ego.set_target_velocity(carla.Vector3D())
                ego.apply_control(carla.VehicleControl(brake=1.0))
                for _ in range(10):
                    world.tick()
                # One static sample.
                for _ in range(4):
                    world.tick()
                if "frame" not in results:
                    continue
                sample_id = f"finetune_{int(progress_m * 10)}_{lane}_{results['frame']}"
                images = torch.from_numpy(
                    np.stack(
                        [
                            results[name]
                            for name in ("front", "left", "right", "rear")
                        ]
                    )
                ).permute(0, 3, 1, 2)
                ego_wp = world.get_map().get_waypoint(
                    ego.get_location(), project_to_road=True
                )
                torch.save(
                    images,
                    output_dir / "images" / f"{sample_id}.pt",
                )
                torch.save(
                    {
                        "camera_bev": torch.zeros(8, 64, 64),
                        "lidar_bev": torch.zeros(4, 64, 64),
                        "ego_features": torch.tensor(
                            ego_features(ego, ego_wp, 32.0)
                        ),
                        "candidate_features": torch.zeros(32, 12),
                        "candidate_mask": torch.zeros(32, dtype=torch.bool),
                        "environment_features": torch.tensor(
                            environment_features(world)
                        ),
                        "camera_view_mask": torch.tensor(
                            [True, True, True, True]
                        ),
                    },
                    output_dir / "tensors" / f"{sample_id}.pt",
                )
                split = "validation" if samples % 10 == 0 else "train"
                manifest.write(
                    json.dumps(
                        {
                            "schema_version": "scene3_multimodal_training_sample/2.0",
                            "sample_id": sample_id,
                            "source_dataset": "CARLA_scene3_finetune_rainy_night",
                            "source_frame": int(results["frame"]),
                            "route_s_m": round(float(progress_m), 3),
                            "split_group": "scene3_finetune:route_000",
                            "split": split,
                            "counterfactual_set_id": sample_id,
                            "variant_type": "scene3_cruise_false_positive_hard_negative",
                            "command_id": "keep_lane_32",
                            "source_text": text,
                            "normalized_text": text,
                            "camera_order": ["front", "left", "right", "rear"],
                            "camera_view_mask": [True, True, True, True],
                            "image_paths": [
                                f"images/{sample_id}.pt"
                            ] * 4,
                            "image_tensor_path": f"images/{sample_id}.pt",
                            "tensor_path": f"tensors/{sample_id}.pt",
                            "intent_tensor_path": "intents/keep_lane_32.pt",
                            "label": {
                                "action": "keep_lane",
                                "target_speed_kmh": 32.0,
                                "target_lane": None,
                            },
                            "risk_level": "low",
                            "risk_reason_codes": [],
                            "weather_profile": "official-rainy-night",
                            "control_speed_cap_kmh": 32.0,
                            "capture_quality": "static_or_moving_scene3_cruise",
                            "sampling_weight": 40.0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                samples += 1
        # Moving samples through the false-positive corridor on both lanes.
        for progress_m in range(60, 501, 4):
            for lane in lanes:
                waypoint = context.adapter.logical_waypoint(
                    lane, float(progress_m)
                )
                if waypoint is None:
                    continue
                transform = carla.Transform(
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z + 0.5,
                    ),
                    waypoint.transform.rotation,
                )
                ego.set_transform(transform)
                forward = transform.get_forward_vector()
                ego.set_target_velocity(
                    carla.Vector3D(
                        x=forward.x * 6.0,
                        y=forward.y * 6.0,
                        z=0.0,
                    )
                )
                ego.apply_control(carla.VehicleControl(throttle=0.3))
                for _ in range(6):
                    world.tick()
                if "frame" not in results:
                    continue
                sample_id = (
                    f"finetune_moving_{progress_m}_{lane}_{results['frame']}"
                )
                images = torch.from_numpy(
                    np.stack(
                        [
                            results[name]
                            for name in ("front", "left", "right", "rear")
                        ]
                    )
                ).permute(0, 3, 1, 2)
                ego_wp = world.get_map().get_waypoint(
                    ego.get_location(), project_to_road=True
                )
                torch.save(
                    images,
                    output_dir / "images" / f"{sample_id}.pt",
                )
                torch.save(
                    {
                        "camera_bev": torch.zeros(8, 64, 64),
                        "lidar_bev": torch.zeros(4, 64, 64),
                        "ego_features": torch.tensor(
                            ego_features(ego, ego_wp, 32.0)
                        ),
                        "candidate_features": torch.zeros(32, 12),
                        "candidate_mask": torch.zeros(32, dtype=torch.bool),
                        "environment_features": torch.tensor(
                            environment_features(world)
                        ),
                        "camera_view_mask": torch.tensor(
                            [True, True, True, True]
                        ),
                    },
                    output_dir / "tensors" / f"{sample_id}.pt",
                )
                split = "validation" if samples % 10 == 0 else "train"
                manifest.write(
                    json.dumps(
                        {
                            "schema_version": "scene3_multimodal_training_sample/2.0",
                            "sample_id": sample_id,
                            "source_dataset": "CARLA_scene3_finetune_rainy_night",
                            "source_frame": int(results["frame"]),
                            "route_s_m": round(float(progress_m), 3),
                            "split_group": "scene3_finetune:route_000",
                            "split": split,
                            "counterfactual_set_id": sample_id,
                            "variant_type": "scene3_cruise_false_positive_hard_negative",
                            "command_id": "keep_lane_32",
                            "source_text": text,
                            "normalized_text": text,
                            "camera_order": ["front", "left", "right", "rear"],
                            "camera_view_mask": [True, True, True, True],
                            "image_paths": [
                                f"images/{sample_id}.pt"
                            ] * 4,
                            "image_tensor_path": f"images/{sample_id}.pt",
                            "tensor_path": f"tensors/{sample_id}.pt",
                            "intent_tensor_path": "intents/keep_lane_32.pt",
                            "label": {
                                "action": "keep_lane",
                                "target_speed_kmh": 32.0,
                                "target_lane": None,
                            },
                            "risk_level": "low",
                            "risk_reason_codes": [],
                            "weather_profile": "official-rainy-night",
                            "control_speed_cap_kmh": 32.0,
                            "capture_quality": "moving_scene3_cruise",
                            "sampling_weight": 40.0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                samples += 1

    for camera in cameras:
        camera.destroy()
    ego.destroy()
    try:
        world.apply_settings(original)
    except RuntimeError:
        pass
    print(f"sampled {samples} rows -> {manifest_path}")
    (output_dir / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "scene3_finetune_sampler/1.0",
                "samples": samples,
                "positions": positions,
                "lanes": list(lanes),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
