"""Sample the Scene-3 blocked-lane hazard for risk-head fine-tuning."""

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
        "--output-dir",
        type=Path,
        default=(
            Path("/root/autodl-tmp/datasets/training")
            / "universal_three_scene_v6_finetune_blocked_lane"
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=Path("/root/autodl-tmp/models/modernbert-drive-command-compositional"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.load_world("Town05_Opt")
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    config = load_runtime_config(
        EXPERIMENT_CARLA
        / "configs"
        / "scene_3_emergency_6km_runtime.json",
        event_variant="cautious_sparse",
        seed=20260807,
    )
    context = build_town05_route_context(world.get_map(), config["map"]["route"])
    runner.carla = carla
    apply_emergency_weather(world, config["weather"], profile="official-rainy-night")
    world.tick()

    library = world.get_blueprint_library()
    ego_bp = library.find("vehicle.lincoln.mkz_2020")
    maintenance_bp = library.find("vehicle.carlamotors.carlacola")
    gap_bp = library.find("vehicle.audi.tt")

    def spawn_at(blueprint, lane, s_m, z=0.5, yaw_offset=0.0):
        wp = context.adapter.logical_waypoint(lane, float(s_m))
        if wp is None:
            return None
        transform = carla.Transform(
            carla.Location(
                x=wp.transform.location.x,
                y=wp.transform.location.y,
                z=wp.transform.location.z + z,
            ),
            carla.Rotation(
                yaw=wp.transform.rotation.yaw + yaw_offset,
                pitch=wp.transform.rotation.pitch,
            ),
        )
        actor = world.try_spawn_actor(blueprint, transform)
        return actor

    ego_wp = context.adapter.logical_waypoint(-2, 4780.0)
    ego = world.spawn_actor(
        ego_bp,
        carla.Transform(
            carla.Location(
                x=ego_wp.transform.location.x,
                y=ego_wp.transform.location.y,
                z=ego_wp.transform.location.z + 1.0,
            ),
            ego_wp.transform.rotation,
        ),
    )
    maintenance = spawn_at(maintenance_bp, -2, 4850.0)

    cameras = []
    results = {}
    for name, (x, y, z, pitch, yaw) in CAMERA_TRANSFORMS.items():
        bp = library.find("sensor.camera.rgb")
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

    from structured_command_parser.src.modernbert_service import (
        ModernBertCommandService,
    )

    service = ModernBertCommandService(str(args.parser_model), device=args.device)
    service.warmup()
    parser = service.parser
    parser.load()
    stop_text = "Brake immediately."
    cruise_text = "Keep the current lane at 32.0 kilometers per hour."
    intents = {}
    for label, text in (("stop", stop_text), ("cruise", cruise_text)):
        encoded = parser.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=parser.max_length
        )
        encoded = {key: value.to(parser.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        intents[label] = (
            tokens[0].detach().float().cpu(),
            encoded["attention_mask"][0].detach().bool().cpu(),
        )

    output_dir = args.output_dir.resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "tensors").mkdir(parents=True, exist_ok=True)
    (output_dir / "intents").mkdir(parents=True, exist_ok=True)
    for label, (tokens, mask) in intents.items():
        torch.save(
            {"intent_tokens": tokens, "intent_mask": mask},
            output_dir / "intents" / f"{label}.pt",
        )

    def environment_features(world_obj):
        weather = world_obj.get_weather()
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

    env_features = environment_features(world)
    samples = 0
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        # Configuration A: maintenance vehicle in front, left lane occupied.
        gap_front = spawn_at(gap_bp, -1, 4865.0)
        for distance_behind in (25.0, 40.0, 60.0):
            s_m = 4850.0 - distance_behind
            wp = context.adapter.logical_waypoint(-2, s_m)
            ego.set_transform(
                carla.Transform(
                    carla.Location(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z + 0.5,
                    ),
                    wp.transform.rotation,
                )
            )
            ego.set_target_velocity(carla.Vector3D())
            ego.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(10):
                world.tick()
            if "frame" not in results:
                continue
            velocity = ego.get_velocity()
            acceleration = ego.get_acceleration()
            angular = ego.get_angular_velocity()
            control = ego.get_control()
            ego_wp_now = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True
            )
            images = torch.from_numpy(
                np.stack(
                    [results[name] for name in ("front", "left", "right", "rear")]
                )
            ).permute(0, 3, 1, 2)
            sample_id = f"blocked_unsafe_{int(distance_behind)}_{results['frame']}"
            torch.save(images, output_dir / "images" / f"{sample_id}.pt")
            torch.save(
                {
                    "camera_bev": torch.zeros(8, 64, 64),
                    "lidar_bev": torch.zeros(4, 64, 64),
                    "ego_features": torch.tensor(
                        [
                            math.sqrt(
                                float(velocity.x) ** 2
                                + float(velocity.y) ** 2
                            ),
                            math.sqrt(
                                float(acceleration.x) ** 2
                                + float(acceleration.y) ** 2
                            ),
                            math.radians(float(angular.z)),
                            float(control.steer),
                            float(control.throttle),
                            float(control.brake),
                            32.0 / 3.6,
                            float(
                                bool(
                                    ego_wp_now is not None
                                    and ego_wp_now.is_junction
                                )
                            ),
                        ]
                    ),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "environment_features": torch.tensor(env_features),
                    "camera_view_mask": torch.tensor(
                        [True, True, True, True]
                    ),
                },
                output_dir / "tensors" / f"{sample_id}.pt",
            )
            manifest.write(
                json.dumps(
                    {
                        "schema_version": "scene3_multimodal_training_sample/2.0",
                        "sample_id": sample_id,
                        "source_dataset": "CARLA_scene3_blocked_lane_finetune",
                        "source_frame": int(results["frame"]),
                        "route_s_m": round(s_m, 3),
                        "split_group": "blocked_lane:route_000",
                        "split": "train" if samples % 10 else "validation",
                        "counterfactual_set_id": sample_id,
                        "variant_type": "blocked_lane_stopped_vehicle",
                        "command_id": "stop",
                        "source_text": stop_text,
                        "normalized_text": stop_text,
                        "camera_order": ["front", "left", "right", "rear"],
                        "camera_view_mask": [True, True, True, True],
                        "image_paths": [f"images/{sample_id}.pt"] * 4,
                        "image_tensor_path": f"images/{sample_id}.pt",
                        "tensor_path": f"tensors/{sample_id}.pt",
                        "intent_tensor_path": "intents/stop.pt",
                        "label": {
                            "action": "stop",
                            "target_speed_kmh": 0.0,
                            "target_lane": None,
                        },
                        "risk_level": "high",
                        "risk_reason_codes": ["blocked_lane_maintenance_vehicle"],
                        "weather_profile": "official-rainy-night",
                        "control_speed_cap_kmh": 32.0,
                        "capture_quality": "blocked_lane_front_hazard",
                        "sampling_weight": 60.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            samples += 1
        if gap_front is not None:
            gap_front.destroy()

        # Configuration B: maintenance vehicle in front, left lane clear.
        for distance_behind in (25.0, 40.0, 60.0):
            s_m = 4850.0 - distance_behind
            wp = context.adapter.logical_waypoint(-2, s_m)
            ego.set_transform(
                carla.Transform(
                    carla.Location(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z + 0.5,
                    ),
                    wp.transform.rotation,
                )
            )
            ego.set_target_velocity(carla.Vector3D())
            ego.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(10):
                world.tick()
            if "frame" not in results:
                continue
            velocity = ego.get_velocity()
            acceleration = ego.get_acceleration()
            angular = ego.get_angular_velocity()
            control = ego.get_control()
            ego_wp_now = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True
            )
            images = torch.from_numpy(
                np.stack(
                    [results[name] for name in ("front", "left", "right", "rear")]
                )
            ).permute(0, 3, 1, 2)
            sample_id = f"blocked_safe_gap_{int(distance_behind)}_{results['frame']}"
            torch.save(images, output_dir / "images" / f"{sample_id}.pt")
            torch.save(
                {
                    "camera_bev": torch.zeros(8, 64, 64),
                    "lidar_bev": torch.zeros(4, 64, 64),
                    "ego_features": torch.tensor(
                        [
                            math.sqrt(
                                float(velocity.x) ** 2
                                + float(velocity.y) ** 2
                            ),
                            math.sqrt(
                                float(acceleration.x) ** 2
                                + float(acceleration.y) ** 2
                            ),
                            math.radians(float(angular.z)),
                            float(control.steer),
                            float(control.throttle),
                            float(control.brake),
                            32.0 / 3.6,
                            float(
                                bool(
                                    ego_wp_now is not None
                                    and ego_wp_now.is_junction
                                )
                            ),
                        ]
                    ),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "environment_features": torch.tensor(env_features),
                    "camera_view_mask": torch.tensor(
                        [True, True, True, True]
                    ),
                },
                output_dir / "tensors" / f"{sample_id}.pt",
            )
            manifest.write(
                json.dumps(
                    {
                        "schema_version": "scene3_multimodal_training_sample/2.0",
                        "sample_id": sample_id,
                        "source_dataset": "CARLA_scene3_blocked_lane_finetune",
                        "source_frame": int(results["frame"]),
                        "route_s_m": round(s_m, 3),
                        "split_group": "blocked_lane:route_000",
                        "split": "train" if samples % 10 else "validation",
                        "counterfactual_set_id": sample_id,
                        "variant_type": "blocked_lane_clear_left_lane",
                        "command_id": "stop",
                        "source_text": stop_text,
                        "normalized_text": stop_text,
                        "camera_order": ["front", "left", "right", "rear"],
                        "camera_view_mask": [True, True, True, True],
                        "image_paths": [f"images/{sample_id}.pt"] * 4,
                        "image_tensor_path": f"images/{sample_id}.pt",
                        "tensor_path": f"tensors/{sample_id}.pt",
                        "intent_tensor_path": "intents/stop.pt",
                        "label": {
                            "action": "stop",
                            "target_speed_kmh": 0.0,
                            "target_lane": None,
                        },
                        "risk_level": "high",
                        "risk_reason_codes": ["blocked_lane_maintenance_vehicle"],
                        "weather_profile": "official-rainy-night",
                        "control_speed_cap_kmh": 32.0,
                        "capture_quality": "blocked_lane_front_hazard_clear_left",
                        "sampling_weight": 60.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            samples += 1

        # Left-view-only samples: clear left lane (target-lane low risk).
        wp = context.adapter.logical_waypoint(-2, 4810.0)
        ego.set_transform(
            carla.Transform(
                carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z + 0.5,
                ),
                wp.transform.rotation,
            )
        )
        ego.set_target_velocity(carla.Vector3D())
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(10):
            world.tick()
        velocity = ego.get_velocity()
        acceleration = ego.get_acceleration()
        angular = ego.get_angular_velocity()
        control = ego.get_control()
        ego_wp_now = world.get_map().get_waypoint(
            ego.get_location(), project_to_road=True
        )
        images = torch.from_numpy(
            np.stack(
                [results[name] for name in ("front", "left", "right", "rear")]
            )
        ).permute(0, 3, 1, 2)
        sample_id = f"blocked_left_clear_{results['frame']}"
        torch.save(images, output_dir / "images" / f"{sample_id}.pt")
        torch.save(
            {
                "camera_bev": torch.zeros(8, 64, 64),
                "lidar_bev": torch.zeros(4, 64, 64),
                "ego_features": torch.tensor(
                    [
                        math.sqrt(
                            float(velocity.x) ** 2 + float(velocity.y) ** 2
                        ),
                        math.sqrt(
                            float(acceleration.x) ** 2
                            + float(acceleration.y) ** 2
                        ),
                        math.radians(float(angular.z)),
                        float(control.steer),
                        float(control.throttle),
                        float(control.brake),
                        32.0 / 3.6,
                        float(
                            bool(
                                ego_wp_now is not None
                                and ego_wp_now.is_junction
                            )
                        ),
                    ]
                ),
                "candidate_features": torch.zeros(32, 12),
                "candidate_mask": torch.zeros(32, dtype=torch.bool),
                "environment_features": torch.tensor(env_features),
                "camera_view_mask": torch.tensor([False, True, False, False]),
            },
            output_dir / "tensors" / f"{sample_id}.pt",
        )
        manifest.write(
            json.dumps(
                {
                    "schema_version": "scene3_multimodal_training_sample/2.0",
                    "sample_id": sample_id,
                    "source_dataset": "CARLA_scene3_blocked_lane_finetune",
                    "source_frame": int(results["frame"]),
                    "route_s_m": 4810.0,
                    "split_group": "blocked_lane:route_000",
                    "split": "train" if samples % 10 else "validation",
                    "counterfactual_set_id": sample_id,
                    "variant_type": "target_lane_clear_left_view",
                    "command_id": "cruise",
                    "source_text": cruise_text,
                    "normalized_text": cruise_text,
                    "camera_order": ["front", "left", "right", "rear"],
                    "camera_view_mask": [False, True, False, False],
                    "image_paths": [f"images/{sample_id}.pt"] * 4,
                    "image_tensor_path": f"images/{sample_id}.pt",
                    "tensor_path": f"tensors/{sample_id}.pt",
                    "intent_tensor_path": "intents/cruise.pt",
                    "label": {
                        "action": "keep_lane",
                        "target_speed_kmh": 32.0,
                        "target_lane": None,
                    },
                    "risk_level": "low",
                    "risk_reason_codes": [],
                    "weather_profile": "official-rainy-night",
                    "control_speed_cap_kmh": 32.0,
                    "capture_quality": "target_lane_clear_left_view",
                    "sampling_weight": 60.0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        samples += 1

    for camera in cameras:
        camera.destroy()
    for actor in (maintenance, ego):
        if actor is not None:
            actor.destroy()
    try:
        world.apply_settings(original)
    except RuntimeError:
        pass
    print(f"sampled {samples} rows -> {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
