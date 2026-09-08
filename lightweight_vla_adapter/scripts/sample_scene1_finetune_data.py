"""Sample Town04 Scene-1 front-camera cruise data for fine-tuning."""

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
from scenarios.basic.urban_voice_5km import UrbanVoice5KmScenario  # noqa: E402

setup_carla_api()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=(
            EXPERIMENT_CARLA
            / "configs"
            / "basic_voice_urban_5km.json"
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=Path,
        default=Path("/root/autodl-tmp/models/modernbert-drive-command-compositional"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/root/autodl-tmp/datasets/training")
            / "universal_three_scene_v6_finetune_scene1"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.load_world("Town04_Opt")
    settings = world.get_settings()
    original = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    scenario = UrbanVoice5KmScenario(
        world,
        external_control=True,
        config_path=args.scenario_config,
    )
    scenario.client = client
    scenario.fixed_delta_s = 0.05
    scenario.setup()
    world.tick()
    ego = scenario.get_ego_vehicle()
    route_manager = scenario.route_manager

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "224")
    bp.set_attribute("image_size_y", "224")
    bp.set_attribute("fov", "100")
    bp.set_attribute("sensor_tick", "0.05")
    camera = world.spawn_actor(
        bp,
        carla.Transform(
            carla.Location(x=1.45, y=0.0, z=1.55),
            carla.Rotation(pitch=-3.0, yaw=0.0),
        ),
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )
    result = {}

    def receive(image: Any) -> None:
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4
        )
        result["rgb"] = np.ascontiguousarray(bgra[:, :, 2::-1])
        result["frame"] = int(image.frame)

    camera.listen(receive)

    from structured_command_parser.src.modernbert_service import (
        ModernBertCommandService,
    )

    service = ModernBertCommandService(str(args.parser_model), device=args.device)
    service.warmup()
    parser = service.parser
    parser.load()
    text = "Keep the current lane at 45.0 kilometers per hour."
    encoded = parser.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=parser.max_length
    )
    encoded = {key: value.to(parser.device) for key, value in encoded.items()}
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
        output_dir / "intents" / "keep_lane_45.pt",
    )

    samples = 0
    progress_points = list(range(0, 5000, 25))
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for progress_m in progress_points:
            point = route_manager.route[
                min(route_manager.current_index, len(route_manager.route) - 1)
            ]
            candidates = [
                item
                for item in route_manager.route
                if abs(float(item["distance_m"]) - progress_m) <= 15.0
            ]
            if not candidates:
                continue
            point = min(
                candidates,
                key=lambda item: abs(float(item["distance_m"]) - progress_m),
            )
            transform = carla.Transform(
                carla.Location(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=float(point["z"]) + 1.0,
                ),
                carla.Rotation(yaw=float(point["yaw"])),
            )
            ego.set_transform(transform)
            ego.set_target_velocity(carla.Vector3D())
            ego.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(8):
                world.tick()
            if "frame" not in result:
                continue
            sample_id = f"scene1_finetune_{int(progress_m)}_{result['frame']}"
            front = torch.from_numpy(result["rgb"]).permute(2, 0, 1)
            images = torch.stack([front, front, front, front])
            velocity = ego.get_velocity()
            acceleration = ego.get_acceleration()
            angular = ego.get_angular_velocity()
            control = ego.get_control()
            ego_wp = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True
            )
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
                            12.5,
                            float(
                                bool(
                                    ego_wp is not None
                                    and ego_wp.is_junction
                                )
                            ),
                        ]
                    ),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "environment_features": torch.tensor(
                        [
                            0.05, 0.0, 0.0, 0.1, -1.0 / 360.0, 0.5,
                            0.02, 0.00075, 0.02, 0.0, 0.1, 0.8, 0.45, 0.45,
                        ]
                    ),
                    "camera_view_mask": torch.tensor(
                        [True, False, False, False]
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
                        "source_dataset": "CARLA_scene1_finetune_clear_day",
                        "source_frame": int(result["frame"]),
                        "route_s_m": round(float(progress_m), 3),
                        "split_group": "scene1_finetune:route_000",
                        "split": split,
                        "counterfactual_set_id": sample_id,
                        "variant_type": "scene1_cruise_front_camera",
                        "command_id": "keep_lane_45",
                        "source_text": text,
                        "normalized_text": text,
                        "camera_order": ["front", "left", "right", "rear"],
                        "camera_view_mask": [True, False, False, False],
                        "image_paths": [f"images/{sample_id}.pt"] * 4,
                        "image_tensor_path": f"images/{sample_id}.pt",
                        "tensor_path": f"tensors/{sample_id}.pt",
                        "intent_tensor_path": "intents/keep_lane_45.pt",
                        "label": {
                            "action": "keep_lane",
                            "target_speed_kmh": 45.0,
                            "target_lane": None,
                        },
                        "risk_level": "low",
                        "risk_reason_codes": [],
                        "weather_profile": "clear-daylight",
                        "control_speed_cap_kmh": 45.0,
                        "capture_quality": "scene1_front_camera_hard_negative",
                        "sampling_weight": 40.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            samples += 1

    camera.destroy()
    scenario.destroy()
    try:
        world.apply_settings(original)
    except RuntimeError:
        pass
    print(f"sampled {samples} rows -> {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
