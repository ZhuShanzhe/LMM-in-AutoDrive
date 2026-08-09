"""Build multimodal VLA supervision from nuScenes camera, LiDAR and 3D boxes.

The output follows the same manifest/tensor contract as the CARLA collection.
All paths are relative to ``--output-dir`` so the prepared dataset can be
moved as a unit.  nuScenes scene splits are preserved to avoid frame leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.io import read_image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structured_command_parser.src.modernbert_service import ModernBertCommandService


CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
COMMANDS = {
    "visual_safe": {
        "zh": "根据当前道路与交通情况安全行驶。",
        "en": "Drive safely according to the current road and traffic situation.",
        "action": None,
        "speed": 40.0,
        "lane": None,
    },
    "keep_lane": {
        "zh": "保持当前车道安全行驶。",
        "en": "Keep the current lane and drive safely.",
        "action": "keep_lane",
        "speed": 40.0,
        "lane": None,
    },
    "accelerate": {
        "zh": "道路安全时平稳提速至五十公里每小时。",
        "en": "Accelerate smoothly to 50 kilometers per hour when safe.",
        "action": "accelerate",
        "speed": 50.0,
        "lane": None,
    },
    "decelerate": {
        "zh": "平稳减速至二十五公里每小时。",
        "en": "Slow down smoothly to 25 kilometers per hour.",
        "action": "decelerate",
        "speed": 25.0,
        "lane": None,
    },
    "stop": {
        "zh": "在安全位置停车。",
        "en": "Stop the vehicle at a safe position.",
        "action": "stop",
        "speed": 0.0,
        "lane": None,
    },
    "emergency_brake": {
        "zh": "立即紧急制动。",
        "en": "Brake immediately.",
        "action": "emergency_brake",
        "speed": 0.0,
        "lane": None,
    },
    "lane_change_left": {
        "zh": "确认安全后向左变道。",
        "en": "Change to the left lane when it is safe.",
        "action": "lane_change_left",
        "speed": 35.0,
        "lane": "left",
    },
    "lane_change_right": {
        "zh": "确认安全后向右变道。",
        "en": "Change to the right lane when it is safe.",
        "action": "lane_change_right",
        "speed": 35.0,
        "lane": "right",
    },
    "turn_left": {
        "zh": "前方路口安全左转。",
        "en": "Turn left safely at the next junction.",
        "action": "turn_left",
        "speed": 25.0,
        "lane": None,
    },
    "turn_right": {
        "zh": "前方路口安全右转。",
        "en": "Turn right safely at the next junction.",
        "action": "turn_right",
        "speed": 25.0,
        "lane": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def category_flags(name: str) -> tuple[bool, bool, bool]:
    lowered = name.lower()
    vehicle = lowered.startswith("vehicle.") and not any(
        token in lowered for token in ("bicycle", "motorcycle")
    )
    vru = lowered.startswith("human.pedestrian") or any(
        token in lowered for token in ("bicycle", "motorcycle")
    )
    static = any(token in lowered for token in ("barrier", "trafficcone"))
    return vehicle, vru, static


def grid_cell(
    x: float,
    y: float,
    *,
    height: int = 64,
    width: int = 64,
) -> tuple[int, int] | None:
    if not (-20.0 <= x <= 60.0 and -30.0 <= y <= 30.0):
        return None
    row = round((60.0 - x) / 80.0 * (height - 1))
    column = round((y + 30.0) / 60.0 * (width - 1))
    return int(row), int(column)


def lidar_bev(path: Path) -> torch.Tensor:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"invalid nuScenes point cloud: {path}")
    points = points.reshape(-1, 5)
    x = points[:, 0]
    # nuScenes uses y-left; the runtime contract uses y-right.
    y = -points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    valid = (x >= -20.0) & (x <= 60.0) & (y >= -30.0) & (y <= 30.0)
    x, y, z, intensity = x[valid], y[valid], z[valid], intensity[valid]
    rows = np.rint((60.0 - x) / 80.0 * 63.0).astype(np.int64)
    columns = np.rint((y + 30.0) / 60.0 * 63.0).astype(np.int64)
    counts = np.zeros((64, 64), dtype=np.float32)
    max_height = np.full((64, 64), -5.0, dtype=np.float32)
    max_intensity = np.zeros((64, 64), dtype=np.float32)
    np.add.at(counts, (rows, columns), 1.0)
    np.maximum.at(max_height, (rows, columns), z)
    np.maximum.at(max_intensity, (rows, columns), intensity)
    occupied = counts > 0
    distance = np.zeros((64, 64), dtype=np.float32)
    distance[rows, columns] = np.minimum(
        np.sqrt(x * x + y * y) / 100.0, 1.0
    )
    result = np.stack(
        [
            occupied.astype(np.float32),
            np.where(occupied, np.clip(max_height / 5.0, -1.0, 1.0), 0.0),
            distance,
            np.minimum(np.log1p(counts) / math.log(17.0), 1.0)
            * np.clip(max_intensity, 0.0, 1.0),
        ]
    )
    return torch.from_numpy(result)


def semantic_tensors(boxes: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, list[str]]:
    camera = torch.zeros((8, 64, 64), dtype=torch.float32)
    candidates = torch.zeros((32, 12), dtype=torch.float32)
    mask = torch.zeros(32, dtype=torch.bool)
    reasons: list[str] = []
    risk = "low"
    ordered = sorted(boxes, key=lambda box: float(np.linalg.norm(box.center[:2])))
    for index, box in enumerate(ordered[:32]):
        x = float(box.center[0])
        y = -float(box.center[1])
        z = float(box.center[2])
        distance = math.sqrt(x * x + y * y + z * z)
        velocity = getattr(box, "velocity", (0.0, 0.0, 0.0))
        vx = float(velocity[0]) if math.isfinite(float(velocity[0])) else 0.0
        vy = -float(velocity[1]) if math.isfinite(float(velocity[1])) else 0.0
        vehicle, vru, static = category_flags(str(box.name))
        left = y < -2.2
        right = y > 2.2
        same = not left and not right
        candidates[index] = torch.tensor(
            [x, y, z, distance, vx, vy, 1.0, left, same, right, vru, vehicle]
        )
        mask[index] = True
        cell = grid_cell(x, y)
        if cell is not None:
            row, column = cell
            camera[0, row, column] = float(vehicle)
            camera[1, row, column] = float(vru)
            camera[2, row, column] = float(static)
            camera[3, row, column] = 1.0
            camera[4, row, column] = float(left)
            camera[5, row, column] = float(same)
            camera[6, row, column] = float(right)
            camera[7, row, column] = min(distance / 100.0, 1.0)
        level = "low"
        if x > 0.0:
            if (vru and abs(y) < 4.5 and x < 15.0) or (
                vehicle and abs(y) < 2.2 and x < 8.0
            ) or (static and abs(y) < 2.2 and x < 7.0):
                level = "high"
            elif (vru and abs(y) < 8.0 and x < 30.0) or (
                vehicle and abs(y) < 3.5 and x < 22.0
            ) or (static and abs(y) < 3.5 and x < 18.0):
                level = "medium"
        if RISK_ORDER[level] > RISK_ORDER[risk]:
            risk = level
        if level != "low":
            reasons.append(f"nuscenes_{level}:{box.name}")
    return camera, candidates, mask, risk, sorted(set(reasons))


def encode_intents(output: Path, parser_model: Path, device: str) -> dict[tuple[str, str], Path]:
    destination = output / "intents"
    destination.mkdir(parents=True, exist_ok=True)
    service = ModernBertCommandService(str(parser_model), device=device)
    service.parser.load()
    parser = service.parser
    paths: dict[tuple[str, str], Path] = {}
    for command_id, command in COMMANDS.items():
        for language in ("zh", "en"):
            text = str(command[language])
            encoded = parser.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=parser.max_length,
            )
            encoded = {key: value.to(parser.device) for key, value in encoded.items()}
            with torch.inference_mode():
                tokens = parser.model.backbone(**encoded).last_hidden_state
            path = destination / f"{command_id}_{language}.pt"
            torch.save(
                {
                    "intent_tokens": tokens[0].detach().float().cpu(),
                    "intent_mask": encoded["attention_mask"][0].detach().bool().cpu(),
                },
                path,
            )
            paths[(command_id, language)] = path
    return paths


def sample_speed(nusc, sample: dict) -> float:
    if not sample.get("prev"):
        return 0.0
    current_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    previous_sample = nusc.get("sample", sample["prev"])
    previous_data = nusc.get("sample_data", previous_sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", current_data["ego_pose_token"])
    previous_pose = nusc.get("ego_pose", previous_data["ego_pose_token"])
    dt = (int(current_data["timestamp"]) - int(previous_data["timestamp"])) / 1_000_000.0
    if dt <= 0.0:
        return 0.0
    distance = float(np.linalg.norm(
        np.asarray(current_pose["translation"]) - np.asarray(previous_pose["translation"])
    ))
    return min(distance / dt, 40.0)


def guarded_label(command: dict, risk: str) -> tuple[str, float, str | None]:
    action = command["action"]
    if action is None:
        action = {"low": "keep_lane", "medium": "decelerate", "high": "emergency_brake"}[risk]
    elif risk == "high" and action not in {"stop", "emergency_brake"}:
        action = "emergency_brake"
    elif risk == "medium" and action == "accelerate":
        action = "decelerate"
    speed = float(command["speed"])
    lane = command["lane"]
    if action == "emergency_brake":
        speed, lane = 0.0, None
    elif action == "decelerate" and risk == "medium":
        speed, lane = min(speed, 20.0), None
    return str(action), speed, lane


def main() -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    args = parse_args()
    output = args.output_dir.resolve()
    (output / "tensors").mkdir(parents=True, exist_ok=True)
    (output / "image_tensors").mkdir(parents=True, exist_ok=True)
    intent_paths = encode_intents(output, args.parser_model, args.device)
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scenes = {scene["token"]: scene for scene in nusc.scene}
    train_names = set(splits.mini_train if args.version == "v1.0-mini" else splits.train)
    official_val = sorted(splits.mini_val if args.version == "v1.0-mini" else splits.val)
    validation_names = set(official_val[::2])
    test_names = set(official_val[1::2])
    if not test_names:
        test_names = set(validation_names)
    manifest = (output / "manifest.jsonl").open("w", encoding="utf-8")
    counts: Counter[str] = Counter()
    processed = 0
    try:
        for sample in nusc.sample:
            scene_name = scenes[sample["scene_token"]]["name"]
            if scene_name in train_names:
                split = "train"
            elif scene_name in validation_names:
                split = "validation"
            elif scene_name in test_names:
                split = "test"
            else:
                continue
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
            camera_bev, candidates, candidate_mask, risk, reasons = semantic_tensors(boxes)
            tensor_path = output / "tensors" / f"{sample['token']}.pt"
            speed_mps = sample_speed(nusc, sample)
            torch.save(
                {
                    "camera_bev": camera_bev,
                    "lidar_bev": lidar_bev(Path(lidar_path)),
                    "ego_features": torch.tensor(
                        [speed_mps, 0.0, 0.0, 0.0, 0.0, 0.0, 13.889, 0.0],
                        dtype=torch.float32,
                    ),
                    "candidate_features": candidates,
                    "candidate_mask": candidate_mask,
                    "environment_features": torch.tensor(
                        [0.2, 0.0, 0.0, 0.1, 0.5, 0.5, 0.0, 1.0, 0.02, 0.0, 0.1, 0.8, 0.5, 0.5],
                        dtype=torch.float32,
                    ),
                },
                tensor_path,
            )
            images = []
            for camera in CAMERAS:
                sample_data = nusc.get("sample_data", sample["data"][camera])
                image = read_image(str(args.dataroot / sample_data["filename"]))[:3]
                images.append(image)
            image_tensor = F.interpolate(
                torch.stack(images).float(),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).clamp_(0.0, 255.0).to(torch.uint8)
            image_path = output / "image_tensors" / f"{sample['token']}.pt"
            torch.save(image_tensor, image_path)
            for command_id, command in COMMANDS.items():
                action, target_speed, target_lane = guarded_label(command, risk)
                for language in ("zh", "en"):
                    sample_id = f"nuscenes_{sample['token']}_{command_id}_{language}"
                    row = {
                        "schema_version": "universal_vla_training_sample/1.0",
                        "sample_id": sample_id,
                        "source_dataset": "nuScenes",
                        "source_frame_id": sample["token"],
                        "scene_name": scene_name,
                        "split_group": scene_name,
                        "split": split,
                        "command_id": command_id,
                        "source_text": command[language],
                        "source_language": language,
                        "camera_order": ["front", "left", "right", "rear"],
                        "camera_view_mask": [True, True, True, True],
                        "image_tensor_path": image_path.relative_to(output).as_posix(),
                        "tensor_path": tensor_path.relative_to(output).as_posix(),
                        "intent_tensor_path": intent_paths[(command_id, language)].relative_to(output).as_posix(),
                        "label": {
                            "action": action,
                            "target_speed_kmh": target_speed,
                            "target_lane": target_lane,
                        },
                        "risk_level": risk,
                        "risk_reason_codes": reasons,
                        "control_speed_cap_kmh": 50.0,
                        "variant_type": "visual_counterfactual" if command_id == "visual_safe" else "external_visual_grounding",
                        "counterfactual_set_id": f"{command_id}:{language}:{scene_name}",
                    }
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[f"split:{split}"] += 1
                    counts[f"action:{action}"] += 1
                    counts[f"risk:{risk}"] += 1
            processed += 1
            if args.limit is not None and processed >= args.limit:
                break
    finally:
        manifest.close()
    inventory = {
        "schema_version": "universal_vla_nuscenes_dataset/1.0",
        "source": "nuScenes",
        "version": args.version,
        "frames": processed,
        "samples": sum(value for key, value in counts.items() if key.startswith("split:")),
        "counts": dict(sorted(counts.items())),
        "modalities": ["four_rgb", "lidar_point_cloud_bev", "3d_candidate_entities", "ego_state", "environment_state", "text_zh_en"],
    }
    (output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
