from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import io
import json
import math
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


ACTION_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}
LANE_TARGET = {"lane_change_left": 1, "lane_change_right": 2}
MODALITIES = ("measurements", "boxes", "lidar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tensor shard from matching SimLingo raw and Dreamer archives"
    )
    parser.add_argument("--raw-archive", required=True)
    parser.add_argument("--dreamer-archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--partition-modulus", type=int, default=1)
    parser.add_argument(
        "--route-partitions",
        default="0",
        help="Comma-separated route hash partitions to include",
    )
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def stable_score(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def nested_json(member: tarfile.TarInfo, archive: tarfile.TarFile) -> Any:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise OSError(f"cannot read {member.name}")
    return json.loads(gzip.decompress(extracted.read()))


def frame_key(path: str, modality: str) -> str | None:
    marker = f"/{modality}/"
    if marker not in path:
        return None
    prefix, filename = path.split(marker, 1)
    parts = prefix.split("/", 1)
    if len(parts) != 2:
        return None
    return f"{parts[1]}/{Path(filename).name.split('.', 1)[0]}"


def select_dreamer_frames(
    archive_path: Path,
    max_frames: int,
    seed: int,
    partition_modulus: int,
    route_partitions: set[int],
) -> dict[str, dict[str, Any]]:
    capacity = max_frames if max_frames > 0 else None
    heap: list[tuple[int, int, str, dict[str, Any]]] = []
    selected: dict[str, dict[str, Any]] = {}
    sequence = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            key = frame_key(member.name, "dreamer")
            if key is None:
                continue
            route_key = key.rsplit("/", 1)[0]
            partition = stable_score(route_key, 0) % partition_modulus
            if partition not in route_partitions:
                continue
            payload = nested_json(member, archive)
            if not isinstance(payload, dict):
                continue
            if capacity is None:
                selected[key] = payload
                continue
            score = stable_score(key, seed)
            item = (-score, sequence, key, payload)
            sequence += 1
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if capacity is not None:
        selected = {item[2]: item[3] for item in heap}
    return selected


def lidar_to_bev(payload: bytes, height: int = 64, width: int = 64) -> torch.Tensor:
    import laspy

    cloud = laspy.read(io.BytesIO(payload))
    x = np.asarray(cloud.x)
    y = np.asarray(cloud.y)
    z = np.asarray(cloud.z)
    valid = (
        (x >= -20.0)
        & (x <= 60.0)
        & (y >= -30.0)
        & (y <= 30.0)
        & (z >= -4.0)
        & (z <= 14.0)
    )
    x = x[valid]
    y = y[valid]
    z = z[valid]
    output = np.zeros((4, height, width), dtype=np.float32)
    if not len(x):
        return torch.from_numpy(output)
    row = np.rint((60.0 - x) / 80.0 * (height - 1)).astype(np.int64)
    column = np.rint((y + 30.0) / 60.0 * (width - 1)).astype(np.int64)
    low_count = np.zeros((height, width), dtype=np.float32)
    high_count = np.zeros((height, width), dtype=np.float32)
    low = z < 2.7
    np.add.at(low_count, (row[low], column[low]), 1.0)
    np.add.at(high_count, (row[~low], column[~low]), 1.0)
    count = low_count + high_count
    z_sum = np.zeros((height, width), dtype=np.float32)
    z_max = np.full((height, width), -4.0, dtype=np.float32)
    np.add.at(z_sum, (row, column), z.astype(np.float32))
    np.maximum.at(z_max, (row, column), z.astype(np.float32))
    occupied = count > 0
    mean_height = np.zeros_like(count)
    mean_height[occupied] = z_sum[occupied] / count[occupied]
    output[0] = np.clip(low_count / 5.0, 0.0, 1.0)
    output[1] = np.clip(high_count / 5.0, 0.0, 1.0)
    output[2, occupied] = np.clip(
        (mean_height[occupied] + 4.0) / 18.0,
        0.0,
        1.0,
    )
    output[3, occupied] = np.clip(
        (z_max[occupied] + 4.0) / 18.0,
        0.0,
        1.0,
    )
    return torch.from_numpy(output)


def load_raw_frames(
    archive_path: Path,
    keys: set[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, torch.Tensor],
]:
    measurements: dict[str, dict[str, Any]] = {}
    boxes: dict[str, list[dict[str, Any]]] = {}
    lidar: dict[str, torch.Tensor] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            modality = next(
                (value for value in MODALITIES if f"/{value}/" in member.name),
                None,
            )
            if modality is None:
                continue
            key = frame_key(member.name, modality)
            if key not in keys:
                continue
            if modality == "measurements":
                value = nested_json(member, archive)
                if isinstance(value, dict):
                    measurements[key] = value
            elif modality == "boxes":
                value = nested_json(member, archive)
                if isinstance(value, list):
                    boxes[key] = [
                        item for item in value if isinstance(item, dict)
                    ]
            else:
                extracted = archive.extractfile(member)
                if extracted is not None:
                    lidar[key] = lidar_to_bev(extracted.read())
    return measurements, boxes, lidar


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def fallback_action(measurement: dict[str, Any]) -> str:
    speed = number(measurement.get("speed"))
    target_speed = number(measurement.get("target_speed"), speed)
    dynamic_hazard = any(
        bool(measurement.get(key))
        for key in (
            "vehicle_hazard",
            "walker_hazard",
        )
    )
    static_hazard = any(
        bool(measurement.get(key))
        for key in (
            "light_hazard",
            "stop_sign_hazard",
            "control_brake",
        )
    )
    if dynamic_hazard and target_speed <= 0.5 and speed >= 4.0:
        return "emergency_brake"
    if dynamic_hazard or static_hazard or bool(measurement.get("brake")):
        return "stop" if target_speed <= 0.5 else "decelerate"
    command = int(number(measurement.get("command"), 4))
    if command == 1:
        return "turn_left"
    if command == 2:
        return "turn_right"
    if command == 5:
        return "lane_change_left"
    if command == 6:
        return "lane_change_right"
    if target_speed <= 0.5 and speed > 0.5:
        return "stop"
    if target_speed > speed + 0.75:
        return "accelerate"
    if target_speed < speed - 0.75:
        return "decelerate"
    return "keep_lane"


def requested_action(
    mode: str,
    alternative: dict[str, Any],
    measurement: dict[str, Any],
) -> str | None:
    if alternative.get("safe_to_execute") is False:
        return fallback_action(measurement)
    lowered = mode.lower()
    info = alternative.get("info")
    info = info if isinstance(info, dict) else {}
    if "lane_change" in lowered:
        direction = str(info.get("lane_change_direction") or "").lower()
        if "right" in direction:
            return "lane_change_right"
        if "left" in direction:
            return "lane_change_left"
        return None
    if lowered in {"stop", "emergency_stop"}:
        return "stop"
    if "crash" in lowered:
        return fallback_action(measurement)
    if lowered in {"faster", "faster_factor"} or "faster" in lowered:
        return "accelerate"
    if lowered in {"slower", "slower_factor"} or "slower" in lowered:
        return "decelerate"
    if "target_speed" in lowered or lowered == "speed":
        current = number(
            info.get("current_speed"),
            number(measurement.get("speed")),
        )
        target = number(
            info.get("target_speed"),
            number(info.get("final_speed"), current),
        )
        if target <= 0.5:
            return "stop"
        if target > current + 0.75:
            return "accelerate"
        if target < current - 0.75:
            return "decelerate"
        return "keep_lane"
    if lowered == "route":
        instruction = alternative.get("dreamer_instruction")
        text = (
            instruction
            if isinstance(instruction, str)
            else " ".join(instruction or [])
        ).lower()
        if "left" in text:
            return "lane_change_left"
        if "right" in text:
            return "lane_change_right"
        route = alternative.get("route")
        if isinstance(route, list):
            lateral = [
                number(point[1])
                for point in route
                if isinstance(point, list) and len(point) >= 2
            ]
            if lateral and max(abs(value) for value in lateral) >= 2.5:
                return (
                    "lane_change_left"
                    if lateral[-1] < 0.0
                    else "lane_change_right"
                )
        return "keep_lane"
    return None


def requested_speed_kmh(
    alternative: dict[str, Any],
    measurement: dict[str, Any],
) -> float:
    if alternative.get("safe_to_execute") is False:
        return min(max(number(measurement.get("target_speed")) * 3.6, 0.0), 100.0)
    info = alternative.get("info")
    info = info if isinstance(info, dict) else {}
    value = info.get("final_speed")
    if value is None:
        value = info.get("target_speed")
    if value is None:
        value = measurement.get("target_speed")
    return min(max(number(value) * 3.6, 0.0), 100.0)


def category(label: str) -> str:
    lowered = label.lower()
    if lowered == "ego_car":
        return "ego"
    if any(token in lowered for token in ("car", "vehicle", "truck", "bus")):
        return "vehicle"
    if any(token in lowered for token in ("walker", "pedestrian")):
        return "pedestrian"
    if any(token in lowered for token in ("bike", "bicycle", "cyclist")):
        return "cyclist"
    if "traffic_light" in lowered:
        return "traffic_light"
    if "traffic_sign" in lowered or "stop_sign" in lowered:
        return "traffic_sign"
    return lowered or "other"


def world_state(
    key: str,
    measurement: dict[str, Any],
    frame_boxes: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    candidates = []
    ego_speed = number(measurement.get("speed"))
    for box in frame_boxes:
        label = category(str(box.get("class") or ""))
        if label == "ego":
            continue
        position = box.get("position")
        if not isinstance(position, list) or len(position) < 2:
            continue
        x = number(position[0])
        y = number(position[1])
        z = number(position[2]) if len(position) >= 3 else 0.0
        distance = number(box.get("distance"), math.sqrt(x * x + y * y + z * z))
        speed = number(box.get("speed"))
        yaw = number(box.get("yaw"))
        if abs(y) <= 2.0:
            lane_relation = "same"
        elif y < 0.0:
            lane_relation = "left"
        else:
            lane_relation = "right"
        candidates.append(
            (
                distance,
                {
                    "entity_id": str(box.get("id") or f"box_{len(candidates)}"),
                    "category": label,
                    "relative_position_m": {"x": x, "y": y, "z": z},
                    "relative_velocity_mps": {
                        "x": speed * math.cos(yaw) - ego_speed,
                        "y": speed * math.sin(yaw),
                    },
                    "lane_relation": lane_relation,
                    "confidence": 1.0,
                },
            )
        )
    candidates.sort(key=lambda item: item[0])
    objects = [item[1] for item in candidates[:32]]
    controls = {
        "steer": number(measurement.get("steer")),
        "throttle": number(measurement.get("throttle")),
        "brake": float(bool(measurement.get("brake"))),
    }
    state = {
        "frame_id": key,
        "ego": {
            "speed_mps": ego_speed,
            "acceleration_mps2": 0.0,
            "yaw_rate_rps": 0.0,
            "speed_limit_mps": number(measurement.get("speed_limit"), 13.9),
            "control": controls,
        },
        "objects": objects,
        "environment": {
            "at_junction": bool(measurement.get("junction")),
            "weather": "unknown",
        },
    }
    return state, [str(item["entity_id"]) for item in objects]


def pointer_target(
    measurement: dict[str, Any],
    entity_ids: list[str],
) -> int:
    target = (
        measurement.get("speed_reduced_by_obj_id")
        or measurement.get("vehicle_affecting_id")
        or measurement.get("walker_affecting_id")
    )
    if target is None:
        return -100
    target = str(target)
    return entity_ids.index(target) if target in entity_ids else -100


def encode_texts(
    texts: list[str],
    model_path: str,
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    tokens = torch.empty(len(texts), max_length, hidden_size, dtype=torch.float16)
    masks = torch.empty(len(texts), max_length, dtype=torch.bool)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            stop = start + int(hidden.shape[0])
            tokens[start:stop].copy_(hidden.float().cpu().to(torch.float16))
            masks[start:stop].copy_(inputs["attention_mask"].bool().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens, masks


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.partition_modulus <= 0:
        raise ValueError("partition-modulus must be positive")
    route_partitions = {
        int(value)
        for value in args.route_partitions.split(",")
        if value.strip()
    }
    if not route_partitions or any(
        value < 0 or value >= args.partition_modulus
        for value in route_partitions
    ):
        raise ValueError("route-partitions must be within partition-modulus")
    raw_archive = Path(args.raw_archive)
    dreamer_archive = Path(args.dreamer_archive)
    labels = select_dreamer_frames(
        dreamer_archive,
        args.max_frames,
        args.seed,
        args.partition_modulus,
        route_partitions,
    )
    measurements, boxes, lidar = load_raw_frames(raw_archive, set(labels))
    common = sorted(set(labels) & set(measurements) & set(boxes) & set(lidar))
    if not common:
        raise ValueError("no complete same-frame records found")

    rasterizer = StructuredBEVRasterizer(height=64, width=64, max_candidates=32)
    frame_features: dict[str, tuple[Any, ...]] = {}
    records = []
    action_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    skipped_unknown_mode = 0
    for key in common:
        state, entity_ids = world_state(key, measurements[key], boxes[key])
        empty_tokens = torch.zeros(1, 1, 768)
        empty_mask = torch.ones(1, 1, dtype=torch.bool)
        batch, _ = rasterizer.build(
            state,
            intent_tokens=empty_tokens,
            intent_mask=empty_mask,
        )
        frame_features[key] = (
            batch.camera_bev[0],
            lidar[key],
            batch.ego_features[0],
            batch.candidate_features[0],
            batch.candidate_mask[0],
            pointer_target(measurements[key], entity_ids),
        )
        for mode, alternatives in labels[key].items():
            if not isinstance(alternatives, list):
                continue
            for alternative_index, alternative in enumerate(alternatives):
                if not isinstance(alternative, dict):
                    continue
                action = requested_action(mode, alternative, measurements[key])
                if action is None:
                    skipped_unknown_mode += 1
                    continue
                raw_instructions = alternative.get("dreamer_instruction") or []
                if isinstance(raw_instructions, str):
                    raw_instructions = [raw_instructions]
                instructions = [
                    value.strip()
                    for value in raw_instructions
                    if isinstance(value, str) and value.strip()
                ]
                if not instructions:
                    continue
                text = min(
                    instructions,
                    key=lambda value: stable_score(
                        f"{key}:{mode}:{alternative_index}:{value}",
                        args.seed,
                    ),
                )
                records.append(
                    (
                        key,
                        text,
                        action,
                        requested_speed_kmh(alternative, measurements[key]),
                        mode,
                        alternative.get("safe_to_execute") is not False,
                    )
                )
                action_counts[action] += 1
                mode_counts[str(mode)] += 1
                safety_counts[str(alternative.get("safe_to_execute"))] += 1
    if not records:
        raise ValueError("no supervised alternatives found")
    records.sort(
        key=lambda item: stable_score(
            f"{args.split}:{item[0]}:{item[1]}",
            args.seed + 29,
        )
    )
    texts = [item[1] for item in records]
    device = torch.device(args.device)
    intent_tokens, intent_mask = encode_texts(
        texts,
        args.model_path,
        max_length=args.max_length,
        batch_size=args.embedding_batch_size,
        device=device,
    )
    count = len(records)
    data = {
        "camera_bev": torch.empty(count, 8, 64, 64, dtype=torch.float16),
        "lidar_bev": torch.empty(count, 4, 64, 64, dtype=torch.float16),
        "ego_features": torch.empty(count, 8, dtype=torch.float16),
        "candidate_features": torch.empty(count, 32, 12, dtype=torch.float16),
        "candidate_mask": torch.empty(count, 32, dtype=torch.bool),
        "intent_tokens": intent_tokens,
        "intent_mask": intent_mask,
        "action_targets": torch.empty(count, dtype=torch.long),
        "speed_targets": torch.empty(count, dtype=torch.float32),
        "lane_targets": torch.empty(count, dtype=torch.long),
        "pointer_targets": torch.empty(count, dtype=torch.long),
        "safety_targets": torch.empty(count, dtype=torch.bool),
    }
    for index, (key, _, action, speed, _, safe) in enumerate(records):
        (
            camera_bev,
            lidar_bev,
            ego,
            candidates,
            candidate_mask,
            pointer,
        ) = frame_features[key]
        data["camera_bev"][index].copy_(camera_bev.to(torch.float16))
        data["lidar_bev"][index].copy_(lidar_bev.to(torch.float16))
        data["ego_features"][index].copy_(ego.to(torch.float16))
        data["candidate_features"][index].copy_(candidates.to(torch.float16))
        data["candidate_mask"][index].copy_(candidate_mask)
        data["action_targets"][index] = ACTION_TO_INDEX[action]
        data["speed_targets"][index] = speed
        data["lane_targets"][index] = LANE_TARGET.get(action, 0)
        data["pointer_targets"][index] = pointer
        data["safety_targets"][index] = safe

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    manifest = {
        "schema_version": "1.0.0",
        "split": args.split,
        "raw_archive": str(raw_archive),
        "dreamer_archive": str(dreamer_archive),
        "selected_label_frames": len(labels),
        "partition_modulus": args.partition_modulus,
        "route_partitions": sorted(route_partitions),
        "complete_frames": len(common),
        "samples": count,
        "action_counts": dict(action_counts),
        "mode_counts": dict(mode_counts),
        "safety_counts": dict(safety_counts),
        "skipped_unknown_mode": skipped_unknown_mode,
        "intent_hidden_size": int(intent_tokens.shape[-1]),
        "seed": args.seed,
        "data_kind": "simlingo_gt_boxes_raw_lidar",
        "camera_bev_source": "official_3d_boxes",
        "lidar_bev_source": "official_raw_laz",
        "limitations": [
            "Camera BEV uses official 3D boxes rather than image-only detections",
            "Raw RGB perception accuracy is evaluated in the upstream scene module",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
