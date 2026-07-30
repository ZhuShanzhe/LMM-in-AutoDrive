from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


ACTION_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}
LANE_TARGET = {
    "lane_change_left": 1,
    "lane_change_right": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a structured-CARLA proxy dataset from existing commands"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples-per-class", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", required=True)
    return parser.parse_args()


def stable_score(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def expected_action(record: dict[str, Any]) -> str | None:
    expected = record.get("expected") or {}
    actions = [str(value).upper() for value in expected.get("actions", [])]
    directions = [str(value).upper() for value in expected.get("directions", [])]
    commands = expected.get("commands") or []
    change = str(expected.get("change") or "").upper()
    text = str(record.get("text_en") or "").lower()

    if "EMERGENCY_BRAKE" in actions:
        return "emergency_brake"
    if any(action in actions for action in ("STOP", "PARK", "PULL_OVER", "WAIT")):
        return "stop"
    if "CHANGE_LANE" in actions or "MERGE" in actions:
        if "RIGHT" in directions or "right" in text:
            return "lane_change_right"
        return "lane_change_left"
    if "TURN" in actions or "U_TURN" in actions:
        if "RIGHT" in directions or "right" in text:
            return "turn_right"
        return "turn_left"
    if "OVERTAKE" in actions:
        return "lane_change_right" if "RIGHT" in directions else "lane_change_left"
    if "ADJUST_SPEED" in actions or "SET_SPEED" in actions:
        command_changes = {
            str(command.get("change") or "").upper()
            for command in commands
            if isinstance(command, dict)
        }
        if (
            change == "INCREASE"
            or "INCREASE" in command_changes
            or any(token in text for token in ("speed up", "accelerate", "faster"))
        ):
            return "accelerate"
        return "decelerate"
    if "FOLLOW" in actions or "APPROACH" in actions or "YIELD" in actions:
        return "decelerate"
    if any(
        action in actions
        for action in ("KEEP_LANE", "GO_STRAIGHT", "CONTINUE", "NAVIGATE", "PROCEED")
    ):
        return "keep_lane"
    return None


def select_records(
    path: Path,
    per_class: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    capacity = max(per_class * 2, per_class + 100)
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        label: [] for label in ACTION_LABELS if label != "emergency_brake"
    }
    sequence = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            action = expected_action(record)
            text = record.get("text_en")
            if action not in heaps or not isinstance(text, str) or not text.strip():
                continue
            score = stable_score(str(record.get("sample_id") or text), seed)
            item = (-score, sequence, record)
            sequence += 1
            heap = heaps[action]
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    selected = {}
    for action, heap in heaps.items():
        records = [item[2] for item in sorted(heap, reverse=True)]
        if len(records) < per_class:
            raise ValueError(
                f"not enough {action} records: need {per_class}, found {len(records)}"
            )
        selected[action] = records[:per_class]
    emergency_pool = [
        record
        for action, records in selected.items()
        if action not in {"stop", "decelerate"}
        for record in records
    ]
    emergency_pool.sort(
        key=lambda record: stable_score(
            str(record.get("sample_id") or record["text_en"]),
            seed + 17,
        )
    )
    if len(emergency_pool) < per_class:
        raise ValueError("not enough records to synthesize emergency scenes")
    selected["emergency_brake"] = emergency_pool[:per_class]
    return selected


def random_object(rng: random.Random, index: int) -> dict[str, Any]:
    lane = rng.choice(("left", "same", "right", "roadside"))
    return {
        "entity_id": f"background_{index}",
        "category": rng.choice(("vehicle", "vehicle", "pedestrian", "traffic_sign")),
        "relative_position_m": {
            "x": rng.uniform(8.0, 58.0),
            "y": {
                "left": rng.uniform(-6.0, -2.5),
                "same": rng.uniform(-1.2, 1.2),
                "right": rng.uniform(2.5, 6.0),
                "roadside": rng.choice((-1, 1)) * rng.uniform(7.0, 14.0),
            }[lane],
            "z": 0.0,
        },
        "relative_velocity_mps": {
            "x": rng.uniform(-3.0, 3.0),
            "y": 0.0,
        },
        "lane_relation": lane,
        "confidence": rng.uniform(0.75, 1.0),
    }


def build_world_state(
    action: str,
    sample_id: str,
    seed: int,
    split: str,
) -> tuple[dict[str, Any], float, int]:
    rng = random.Random(stable_score(f"{split}:{sample_id}:{action}", seed))
    speed_mps = rng.uniform(4.0, 13.0)
    objects = [random_object(rng, index) for index in range(rng.randint(1, 6))]
    pointer_target = -100
    at_junction = action in {"turn_left", "turn_right"}
    if action == "emergency_brake":
        objects.insert(
            0,
            {
                "entity_id": "critical_hazard",
                "category": rng.choice(("vehicle", "pedestrian")),
                "relative_position_m": {
                    "x": rng.uniform(2.5, 6.5),
                    "y": rng.uniform(-0.7, 0.7),
                    "z": 0.0,
                },
                "relative_velocity_mps": {"x": rng.uniform(-12.0, -5.0), "y": 0.0},
                "lane_relation": "same",
                "confidence": 1.0,
            },
        )
        target_speed = 0.0
        pointer_target = 0
    elif action == "stop":
        objects.insert(
            0,
            {
                "entity_id": "stop_reference",
                "category": rng.choice(("traffic_light", "traffic_sign", "vehicle")),
                "relative_position_m": {
                    "x": rng.uniform(7.0, 22.0),
                    "y": rng.uniform(-0.8, 0.8),
                    "z": 0.0,
                },
                "relative_velocity_mps": {"x": 0.0, "y": 0.0},
                "lane_relation": "same",
                "confidence": 1.0,
            },
        )
        target_speed = 0.0
        pointer_target = 0
    elif action == "decelerate":
        objects.insert(
            0,
            {
                "entity_id": "lead_vehicle",
                "category": "vehicle",
                "relative_position_m": {
                    "x": rng.uniform(10.0, 28.0),
                    "y": rng.uniform(-0.8, 0.8),
                    "z": 0.0,
                },
                "relative_velocity_mps": {"x": rng.uniform(-6.0, -1.0), "y": 0.0},
                "lane_relation": "same",
                "confidence": 1.0,
            },
        )
        target_speed = rng.uniform(12.0, 25.0)
        pointer_target = 0
    elif action == "accelerate":
        objects = [
            obj
            for obj in objects
            if obj["lane_relation"] != "same"
            or obj["relative_position_m"]["x"] > 35.0
        ]
        target_speed = rng.uniform(38.0, 58.0)
    elif action in {"turn_left", "turn_right"}:
        target_speed = rng.uniform(12.0, 25.0)
    else:
        target_speed = rng.uniform(28.0, 46.0)
    world_state = {
        "frame_id": f"proxy_{sample_id}",
        "ego": {
            "speed_mps": speed_mps,
            "acceleration_mps2": rng.uniform(-1.0, 1.0),
            "yaw_rate_rps": rng.uniform(-0.08, 0.08),
            "speed_limit_mps": rng.uniform(11.0, 17.0),
            "control": {
                "steer": rng.uniform(-0.08, 0.08),
                "throttle": rng.uniform(0.0, 0.6),
                "brake": 0.0,
            },
        },
        "objects": objects,
        "environment": {
            "at_junction": at_junction,
            "weather": rng.choice(("clear", "clear", "rain", "fog")),
        },
    }
    return world_state, target_speed, pointer_target


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
    tokens = torch.empty(
        len(texts),
        max_length,
        hidden_size,
        dtype=torch.float16,
    )
    masks = torch.empty(len(texts), max_length, dtype=torch.bool)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            stop = start + len(batch_texts)
            tokens[start:stop].copy_(hidden.float().cpu().to(torch.float16))
            masks[start:stop].copy_(inputs["attention_mask"].bool().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens, masks


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    selected = select_records(
        Path(args.input),
        args.samples_per_class,
        args.seed,
    )
    samples = []
    for action in ACTION_LABELS:
        for record in selected[action]:
            samples.append((action, record))
    samples.sort(
        key=lambda item: stable_score(
            str(item[1].get("sample_id") or item[1]["text_en"]),
            args.seed + 101,
        )
    )
    texts = [str(record["text_en"]) for _, record in samples]
    device = torch.device(args.device)
    intent_tokens, intent_mask = encode_texts(
        texts,
        args.model_path,
        max_length=args.max_length,
        batch_size=args.embedding_batch_size,
        device=device,
    )
    count = len(samples)
    rasterizer = StructuredBEVRasterizer(height=64, width=64, max_candidates=32)
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
    }
    source_counts: Counter[str] = Counter()
    for index, (action, record) in enumerate(samples):
        sample_id = str(record.get("sample_id") or index)
        world_state, target_speed, pointer_target = build_world_state(
            action,
            sample_id,
            args.seed,
            args.split,
        )
        batch, _ = rasterizer.build(
            world_state,
            intent_tokens=intent_tokens[index : index + 1],
            intent_mask=intent_mask[index : index + 1],
        )
        data["camera_bev"][index].copy_(batch.camera_bev[0].to(torch.float16))
        data["lidar_bev"][index].copy_(batch.lidar_bev[0].to(torch.float16))
        data["ego_features"][index].copy_(batch.ego_features[0].to(torch.float16))
        data["candidate_features"][index].copy_(
            batch.candidate_features[0].to(torch.float16)
        )
        data["candidate_mask"][index].copy_(batch.candidate_mask[0])
        data["action_targets"][index] = ACTION_TO_INDEX[action]
        data["speed_targets"][index] = target_speed
        data["lane_targets"][index] = LANE_TARGET.get(action, 0)
        data["pointer_targets"][index] = pointer_target
        source_counts[str(record.get("source") or "unknown")] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    manifest = {
        "schema_version": "1.0.0",
        "split": args.split,
        "samples": count,
        "samples_per_class": args.samples_per_class,
        "action_labels": list(ACTION_LABELS),
        "source_counts": dict(source_counts),
        "input": str(Path(args.input)),
        "model_path": args.model_path,
        "intent_hidden_size": int(intent_tokens.shape[-1]),
        "max_length": args.max_length,
        "seed": args.seed,
        "data_kind": "structured_carla_proxy",
        "limitations": [
            "BEV tensors are rasterized from structured synthetic WorldState records",
            "This split does not measure raw camera or LiDAR perception accuracy",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
