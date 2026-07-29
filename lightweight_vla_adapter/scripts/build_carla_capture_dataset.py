from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_understanding.src.risk_interface import assess_scene_risk

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


ACTION_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}
LANE_TARGET = {"lane_change_left": 1, "lane_change_right": 2}
TEMPLATES = {
    "straight_driving": (
        ("continuous", "Keep the current lane."),
        ("continuous", "Stay in this lane."),
        ("continuous", "Maintain the present lane."),
        ("continuous", "Drive straight and remain in the current lane."),
        ("continuous", "Continue straight without changing lanes."),
    ),
    "pedestrian_crossing": (
        ("continuous", "Keep the current lane."),
        ("continuous", "Continue straight and watch the road ahead."),
        ("decelerate", "Slow down for the pedestrian ahead."),
        ("decelerate", "Reduce speed near the crossing pedestrian."),
        ("decelerate", "Brake gently and yield to the pedestrian."),
    ),
    "emergency_brake": (
        ("continuous", "Keep the current lane."),
        ("continuous", "Continue straight while maintaining a safe distance."),
        ("decelerate", "Slow down behind the vehicle ahead."),
        ("decelerate", "Reduce speed and increase the following distance."),
        ("emergency", "Brake hard now."),
        ("emergency", "Apply the emergency brake immediately."),
    ),
}


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
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            stop = start + len(batch_texts)
            tokens[start:stop].copy_(hidden.float().cpu().to(torch.float16))
            masks[start:stop].copy_(encoded["attention_mask"].bool().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens, masks


def target_action(intent: str, recommended: str) -> str:
    if intent == "emergency":
        return "emergency_brake"
    if intent == "decelerate":
        return "emergency_brake" if recommended == "emergency_brake" else "decelerate"
    if recommended == "emergency_brake":
        return "emergency_brake"
    if recommended == "decelerate":
        return "decelerate"
    return "keep_lane"


def pointer_target(world_state: dict[str, Any]) -> int:
    for index, obj in enumerate(world_state.get("objects", [])[:32]):
        if obj.get("lane_relation") not in {"ego_lane", "same"}:
            continue
        relative = obj.get("relative_position_ego_m") or {}
        if float(relative.get("longitudinal", 0.0)) <= 0.0:
            continue
        if obj.get("category") in {"vehicle", "pedestrian", "cyclist"}:
            return index
    return -100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build route-isolated CARLA capture tensors"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    args = parser.parse_args()

    split_root = Path(args.capture_root) / args.split
    indices = sorted(split_root.glob("*/scene_understanding/capture_index.jsonl"))
    if not indices:
        raise ValueError(f"no capture indices found below {split_root}")
    rasterizer = StructuredBEVRasterizer(height=64, width=64, max_candidates=32)
    empty_tokens = torch.zeros(1, 1, 768)
    empty_mask = torch.ones(1, 1, dtype=torch.bool)
    frame_features = {}
    records = []
    action_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for index_path in indices:
        scenario = index_path.parents[1].name
        templates = TEMPLATES[scenario]
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            world_state = json.loads(
                Path(item["world_state_path"]).read_text(encoding="utf-8")
            )
            risk = assess_scene_risk(world_state)
            batch, _ = rasterizer.build(
                world_state,
                intent_tokens=empty_tokens,
                intent_mask=empty_mask,
            )
            key = f"{scenario}:{world_state['frame_id']}"
            frame_features[key] = (
                batch.camera_bev[0],
                batch.lidar_bev[0],
                batch.ego_features[0],
                batch.candidate_features[0],
                batch.candidate_mask[0],
                pointer_target(world_state),
            )
            current_speed_kmh = float(world_state["ego"]["speed_mps"]) * 3.6
            for intent, text in templates:
                action = target_action(intent, risk["recommended_action"])
                if action in {"stop", "emergency_brake"}:
                    speed = 0.0
                elif action == "decelerate":
                    speed = max(0.0, current_speed_kmh - 5.0)
                else:
                    speed = current_speed_kmh
                records.append(
                    (
                        key,
                        text,
                        action,
                        speed,
                        risk["recommended_action"] not in {
                            "decelerate",
                            "emergency_brake",
                        },
                    )
                )
                action_counts[action] += 1
                scenario_counts[scenario] += 1
                risk_counts[risk["recommended_action"]] += 1

    texts = [record[1] for record in records]
    intent_tokens, intent_mask = encode_texts(
        texts,
        args.model_path,
        max_length=args.max_length,
        batch_size=args.embedding_batch_size,
        device=torch.device(args.device),
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
        "lane_targets": torch.zeros(count, dtype=torch.long),
        "pointer_targets": torch.empty(count, dtype=torch.long),
        "safety_targets": torch.empty(count, dtype=torch.bool),
    }
    for index, (key, _, action, speed, safe) in enumerate(records):
        camera, lidar, ego, candidates, mask, pointer = frame_features[key]
        data["camera_bev"][index].copy_(camera.to(torch.float16))
        data["lidar_bev"][index].copy_(lidar.to(torch.float16))
        data["ego_features"][index].copy_(ego.to(torch.float16))
        data["candidate_features"][index].copy_(candidates.to(torch.float16))
        data["candidate_mask"][index].copy_(mask)
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
        "samples": count,
        "capture_indices": [str(path) for path in indices],
        "scenario_counts": dict(scenario_counts),
        "risk_counts": dict(risk_counts),
        "action_counts": dict(action_counts),
        "data_kind": "carla_0.9.16_structured_world_state_proxy",
        "limitations": [
            "Camera and lidar BEV tensors are rasterized from synchronized CARLA truth",
            "Each split uses a separate CARLA run but the same three scenario families",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
