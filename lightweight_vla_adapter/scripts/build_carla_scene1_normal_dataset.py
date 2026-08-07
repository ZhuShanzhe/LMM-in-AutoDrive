"""Build front-camera Scene 1 hard negatives from a CARLA collection.

The input is produced by ``run_control_experiment.py --record-images``.  The
output follows the portable universal VLA manifest contract: generated paths
are relative to the output directory, while source PNGs remain referenced by
their resolved path and may be copied into the submitted image separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structured_command_parser.src.modernbert_service import ModernBertCommandService


COMMANDS = {
    "keep_lane_45": {
        "text": "Keep the current lane at 45.0 kilometers per hour.",
        "action": "keep_lane",
        "speed": 45.0,
        "lane": None,
    },
    "accelerate_45": {
        "text": "Accelerate smoothly to 45.0 kilometers per hour when safe.",
        "action": "accelerate",
        "speed": 45.0,
        "lane": None,
    },
    "decelerate_30": {
        "text": "Slow down smoothly to 30.0 kilometers per hour.",
        "action": "decelerate",
        "speed": 30.0,
        "lane": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-weight", type=float, default=4.0)
    return parser.parse_args()


def encode_intents(output: Path, model: Path, device: str) -> dict[str, Path]:
    destination = output / "intents"
    destination.mkdir(parents=True, exist_ok=True)
    service = ModernBertCommandService(str(model), device=device)
    service.parser.load()
    parser = service.parser
    paths = {}
    for command_id, command in COMMANDS.items():
        encoded = parser.tokenizer(
            command["text"], return_tensors="pt", truncation=True,
            max_length=parser.max_length,
        )
        encoded = {key: value.to(parser.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        path = destination / f"{command_id}.pt"
        torch.save(
            {
                "intent_tokens": tokens[0].detach().float().cpu(),
                "intent_mask": encoded["attention_mask"][0].detach().bool().cpu(),
            },
            path,
        )
        paths[command_id] = path
    return paths


def load_frames(path: Path) -> dict[int, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[int(row["frame"])] = row
    return records


def split_for(index: int, count: int) -> str:
    ratio = index / float(max(1, count))
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "validation"
    return "test"


def main() -> None:
    args = parse_args()
    if args.sampling_weight <= 0.0:
        raise ValueError("--sampling-weight must be positive")
    collection = args.collection_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "tensors").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    images = sorted((collection / "camera_frames").glob("*.png"))
    if not images:
        raise ValueError("collection contains no camera_frames/*.png")
    frame_records = load_frames(collection / "frames.jsonl")
    intent_paths = encode_intents(output, args.parser_model, args.device)
    counts: Counter[str] = Counter()
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, image in enumerate(images):
            frame = int(image.stem)
            portable_image = output / "images" / image.name
            shutil.copy2(image, portable_image)
            record = frame_records.get(frame, {})
            speed_mps = float(record.get("ego", {}).get("speed_kmh", 0.0)) / 3.6
            progress_m = float(record.get("distance_m", 0.0))
            state_path = output / "tensors" / f"{frame:08d}.pt"
            torch.save(
                {
                    "camera_bev": torch.zeros(8, 64, 64),
                    "lidar_bev": torch.zeros(4, 64, 64),
                    "ego_features": torch.tensor(
                        [speed_mps, 0.0, 0.0, 0.0, progress_m / 5000.0, 0.0, 12.5, 0.0],
                        dtype=torch.float32,
                    ),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "environment_features": torch.tensor(
                        [0.0, 0.0, 0.0, 0.1, 0.5, 0.65, 0.0, 1.0, 0.02, 0.0, 0.1, 0.8, 0.45, 0.45],
                        dtype=torch.float32,
                    ),
                },
                state_path,
            )
            group_index = index // 20
            group_count = (len(images) + 19) // 20
            split = split_for(group_index, group_count)
            for command_id, command in COMMANDS.items():
                row = {
                    "schema_version": "universal_vla_training_sample/1.0",
                    "sample_id": f"scene1_town04_{frame:08d}_{command_id}",
                    "source_dataset": "CARLA_scene1_hard_negative",
                    "source_frame": frame,
                    "route_s_m": round(progress_m, 3),
                    "split_group": f"scene1_route_{group_index:03d}",
                    "split": split,
                    "counterfactual_set_id": f"scene1_town04:{frame:08d}",
                    "variant_type": "scene1_front_camera_hard_negative",
                    "command_id": command_id,
                    "source_text": command["text"],
                    "normalized_text": command["text"],
                    "camera_order": ["front", "left", "right", "rear"],
                    "camera_view_mask": [True, False, False, False],
                    "image_paths": [portable_image.relative_to(output).as_posix()] * 4,
                    "tensor_path": state_path.relative_to(output).as_posix(),
                    "intent_tensor_path": intent_paths[command_id].relative_to(output).as_posix(),
                    "label": {
                        "action": command["action"],
                        "target_speed_kmh": command["speed"],
                        "target_lane": command["lane"],
                    },
                    "risk_level": "low",
                    "risk_reason_codes": [],
                    "weather_profile": "clear-daylight",
                    "control_speed_cap_kmh": 45.0,
                    "capture_quality": "exact_front_camera",
                    "sampling_weight": args.sampling_weight,
                }
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"split:{split}"] += 1
                counts[f"action:{command['action']}"] += 1
    inventory = {
        "schema_version": "carla_scene1_normal_dataset/1.0",
        "source_collection": str(collection),
        "frames": len(images),
        "samples": sum(value for key, value in counts.items() if key.startswith("split:")),
        "counts": dict(sorted(counts.items())),
        "modalities": ["front_rgb", "ego_state", "environment_state", "text"],
        "camera_view_mask": [True, False, False, False],
    }
    (output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
