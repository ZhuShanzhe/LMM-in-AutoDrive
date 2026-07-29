from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TENSOR_KEYS = (
    "camera_bev",
    "lidar_bev",
    "ego_features",
    "candidate_features",
    "candidate_mask",
    "intent_tokens",
    "intent_mask",
    "action_targets",
    "speed_targets",
    "lane_targets",
    "pointer_targets",
)
OPTIONAL_TENSOR_KEYS = (
    "safety_targets",
    "teacher_action_logits",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge VLA tensor datasets")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    datasets = []
    source_counts = {}
    for value in args.inputs:
        path = Path(value)
        data = torch.load(path, map_location="cpu", weights_only=True)
        missing = [key for key in TENSOR_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path} is missing tensors: {', '.join(missing)}")
        sample_count = int(data["action_targets"].shape[0])
        if any(int(data[key].shape[0]) != sample_count for key in TENSOR_KEYS):
            raise ValueError(f"{path} has inconsistent first dimensions")
        datasets.append(data)
        source_counts[str(path)] = source_counts.get(str(path), 0) + sample_count
    merged_keys = list(TENSOR_KEYS)
    merged_keys.extend(
        key
        for key in OPTIONAL_TENSOR_KEYS
        if all(
            key in data and int(data[key].shape[0]) == int(data["action_targets"].shape[0])
            for data in datasets
        )
    )
    merged = {
        key: torch.cat([data[key] for data in datasets], dim=0)
        for key in merged_keys
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    action_counts = torch.bincount(
        merged["action_targets"].long(),
        minlength=9,
    ).tolist()
    manifest = {
        "schema_version": "1.0.0",
        "split": args.split,
        "samples": int(merged["action_targets"].shape[0]),
        "source_counts": source_counts,
        "action_counts": action_counts,
        "keys": merged_keys,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
