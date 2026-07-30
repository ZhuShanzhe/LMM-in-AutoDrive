from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REQUIRED_KEYS = (
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
ACTION_LABELS = (
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
)


def choose(
    candidates: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if candidates.numel() == 0:
        raise ValueError("cannot sample from an empty candidate set")
    if count <= candidates.numel():
        order = torch.randperm(candidates.numel(), generator=generator)[:count]
        return candidates[order]
    positions = torch.randint(
        candidates.numel(),
        (count,),
        generator=generator,
    )
    return candidates[positions]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append safety- and class-balanced real samples"
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unsafe-extra", type=int, default=0)
    parser.add_argument("--class-minimum", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    for name, data in (("base", base), ("source", source)):
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{name} is missing tensors: {', '.join(missing)}")
    if args.unsafe_extra and "safety_targets" not in source:
        raise ValueError("source needs safety_targets for unsafe sampling")

    generator = torch.Generator().manual_seed(args.seed)
    selected_parts = []
    reasons: dict[str, int] = {}
    if args.unsafe_extra:
        unsafe = torch.where(~source["safety_targets"].bool())[0]
        selected = choose(unsafe, args.unsafe_extra, generator)
        selected_parts.append(selected)
        reasons["unsafe"] = int(selected.numel())

    base_targets = base["action_targets"].long()
    counts = torch.bincount(base_targets, minlength=len(ACTION_LABELS))
    if selected_parts:
        selected_targets = source["action_targets"][torch.cat(selected_parts)].long()
        counts += torch.bincount(selected_targets, minlength=len(ACTION_LABELS))
    for class_index, label in enumerate(ACTION_LABELS):
        deficit = max(0, args.class_minimum - int(counts[class_index]))
        if not deficit:
            continue
        candidates = torch.where(
            source["action_targets"].long() == class_index
        )[0]
        selected = choose(candidates, deficit, generator)
        selected_parts.append(selected)
        counts[class_index] += selected.numel()
        reasons[f"class:{label}"] = int(selected.numel())

    selected = (
        torch.cat(selected_parts)
        if selected_parts
        else torch.empty(0, dtype=torch.long)
    )
    output_data = {
        key: torch.cat([base[key], source[key][selected]], dim=0)
        for key in REQUIRED_KEYS
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output)
    manifest = {
        "schema_version": "1.0.0",
        "base": str(Path(args.base)),
        "source": str(Path(args.source)),
        "base_samples": int(base_targets.numel()),
        "appended_samples": int(selected.numel()),
        "samples": int(output_data["action_targets"].shape[0]),
        "selection": reasons,
        "action_counts": {
            label: int(count)
            for label, count in zip(ACTION_LABELS, counts.tolist())
        },
        "seed": args.seed,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
