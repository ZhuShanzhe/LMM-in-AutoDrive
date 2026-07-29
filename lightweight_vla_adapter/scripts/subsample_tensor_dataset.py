from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically subsample a trusted VLA tensor dataset"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    source_path = Path(args.input)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(source, dict) or "action_targets" not in source:
        raise ValueError("input must be a trusted VLA tensor dictionary")
    samples = int(source["action_targets"].shape[0])
    for key, value in source.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key}: expected tensor")
        if value.ndim == 0 or int(value.shape[0]) != samples:
            raise ValueError(f"{key}: inconsistent sample dimension")

    selected_count = min(samples, args.max_samples)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(samples, generator=generator)[:selected_count]
    output_data = {
        key: value.index_select(0, indices)
        for key, value in source.items()
    }
    action_counts = torch.bincount(
        output_data["action_targets"].long(),
        minlength=9,
    ).tolist()

    del source
    gc.collect()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    manifest = {
        "schema_version": "1.0.0",
        "split": args.split,
        "source": str(source_path),
        "source_samples": samples,
        "samples": selected_count,
        "seed": args.seed,
        "action_counts": action_counts,
        "keys": sorted(output_data),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
