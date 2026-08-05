#!/usr/bin/env python3
"""Build the eight-class YOLO dataset from portable JSONL manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from build_specialized_yolo_dataset import (
    CLASSES,
    load_candidates,
    uniform_sample,
    weighted_sample,
    write_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--val-limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    quality = Counter()
    train = load_candidates(args.train_manifest, quality_counts=quality)
    val = load_candidates(args.val_manifest, quality_counts=quality)
    selected_train = weighted_sample(train, args.train_limit, seed=args.seed)
    selected_val = uniform_sample(val, args.val_limit, seed=args.seed + 1)
    random.Random(args.seed + 2).shuffle(selected_train)
    random.Random(args.seed + 3).shuffle(selected_val)
    train_inventory = write_split(args.output, "train", selected_train)
    val_inventory = write_split(args.output, "val", selected_val)
    dataset_yaml = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
        "",
    ]
    (args.output / "dataset.yaml").write_text("\n".join(dataset_yaml), encoding="utf-8")
    inventory = {
        "dataset_version": "manifest_yolo_v1",
        "classes": list(CLASSES),
        "seed": args.seed,
        "quality_counts": dict(quality),
        "train": train_inventory,
        "val": val_inventory,
        "source_manifests": {
            "train": [os.path.relpath(path, Path.cwd()) for path in args.train_manifest],
            "val": [os.path.relpath(path, Path.cwd()) for path in args.val_manifest],
        },
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
