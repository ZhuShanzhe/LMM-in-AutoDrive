#!/usr/bin/env python3
"""Build a mixed YOLO dataset from CARLA capture projections."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

CATEGORY_IDS = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "motorcycle": 3,
    "traffic_light": 4,
    "traffic_sign": 5,
    "road_barrier": 6,
    "traffic_cone": 7,
}


def yolo_line(category: str, bbox: list[float]) -> str | None:
    class_id = CATEGORY_IDS.get(category)
    if class_id is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return (
        f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
        f"{x2 - x1:.6f} {y2 - y1:.6f}"
    )


def load_records(index_paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for index_path in index_paths:
        scenario = index_path.parent.parent.name
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["_scenario"] = scenario
            records.append(record)
    return records


def split_by_scenario(
    records: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    scenarios = sorted({str(record["_scenario"]) for record in records})
    for scenario in scenarios:
        group = [record for record in records if record["_scenario"] == scenario]
        random.Random(f"{seed}:{scenario}").shuffle(group)
        val_count = max(1, round(len(group) * val_fraction))
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    return train, val


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(
    records: list[dict[str, Any]],
    output_root: Path,
    split: str,
    repeats: int,
) -> dict[str, int]:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    box_count = 0
    for record in records:
        image_path = Path(record["image_path"])
        projection = json.loads(
            Path(record["projection_path"]).read_text(encoding="utf-8")
        )
        labels = [
            converted
            for obj in projection.get("objects", [])
            if (
                converted := yolo_line(
                    str(obj.get("category")), list(obj.get("bbox_2d", []))
                )
            )
            is not None
        ]
        stem = f"{record['_scenario']}_{record['frame_id']}"
        for repeat in range(repeats):
            name = f"{stem}_r{repeat:03d}"
            destination = image_dir / f"{name}{image_path.suffix.lower()}"
            link_or_copy(image_path, destination)
            (label_dir / f"{name}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            image_count += 1
            box_count += len(labels)
    return {"images": image_count, "boxes": box_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, action="append", required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-repeats", type=int, default=30)
    parser.add_argument(
        "--base-train-limit",
        type=int,
        help="Deterministically sample this many images from the base train split",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.train_repeats <= 0:
        raise ValueError("train-repeats must be positive")
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    records = load_records(args.capture_index)
    train, val = split_by_scenario(records, args.val_fraction, args.seed)
    train_stats = materialize(
        train, args.output, "train", repeats=args.train_repeats
    )
    val_stats = materialize(val, args.output, "val", repeats=1)
    base = args.base_dataset.resolve()
    base_train_entry = str(base / "images" / "train")
    if args.base_train_limit is not None:
        if args.base_train_limit <= 0:
            raise ValueError("base-train-limit must be positive")
        base_images = sorted(
            path
            for path in (base / "images" / "train").iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        random.Random(args.seed).shuffle(base_images)
        selected = sorted(base_images[: args.base_train_limit])
        base_train_list = args.output / "base_train.txt"
        base_train_list.write_text(
            "\n".join(str(path.absolute()) for path in selected) + "\n",
            encoding="utf-8",
        )
        base_train_entry = "base_train.txt"
    yaml = (
        f"path: {args.output.resolve()}\n"
        "train:\n"
        f"  - {base_train_entry}\n"
        "  - images/train\n"
        "val:\n"
        f"  - {base / 'images' / 'val'}\n"
        "  - images/val\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for name, index in CATEGORY_IDS.items())
    )
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    stats = {
        "source_records": len(records),
        "train_source_records": len(train),
        "val_source_records": len(val),
        "train_repeats": args.train_repeats,
        "base_train_limit": args.base_train_limit,
        "train": train_stats,
        "val": val_stats,
        "base_dataset": str(base),
    }
    (args.output / "build_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
