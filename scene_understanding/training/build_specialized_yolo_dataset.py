#!/usr/bin/env python3
"""Build a balanced, symlink-based YOLO dataset from BDD100K and nuScenes."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CLASSES = (
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcycle",
    "traffic_light",
    "traffic_sign",
    "road_barrier",
    "traffic_cone",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
BDD_TO_DRIVING = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "person": "pedestrian",
    "rider": "cyclist",
    "bike": "cyclist",
    "bicycle": "cyclist",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}
TRAIN_CLASS_WEIGHTS = {
    "vehicle": 1.0,
    "pedestrian": 3.0,
    "cyclist": 8.0,
    "motorcycle": 8.0,
    "traffic_light": 3.0,
    "traffic_sign": 3.0,
    "road_barrier": 5.0,
    "traffic_cone": 5.0,
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    frame_id: str
    image_path: Path
    boxes: tuple[tuple[int, float, float, float, float], ...]
    categories: frozenset[str]


def normalized_box(record: dict, annotation: dict) -> tuple[float, float, float, float]:
    if "bbox_2d" in annotation:
        x1, y1, x2, y2 = (float(value) for value in annotation["bbox_2d"])
    else:
        x1, y1, x2, y2 = (float(value) for value in annotation["bbox_xyxy"])
        x1, x2 = x1 / float(record["width"]), x2 / float(record["width"])
        y1, y2 = y1 / float(record["height"]), y2 / float(record["height"])
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def category_name(record: dict, annotation: dict) -> str | None:
    raw = str(annotation["category"]).lower()
    if str(record.get("dataset", "")).lower() == "bdd100k":
        return BDD_TO_DRIVING.get(raw)
    return raw if raw in CLASS_TO_ID else None


def load_candidates(
    paths: Iterable[Path], *, quality_counts: Counter[str]
) -> list[Candidate]:
    candidates = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                boxes = []
                categories = set()
                for annotation in record.get("annotations", []):
                    category = category_name(record, annotation)
                    if category is None:
                        continue
                    try:
                        x1, y1, x2, y2 = normalized_box(record, annotation)
                    except ValueError:
                        quality_counts["invalid_boxes_skipped"] += 1
                        continue
                    boxes.append((CLASS_TO_ID[category], x1, y1, x2, y2))
                    categories.add(category)
                if not boxes:
                    quality_counts["records_without_valid_target_skipped"] += 1
                    continue
                image_path = Path(record["image_path"])
                if not image_path.is_absolute():
                    image_path = (path.parent / image_path).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                candidates.append(
                    Candidate(
                        dataset=str(record.get("dataset") or record.get("source") or "unknown"),
                        frame_id=str(record.get("frame_id") or record.get("image_id")),
                        image_path=image_path,
                        boxes=tuple(boxes),
                        categories=frozenset(categories),
                    )
                )
    return candidates


def weighted_sample(
    candidates: list[Candidate], limit: int, *, seed: int
) -> list[Candidate]:
    if limit >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    heap: list[tuple[float, int, Candidate]] = []
    for index, candidate in enumerate(candidates):
        weight = 1.0 + sum(TRAIN_CLASS_WEIGHTS[name] for name in candidate.categories)
        key = rng.random() ** (1.0 / weight)
        item = (key, index, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif key > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, reverse=True)]


def uniform_sample(
    candidates: list[Candidate], limit: int, *, seed: int
) -> list[Candidate]:
    if limit >= len(candidates):
        return list(candidates)
    return random.Random(seed).sample(candidates, limit)


def write_split(output: Path, split: str, candidates: list[Candidate]) -> dict:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    record_dir = output / "records"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    frame_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()

    with (record_dir / f"{split}.jsonl").open("w", encoding="utf-8") as audit:
        for candidate in candidates:
            digest = hashlib.sha1(str(candidate.image_path).encode("utf-8")).hexdigest()[:20]
            stem = f"{candidate.dataset.lower()}_{digest}"
            image_link = image_dir / f"{stem}{candidate.image_path.suffix.lower()}"
            label_path = label_dir / f"{stem}.txt"
            if not image_link.exists():
                image_link.symlink_to(candidate.image_path)
            label_lines = []
            for class_id, x1, y1, x2, y2 in candidate.boxes:
                width, height = x2 - x1, y2 - y1
                center_x, center_y = x1 + width / 2, y1 + height / 2
                label_lines.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
                )
                box_counts[CLASSES[class_id]] += 1
            label_path.write_text("\n".join(label_lines) + "\n", encoding="ascii")
            frame_counts[candidate.dataset] += 1
            audit.write(
                json.dumps(
                    {
                        "dataset": candidate.dataset,
                        "frame_id": candidate.frame_id,
                        "image_path": str(candidate.image_path),
                        "image_link": str(image_link),
                        "label_path": str(label_path),
                        "categories": sorted(candidate.categories),
                        "box_count": len(candidate.boxes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "frames": len(candidates),
        "frames_by_dataset": dict(frame_counts),
        "boxes": sum(box_counts.values()),
        "boxes_by_category": dict(box_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-train", type=Path, required=True)
    parser.add_argument("--bdd-val", type=Path, required=True)
    parser.add_argument("--nuscenes-train", type=Path, action="append", required=True)
    parser.add_argument("--nuscenes-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bdd-train-limit", type=int, default=30_000)
    parser.add_argument("--nuscenes-train-limit", type=int, default=20_000)
    parser.add_argument("--bdd-val-limit", type=int, default=2_500)
    parser.add_argument("--nuscenes-val-limit", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    quality_counts: Counter[str] = Counter()
    bdd_train = load_candidates([args.bdd_train], quality_counts=quality_counts)
    nuscenes_train = load_candidates(
        args.nuscenes_train, quality_counts=quality_counts
    )
    bdd_val = load_candidates([args.bdd_val], quality_counts=quality_counts)
    nuscenes_val = load_candidates(
        [args.nuscenes_val], quality_counts=quality_counts
    )
    selected_train = [
        *weighted_sample(bdd_train, args.bdd_train_limit, seed=args.seed),
        *weighted_sample(
            nuscenes_train, args.nuscenes_train_limit, seed=args.seed + 1
        ),
    ]
    selected_val = [
        *uniform_sample(bdd_val, args.bdd_val_limit, seed=args.seed + 2),
        *uniform_sample(
            nuscenes_val, args.nuscenes_val_limit, seed=args.seed + 3
        ),
    ]
    random.Random(args.seed + 4).shuffle(selected_train)
    random.Random(args.seed + 5).shuffle(selected_val)

    train_inventory = write_split(args.output, "train", selected_train)
    val_inventory = write_split(args.output, "val", selected_val)
    dataset_yaml = [
        f"path: {args.output}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
        "",
    ]
    (args.output / "dataset.yaml").write_text("\n".join(dataset_yaml), encoding="utf-8")
    inventory = {
        "dataset_version": "specialized_yolo_v1",
        "seed": args.seed,
        "classes": list(CLASSES),
        "sampling": {
            "train": "weighted without replacement; rare safety classes emphasized",
            "val": "uniform without replacement; natural source distribution retained",
        },
        "quality_counts": dict(quality_counts),
        "train": train_inventory,
        "val": val_inventory,
        "source_manifests": {
            "bdd_train": str(args.bdd_train),
            "bdd_val": str(args.bdd_val),
            "nuscenes_train": [str(path) for path in args.nuscenes_train],
            "nuscenes_val": str(args.nuscenes_val),
        },
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
