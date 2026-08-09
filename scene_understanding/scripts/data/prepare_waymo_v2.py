#!/usr/bin/env python3
"""Convert Waymo v2 camera image/box parquet subsets to portable manifests."""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


IMAGE_COMPONENT = "[CameraImageComponent].image"
BOX_PREFIX = "[CameraBoxComponent]"
KEY_COLUMNS = (
    "key.segment_context_name",
    "key.frame_timestamp_micros",
    "key.camera_name",
)
TYPE_TO_CATEGORY = {1: "vehicle", 2: "pedestrian", 4: "cyclist"}
CAMERA_NAMES = {1: "front", 2: "front_left", 3: "front_right", 4: "side_left", 5: "side_right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("training", "validation"))
    return parser.parse_args()


def key(row: dict) -> tuple[str, int, int]:
    return tuple(row[name] for name in KEY_COLUMNS)  # type: ignore[return-value]


def read_boxes(path: Path) -> dict[tuple[str, int, int], list[dict]]:
    columns = [
        *KEY_COLUMNS,
        f"{BOX_PREFIX}.box.center.x",
        f"{BOX_PREFIX}.box.center.y",
        f"{BOX_PREFIX}.box.size.x",
        f"{BOX_PREFIX}.box.size.y",
        f"{BOX_PREFIX}.type",
        f"{BOX_PREFIX}.difficulty_level.detection",
    ]
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            grouped[key(row)].append(row)
    return grouped


def convert_split(input_root: Path, output: Path, split: str) -> dict:
    image_dir = input_root / split / "camera_image"
    box_dir = input_root / split / "camera_box"
    manifest_dir = output / "manifests"
    decoded_dir = output / "images" / split
    manifest_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    image_files = {path.stem: path for path in image_dir.glob("*.parquet")}
    box_files = {path.stem: path for path in box_dir.glob("*.parquet")}
    shared = sorted(image_files.keys() & box_files.keys())
    if not shared:
        raise FileNotFoundError(f"no matching camera_image/camera_box parquet in {input_root / split}")

    counters: Counter[str] = Counter()
    manifest_path = manifest_dir / f"{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as writer:
        for segment in shared:
            boxes_by_key = read_boxes(box_files[segment])
            columns = [*KEY_COLUMNS, IMAGE_COMPONENT]
            for batch in pq.ParquetFile(image_files[segment]).iter_batches(
                columns=columns, batch_size=16
            ):
                for image_row in batch.to_pylist():
                    frame_key = key(image_row)
                    encoded = image_row[IMAGE_COMPONENT]
                    with Image.open(io.BytesIO(encoded)) as image:
                        width, height = image.size
                    annotations = []
                    for box in boxes_by_key.get(frame_key, []):
                        category = TYPE_TO_CATEGORY.get(int(box[f"{BOX_PREFIX}.type"]))
                        if category is None:
                            counters["unsupported_boxes"] += 1
                            continue
                        center_x = float(box[f"{BOX_PREFIX}.box.center.x"])
                        center_y = float(box[f"{BOX_PREFIX}.box.center.y"])
                        size_x = float(box[f"{BOX_PREFIX}.box.size.x"])
                        size_y = float(box[f"{BOX_PREFIX}.box.size.y"])
                        x1 = max(0.0, center_x - size_x / 2.0)
                        y1 = max(0.0, center_y - size_y / 2.0)
                        x2 = min(float(width), center_x + size_x / 2.0)
                        y2 = min(float(height), center_y + size_y / 2.0)
                        if x2 <= x1 or y2 <= y1:
                            counters["invalid_boxes"] += 1
                            continue
                        annotations.append(
                            {
                                "category": category,
                                "category_raw": int(box[f"{BOX_PREFIX}.type"]),
                                "bbox_xyxy": [x1, y1, x2, y2],
                                "bbox_2d": [x1 / width, y1 / height, x2 / width, y2 / height],
                                "difficulty_detection": int(
                                    box[f"{BOX_PREFIX}.difficulty_level.detection"] or 0
                                ),
                            }
                        )
                        counters[f"boxes:{category}"] += 1
                    camera = int(image_row["key.camera_name"])
                    timestamp = int(image_row["key.frame_timestamp_micros"])
                    image_path = decoded_dir / f"{segment}_{timestamp}_{camera}.jpg"
                    if not image_path.exists():
                        image_path.write_bytes(encoded)
                    relative_image = os.path.relpath(image_path, manifest_dir)
                    writer.write(
                        json.dumps(
                            {
                                "manifest_version": "1.0",
                                "dataset": "Waymo Open Dataset",
                                "source": "waymo_v2",
                                "split": split,
                                "frame_id": f"{segment}_{timestamp}_{camera}",
                                "segment_context_name": segment,
                                "timestamp_micros": timestamp,
                                "camera_name": CAMERA_NAMES.get(camera, str(camera)),
                                "image_path": relative_image,
                                "width": width,
                                "height": height,
                                "annotations": annotations,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counters["frames"] += 1
                    counters["frames_with_targets"] += int(bool(annotations))
            counters["segments"] += 1
    return dict(sorted(counters.items()))


def main() -> None:
    args = parse_args()
    splits = args.split or ["training", "validation"]
    summary = {
        "schema_version": "waymo_v2_camera_manifest/1.0",
        "source_root": os.path.relpath(args.input_root, Path.cwd()),
        "splits": {split: convert_split(args.input_root, args.output, split) for split in splits},
        "type_mapping": {str(key): value for key, value in TYPE_TO_CATEGORY.items()},
    }
    (args.output / "inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
