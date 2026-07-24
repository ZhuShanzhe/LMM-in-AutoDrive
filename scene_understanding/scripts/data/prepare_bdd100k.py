#!/usr/bin/env python3
"""Extract BDD100K Parquet shards into images plus readable JSONL manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "image_id",
    "split",
    "image_bytes",
    "width",
    "height",
    "weather",
    "scene",
    "timeofday",
    "ann_categories",
    "ann_bboxes",
    "ann_occluded",
    "ann_truncated",
    "ann_traffic_light_colors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80_500)
    return parser.parse_args()


def image_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def serializable(value: object) -> object:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def main() -> None:
    args = parse_args()
    shards = sorted(args.source.glob("default/train/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards below {args.source}")

    manifests = args.output / "manifests"
    images = args.output / "images"
    manifests.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    writers: dict[str, object] = {}

    try:
        for shard in shards:
            # The mirror's optional embedding column contains a few empty vectors
            # despite a fixed-size schema. It is irrelevant to perception training.
            table = pq.read_table(shard, columns=REQUIRED_COLUMNS)
            for row in table.to_pylist():
                split = str(row.get("split") or "train")
                image_id = str(row["image_id"])
                image_path = images / split / f"{image_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not image_path.exists():
                    raw = image_bytes(row["image_bytes"])
                    image_path.write_bytes(raw)

                annotations = []
                categories = row.get("ann_categories") or []
                boxes = row.get("ann_bboxes") or []
                occluded = row.get("ann_occluded") or []
                truncated = row.get("ann_truncated") or []
                colors = row.get("ann_traffic_light_colors") or []
                for index, category in enumerate(categories):
                    category_counts[str(serializable(category))] += 1
                    annotations.append(
                        {
                            "category": serializable(category),
                            "bbox_xyxy": serializable(boxes[index]),
                            "occluded": bool(occluded[index]) if index < len(occluded) else None,
                            "truncated": bool(truncated[index]) if index < len(truncated) else None,
                            "traffic_light_color": (
                                serializable(colors[index]) if index < len(colors) else None
                            ),
                        }
                    )

                record = {
                    "dataset": "BDD100K",
                    "split": split,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "weather": row.get("weather"),
                    "scene": row.get("scene"),
                    "timeofday": row.get("timeofday"),
                    "annotations": annotations,
                }
                if split not in writers:
                    writers[split] = (manifests / f"{split}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    total_count = sum(counts.values())
    if total_count != args.expected_count:
        raise RuntimeError(
            f"incomplete BDD100K extraction: found {total_count}/{args.expected_count} rows"
        )

    inventory = {
        "dataset": "BDD100K",
        "source": "lance-format/BDD100K-enriched",
        "source_revision": "refs/convert/parquet",
        "total_images": total_count,
        "counts": dict(counts),
        "annotation_category_counts": dict(category_counts.most_common()),
        "task_scope": ["2d_detection", "traffic_light_state", "domain_generalization"],
        "limitations": [
            "This mirror does not contain lane or drivable-area ground truth.",
            "Use the official ETH label archives when the endpoint is available.",
        ],
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
