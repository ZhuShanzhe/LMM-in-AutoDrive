#!/usr/bin/env python3
"""Merge per-camera nuScenes inventories after parallel preparation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CAMERAS = (
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_back",
    "cam_back_left",
    "cam_back_right",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parts = []
    for camera in CAMERAS:
        path = args.output / f"inventory_{camera}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(json.loads(path.read_text(encoding="utf-8")))
    merged = {
        "dataset": "nuScenes",
        "version": parts[0]["version"],
        "dataroot": parts[0]["dataroot"],
        "official_scene_splits": parts[0]["official_scene_splits"],
        "samples": parts[0]["samples"],
        "sample_annotations_3d": parts[0]["sample_annotations_3d"],
        "cameras": [part["cameras"][0] for part in parts],
        "manifest_frame_counts": {},
        "projected_annotation_counts": {},
        "contents": parts[0]["contents"],
    }
    category_counts: Counter[str] = Counter()
    for part in parts:
        merged["manifest_frame_counts"].update(part["manifest_frame_counts"])
        category_counts.update(part["projected_annotation_counts"])
    merged["projected_annotation_counts"] = dict(sorted(category_counts.items()))
    (args.output / "inventory.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
