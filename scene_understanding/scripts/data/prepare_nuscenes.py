#!/usr/bin/env python3
"""Build readable train/val camera manifests from full nuScenes trainval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def driving_category(category: str) -> str:
    if category.startswith("vehicle.car") or category.startswith("vehicle.bus"):
        return "vehicle"
    if category.startswith(("vehicle.truck", "vehicle.trailer", "vehicle.construction")):
        return "vehicle"
    if category.startswith("human.pedestrian"):
        return "pedestrian"
    if category == "vehicle.bicycle":
        return "cyclist"
    if category == "vehicle.motorcycle":
        return "motorcycle"
    if category == "movable_object.trafficcone":
        return "traffic_cone"
    if category == "movable_object.barrier":
        return "road_barrier"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append", choices=CAMERAS)
    return parser.parse_args()


def main() -> None:
    from nuscenes.nuscenes import NuScenes
    import nuscenes.scripts.export_2d_annotations_as_json as exporter
    from nuscenes.utils import splits

    args = parse_args()
    cameras = tuple(args.camera or CAMERAS)
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = args.output / "manifests"
    manifests.mkdir(exist_ok=True)

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(args.dataroot), verbose=False)
    exporter.nusc = nusc
    split_names = {
        **{name: "train" for name in splits.train},
        **{name: "val" for name in splits.val},
    }
    scenes = {item["token"]: item for item in nusc.scene}
    logs = {item["token"]: item for item in nusc.log}
    attributes = {item["token"]: item["name"] for item in nusc.attribute}
    writers = {
        (split, camera): (manifests / f"{split}_{camera.lower()}.jsonl").open(
            "w", encoding="utf-8"
        )
        for split in ("train", "val")
        for camera in cameras
    }
    frame_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    try:
        for sample in nusc.sample:
            scene = scenes[sample["scene_token"]]
            split = split_names.get(scene["name"])
            if split is None:
                continue
            log = logs[scene["log_token"]]
            for camera in cameras:
                sample_data_token = sample["data"][camera]
                sample_data = nusc.get("sample_data", sample_data_token)
                projected = exporter.get_2d_boxes(
                    sample_data_token, visibilities=["1", "2", "3", "4", ""]
                )
                annotations = []
                for item in projected:
                    category = driving_category(item["category_name"])
                    x1, y1, x2, y2 = item["bbox_corners"]
                    annotations.append(
                        {
                            "sample_annotation_token": item["sample_annotation_token"],
                            "instance_token": item["instance_token"],
                            "category_raw": item["category_name"],
                            "category": category,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_2d": [
                                x1 / sample_data["width"],
                                y1 / sample_data["height"],
                                x2 / sample_data["width"],
                                y2 / sample_data["height"],
                            ],
                            "visibility_token": item["visibility_token"],
                            "attribute_names": [
                                attributes[token] for token in item["attribute_tokens"]
                            ],
                            "num_lidar_pts": item["num_lidar_pts"],
                            "num_radar_pts": item["num_radar_pts"],
                        }
                    )
                    category_counts[f"{split}:{category}"] += 1
                record = {
                    "manifest_version": "1.0",
                    "dataset": "nuScenes",
                    "split": split,
                    "frame_id": f"{sample['token']}_{sample_data_token}",
                    "sample_token": sample["token"],
                    "sample_data_token": sample_data_token,
                    "scene_token": scene["token"],
                    "scene_name": scene["name"],
                    "location": log["location"],
                    "source": "nuscenes",
                    "camera_name": camera,
                    "timestamp_s": sample_data["timestamp"] / 1_000_000,
                    "image_path": str(args.dataroot / sample_data["filename"]),
                    "width": sample_data["width"],
                    "height": sample_data["height"],
                    "annotations": annotations,
                }
                writers[(split, camera)].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                frame_counts[f"{split}:{camera}"] += 1
    finally:
        for writer in writers.values():
            writer.close()

    inventory = {
        "dataset": "nuScenes",
        "version": "v1.0-trainval",
        "dataroot": str(args.dataroot),
        "official_scene_splits": {"train": len(splits.train), "val": len(splits.val)},
        "samples": len(nusc.sample),
        "sample_annotations_3d": len(nusc.sample_annotation),
        "cameras": list(cameras),
        "manifest_frame_counts": dict(sorted(frame_counts.items())),
        "projected_annotation_counts": dict(sorted(category_counts.items())),
        "contents": [
            "six-camera keyframe manifests with official train/val scene split",
            "official 3D boxes projected to normalized 2D camera boxes",
            "raw samples, sweeps, maps, prediction and trainval metadata remain in dataroot",
        ],
    }
    inventory_name = (
        f"inventory_{cameras[0].lower()}.json" if len(cameras) == 1 else "inventory.json"
    )
    (args.output / inventory_name).write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
