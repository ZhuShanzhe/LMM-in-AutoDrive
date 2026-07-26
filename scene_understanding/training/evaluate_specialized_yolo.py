"""Evaluate a specialized YOLO detector and save per-class metrics as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        split="val",
        device=args.device,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        plots=False,
        verbose=False,
    )
    names = metrics.names
    per_class = {}
    for class_id in sorted(names):
        per_class[str(names[class_id])] = {
            "precision": round(float(metrics.box.p[class_id]), 6),
            "recall": round(float(metrics.box.r[class_id]), 6),
            "map50": round(float(metrics.box.ap50[class_id]), 6),
            "map50_95": round(float(metrics.box.maps[class_id]), 6),
        }
    result = {
        "weights": str(args.weights),
        "data": str(args.data),
        "overall": {
            key: round(float(value), 6)
            for key, value in metrics.results_dict.items()
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: round(float(value), 6) for key, value in metrics.speed.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
