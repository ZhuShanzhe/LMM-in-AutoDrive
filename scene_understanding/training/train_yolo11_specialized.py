#!/usr/bin/env python3
"""Fine-tune a YOLO11 checkpoint for the shared driving taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="yolo11n_specialized_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fraction", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    run_dir = args.project / args.name
    if run_dir.exists():
        raise SystemExit(f"training run already exists: {run_dir}")
    args.project.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(args.data),
        "model": str(args.model),
        "project": str(args.project),
        "name": args.name,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.image_size,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "fraction": args.fraction,
        "deterministic": True,
        "pretrained": True,
        "amp": True,
        "patience": 10,
        "cos_lr": True,
        "close_mosaic": 5,
        "mosaic": 0.75,
        "mixup": 0.05,
        "plots": True,
        "save": True,
        "verbose": True,
    }
    (args.project / f"{args.name}_launch.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    model = YOLO(str(args.model))
    result = model.train(**config)
    print(result)


if __name__ == "__main__":
    main()
