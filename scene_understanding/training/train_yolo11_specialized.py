#!/usr/bin/env python3
"""Fine-tune a YOLO11 checkpoint for the shared driving taxonomy."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze this many leading model layers (0 trains every layer).",
    )
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    data_path = args.data.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    project_path = args.project.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    run_dir = project_path / args.name
    if run_dir.exists():
        raise SystemExit(f"training run already exists: {run_dir}")
    project_path.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(data_path),
        "model": str(model_path),
        "project": str(project_path),
        "name": args.name,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.image_size,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "fraction": args.fraction,
        "lr0": args.learning_rate,
        "freeze": args.freeze,
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
    (project_path / f"{args.name}_launch.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    model = YOLO(str(model_path))
    # Ultralytics resolves a portable YAML ``path: .`` against the process
    # working directory, not the YAML location.  Run from the dataset root so
    # relative image paths remain portable across Linux machines.
    original_working_directory = Path.cwd()
    try:
        os.chdir(data_path.parent)
        result = model.train(**config)
    finally:
        os.chdir(original_working_directory)
    print(result)


if __name__ == "__main__":
    main()
