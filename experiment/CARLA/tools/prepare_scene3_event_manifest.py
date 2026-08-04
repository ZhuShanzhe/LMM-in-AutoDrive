#!/usr/bin/env python3
"""Build a frame-aligned VLM manifest from a completed Scene 3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT = REPO_ROOT / "scene_understanding" / "prompts" / "scene_understanding.txt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_records(
    run_dir: Path,
    prompt_path: Path,
    *,
    maximum_frame_gap: int,
    camera_name: str = "chase_rgb",
    image_dir: Path | None = None,
) -> list[dict[str, Any]]:
    active_events = [
        row
        for row in read_jsonl(run_dir / "event_timeline.jsonl")
        if row.get("state") == "ACTIVE"
    ]
    image_dir = image_dir or (run_dir / "camera_frames")
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise ValueError(f"no camera frames found under {image_dir}")
    frame_images = [(int(path.stem), path.resolve()) for path in images]
    prompt_template = prompt_path.read_text(encoding="utf-8")
    records = []
    for event in active_events:
        event_frame = int(event["simulation_frame"])
        image_frame, image_path = min(
            frame_images,
            key=lambda item: abs(item[0] - event_frame),
        )
        gap = abs(image_frame - event_frame)
        if gap > maximum_frame_gap:
            raise ValueError(
                f"nearest image for {event['event_id']} is {gap} frames away; "
                f"limit is {maximum_frame_gap}"
            )
        frame_id = f"carla_{image_frame}"
        records.append(
            {
                "frame_id": frame_id,
                "source": "carla",
                "camera_name": camera_name,
                "image_path": str(image_path),
                "prompt": (
                    prompt_template
                    .replace("{frame_id}", frame_id)
                    .replace("{source}", "carla")
                    .replace("{camera_name}", camera_name)
                ),
                "scene3_event": {
                    "event_id": event["event_id"],
                    "scenario": event["scenario"],
                    "voice_command_id": event["voice_command_id"],
                    "event_frame": event_frame,
                    "selected_image_frame": image_frame,
                    "frame_gap": gap,
                    "route_s_m": event["route_s_m"],
                    "elapsed_s": event["elapsed_s"],
                },
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--maximum-frame-gap", type=int, default=120)
    parser.add_argument("--camera-name", default="chase_rgb")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Camera image directory; defaults to RUN_DIR/camera_frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_records(
        args.run_dir.resolve(),
        args.prompt.resolve(),
        maximum_frame_gap=args.maximum_frame_gap,
        camera_name=args.camera_name,
        image_dir=args.image_dir.resolve() if args.image_dir else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} event records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
