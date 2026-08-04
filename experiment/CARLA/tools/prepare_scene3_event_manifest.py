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


def portable_image_path(path: Path) -> str:
    """Use a repository-relative path when the image is stored in the repo."""
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _observed_target_choice(
    ground_truth: list[dict[str, Any]],
    event_id: str,
    *,
    target_distance_m: float,
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, str, float, float] | None]:
    """Choose an observed event frame whose evidence actor is near the target range.

    The relative-position test is deliberately conservative, but it is still a
    geometry proxy: it does not claim that the actor is unoccluded in the RGB
    image.  Keeping that distinction in the manifest prevents simulator truth
    from being mistaken for an image-space annotation.
    """
    choices = []
    fallbacks = []
    for row in ground_truth:
        event = next(
            (
                item
                for item in row.get("active_events", [])
                if item.get("event_id") == event_id
            ),
            None,
        )
        if event is None:
            continue
        fallbacks.append((row, event))
        for role in event.get("evidence", {}).get("observed", []):
            relation = (
                row.get("actors", {})
                .get(role, {})
                .get("relation_to_ego", {})
            )
            longitudinal = relation.get("longitudinal_m")
            lateral = relation.get("lateral_m")
            distance = relation.get("euclidean_distance_m")
            if None in (longitudinal, lateral, distance):
                continue
            longitudinal = float(longitudinal)
            lateral = float(lateral)
            distance = float(distance)
            if longitudinal < 5.0 or abs(lateral) > max(8.0, longitudinal):
                continue
            actor = (distance, role, longitudinal, lateral)
            choices.append((abs(distance - target_distance_m), row, event, actor))
    if choices:
        _, row, event, actor = min(choices, key=lambda item: item[0])
        return row, event, actor
    if not fallbacks:
        raise ValueError(f"no active ground-truth frames found for {event_id}")
    row, event = fallbacks[0]
    return row, event, None


def build_records(
    run_dir: Path,
    prompt_path: Path,
    *,
    maximum_frame_gap: int,
    camera_name: str = "chase_rgb",
    image_dir: Path | None = None,
    selection: str = "activation",
    ground_truth_path: Path | None = None,
    target_distance_m: float = 20.0,
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
    ground_truth = None
    if selection == "observed-target":
        ground_truth_path = ground_truth_path or (run_dir / "frame_ground_truth.jsonl")
        ground_truth = read_jsonl(ground_truth_path)
    records = []
    for event in active_events:
        activation_frame = int(event["simulation_frame"])
        selected_actor = None
        if ground_truth is not None:
            truth_row, truth_event, actor = _observed_target_choice(
                ground_truth,
                event["event_id"],
                target_distance_m=target_distance_m,
            )
            event_frame = int(truth_row["simulation_frame"])
            selected_actor = None if actor is None else {
                "role": actor[1],
                "distance_m": actor[0],
                "longitudinal_m": actor[2],
                "lateral_m": actor[3],
            }
            scenario = truth_event.get("scenario", event["scenario"])
            route_s_m = truth_row["route_s_m"]
            elapsed_s = truth_row["timestamp_s"]
        else:
            event_frame = activation_frame
            scenario = event["scenario"]
            route_s_m = event["route_s_m"]
            elapsed_s = event["elapsed_s"]
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
                "image_path": portable_image_path(image_path),
                "prompt": (
                    prompt_template
                    .replace("{frame_id}", frame_id)
                    .replace("{source}", "carla")
                    .replace("{camera_name}", camera_name)
                ),
                "scene3_event": {
                    "event_id": event["event_id"],
                    "scenario": scenario,
                    "voice_command_id": event["voice_command_id"],
                    "selection": selection,
                    "activation_frame": activation_frame,
                    "ground_truth_frame": (
                        event_frame if ground_truth is not None else None
                    ),
                    "selected_image_frame": image_frame,
                    "frame_gap": gap,
                    "selected_actor": selected_actor,
                    "route_s_m": route_s_m,
                    "elapsed_s": elapsed_s,
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
        "--selection",
        choices=("activation", "observed-target"),
        default="activation",
        help=(
            "activation selects the event start; observed-target selects a "
            "ground-truth frame with an observed evidence actor near the target range"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Ground-truth JSONL; defaults to RUN_DIR/frame_ground_truth.jsonl",
    )
    parser.add_argument("--target-distance-m", type=float, default=20.0)
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
        selection=args.selection,
        ground_truth_path=(
            args.ground_truth.resolve() if args.ground_truth else None
        ),
        target_distance_m=args.target_distance_m,
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
